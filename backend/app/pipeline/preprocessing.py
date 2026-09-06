import logging
import re
from urllib.parse import urlparse

_WORD_RE = re.compile(r"[a-zA-Z']+")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")

_lemmatizer = None
_stopwords = None
_logger = logging.getLogger("phishing_detection.preprocessing")


def _load_nltk():
    """Lazily load NLTK's lemmatizer/stopwords, if already downloaded.

    Deliberately does NOT call nltk.download() here -- that used to run
    synchronously inside a request (a whitespace-only email request measured
    ~1.4-2.4s instead of a few ms, entirely spent attempting a network
    download). Corpora must be fetched ahead of time via
    `python -m scripts.download_nltk_data` (run once during setup/deploy);
    if they're missing, this falls back to a minimal built-in stop-word list
    instead of touching the network mid-request.
    """
    global _lemmatizer, _stopwords
    if _lemmatizer is not None:
        return
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        for resource in ("corpora/stopwords", "corpora/wordnet", "corpora/omw-1.4"):
            nltk.data.find(resource)

        _lemmatizer = WordNetLemmatizer()
        _stopwords = set(stopwords.words("english"))
    except Exception:
        _logger.warning(
            "NLTK corpora not found -- falling back to a minimal stop-word list. "
            "Run 'python -m scripts.download_nltk_data' once to fetch them."
        )
        _lemmatizer = False
        _stopwords = set()


def normalize_url(url: str) -> str:
    """Lowercase, strip whitespace, and ensure a scheme is present so urlparse
    reliably splits netloc/path for feature extraction."""
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url.lower()


def get_domain(url: str) -> str:
    # .hostname (not .netloc.split(":")[0]) is required here: netloc can contain
    # userinfo ("user:pass@host"), and splitting on ":" naively takes the
    # userinfo's username for a URL like "https://trusted.com:password@evil.example/"
    # -- returning "trusted.com" instead of the real host "evil.example". That
    # was a confirmed, live whitelist bypass (an attacker-controlled host with a
    # trusted domain name stuffed into the userinfo). .hostname parses per the
    # URL spec and also normalizes case, so no extra .lower() is needed.
    return urlparse(normalize_url(url)).hostname or ""


def strip_email_headers(text: str) -> str:
    """If `text` is a raw RFC822 message (starts with real header lines),
    return just the body payload instead of feeding "Return-Path:",
    "Received:", etc. lines into NLP feature extraction as if they were
    email content. Shared by feature_extraction_email.py (live serving) and
    ml/training/build_email_features.py (training) so both see email text the
    same way -- an earlier version of the training script called
    preprocess_email_text() directly on raw CSV text without this step,
    while live serving already had it, a real train/serve mismatch. Falls
    back to the original text unchanged if there's no header structure to
    strip (the common case: most input is plain body text, not a full raw
    message) -- email.message_from_string() only recognizes headers that
    start on the very first line, so text with any leading prose is returned
    unchanged, not mangled."""
    from email import message_from_string

    try:
        msg = message_from_string(text)
        payload = msg.get_payload()
        if isinstance(payload, str) and payload.strip():
            return payload
    except Exception:
        pass
    return text


def redact_email_and_urls(text: str) -> str:
    """Replace email addresses and URLs with placeholder tokens BEFORE any
    further processing. Confirmed why this matters: in
    ml/training/build_email_features.py's training data, a trained model's
    top features included "jose"/"monkey" -- fragments of "jose@monkey.org",
    the address every phishing example from one source happened to be sent
    to, since they were all collected from one person's mailbox. Left
    unredacted, a classifier can key on a specific recipient's address rather
    than general phishing language, which won't generalize to email
    addressed to anyone else. Standard practice in spam/phishing text
    classification, not a one-off patch for this dataset.

    Shared by preprocess_email_text() (classical TF-IDF pipeline, which
    lowercases/tokenizes/lemmatizes afterward) and
    ml/training/train_email_bert.py (BERT fine-tuning, which wants natural
    casing/punctuation preserved for its own subword tokenizer) so both apply
    the identical fix rather than two independently-maintained copies."""
    text = _EMAIL_RE.sub(" emailaddresstoken ", text)
    text = _URL_RE.sub(" urltoken ", text)
    return text


def preprocess_email_text(text: str) -> list[str]:
    """Tokenize, remove stop-words, and lemmatize email body text for
    TF-IDF / classical-model feature extraction."""
    _load_nltk()
    text = redact_email_and_urls(text)
    tokens = [t.lower() for t in _WORD_RE.findall(text)]

    if _lemmatizer:
        tokens = [t for t in tokens if t not in _stopwords]
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    else:
        # minimal fallback stop-word list when NLTK data is unavailable
        basic_stopwords = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "for", "on", "it"}
        tokens = [t for t in tokens if t not in basic_stopwords]

    return tokens
