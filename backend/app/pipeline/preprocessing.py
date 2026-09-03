import re
from urllib.parse import urlparse

_WORD_RE = re.compile(r"[a-zA-Z']+")

_lemmatizer = None
_stopwords = None


def _load_nltk():
    """Lazily load NLTK's lemmatizer/stopwords, downloading corpora on first use.

    Falls back to no-op behavior if NLTK data can't be fetched (e.g. no network),
    so preprocessing degrades gracefully instead of crashing the pipeline.
    """
    global _lemmatizer, _stopwords
    if _lemmatizer is not None:
        return
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        for resource in ("corpora/stopwords", "corpora/wordnet", "corpora/omw-1.4"):
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(resource.split("/")[-1], quiet=True)

        _lemmatizer = WordNetLemmatizer()
        _stopwords = set(stopwords.words("english"))
    except Exception:
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
    return urlparse(normalize_url(url)).netloc.split(":")[0]


def preprocess_email_text(text: str) -> list[str]:
    """Tokenize, remove stop-words, and lemmatize email body text for
    TF-IDF / classical-model feature extraction."""
    _load_nltk()
    tokens = [t.lower() for t in _WORD_RE.findall(text)]

    if _lemmatizer:
        tokens = [t for t in tokens if t not in _stopwords]
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    else:
        # minimal fallback stop-word list when NLTK data is unavailable
        basic_stopwords = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "for", "on", "it"}
        tokens = [t for t in tokens if t not in basic_stopwords]

    return tokens
