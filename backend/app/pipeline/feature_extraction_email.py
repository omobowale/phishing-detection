import re
from email import message_from_string
from email.utils import parseaddr

from app.pipeline.preprocessing import preprocess_email_text, strip_email_headers

_URGENCY_KEYWORDS = [
    "urgent", "verify your account", "suspended", "click here", "act now",
    "password expire", "confirm your identity", "unusual activity", "limited time",
]
_URL_RE = re.compile(r"https?://[^\s<>\"']+")


def _extract_header_features(raw_text: str) -> dict:
    """SPF/DKIM/DMARC + reply-to/from mismatch, per section 2 of the spec.

    Only meaningful when the submitted email_text includes raw headers (as you'd
    get from a .eml export); falls back to neutral/unknown values for plain body
    text, since that's what most end users will paste in.
    """
    msg = message_from_string(raw_text)
    auth_results = msg.get("Authentication-Results", "") or ""

    from_addr = parseaddr(msg.get("From", ""))[1]
    reply_to_addr = parseaddr(msg.get("Reply-To", ""))[1]
    reply_to_mismatch = bool(reply_to_addr) and bool(from_addr) and (
        reply_to_addr.split("@")[-1].lower() != from_addr.split("@")[-1].lower()
    )

    received_hops = len(msg.get_all("Received", []) or [])

    return {
        "has_headers": bool(msg.get("From") or msg.get("Subject")),
        "spf_pass": "spf=pass" in auth_results.lower(),
        "dkim_pass": "dkim=pass" in auth_results.lower(),
        "dmarc_pass": "dmarc=pass" in auth_results.lower(),
        "reply_to_mismatch": reply_to_mismatch,
        "received_hop_count": received_hops,
    }


def extract_email_features(email_text: str) -> dict:
    """Header features + lightweight NLP signals for the classical models.
    Real semantic scoring comes from the fine-tuned BERT model on the tokenized
    body (see preprocess_email_text / classifiers.py)."""
    header_features = _extract_header_features(email_text)

    body = strip_email_headers(email_text)

    lower_body = body.lower()
    tokens = preprocess_email_text(body)
    urls_in_body = _URL_RE.findall(body)

    nlp_features = {
        "token_count": len(tokens),
        "urgency_keyword_count": sum(1 for kw in _URGENCY_KEYWORDS if kw in lower_body),
        "url_count_in_body": len(urls_in_body),
        "exclamation_count": body.count("!"),
        "all_caps_word_count": sum(1 for w in re.findall(r"\b[A-Z]{3,}\b", body)),
    }

    return {**header_features, **nlp_features, "urls_in_body": urls_in_body}
