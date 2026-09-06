import re
from urllib.parse import urlparse

from app.pipeline.preprocessing import normalize_url

_IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "facebook", "netflix",
    "bank", "chase", "wellsfargo", "irs", "dhl", "fedex", "linkedin", "instagram",
]
_SUSPICIOUS_TLDS = {"zip", "mov", "top", "xyz", "tk", "gq", "ml"}


def extract_url_features(url: str) -> dict:
    """Structured features for the RF/XGBoost classifiers, per section 2 of the spec:
    length, subdomain count, IP-as-domain, special chars, brand keywords."""
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    # .hostname, not .netloc.split(":")[0] -- see preprocessing.get_domain()'s
    # comment for why the naive split is wrong (userinfo confusion: it takes
    # the userinfo username, not the real host, for "user:pass@host" netlocs).
    host = parsed.hostname or ""
    host_labels = host.split(".")

    is_ip = bool(_IP_RE.match(host))
    subdomain_count = max(len(host_labels) - 2, 0) if not is_ip else 0
    tld = host_labels[-1] if len(host_labels) > 1 else ""

    full_lower = normalized.lower()
    brand_hits = [b for b in _BRAND_KEYWORDS if b in full_lower]
    # a brand keyword appearing outside the registrable domain (e.g. in a
    # subdomain or path) is a classic impersonation pattern.
    registrable_domain = ".".join(host_labels[-2:]) if len(host_labels) >= 2 else host
    brand_in_domain = any(b in registrable_domain for b in brand_hits)

    # normalize_url() always adds a scheme when one is missing, so "://" is
    # always present here -- stripping it keeps url_length/special_char_count/
    # digit_count invariant to whether the caller happened to type "https://",
    # "http://", or nothing. Without this, these three features silently
    # encoded scheme-presence the same way has_https did (see ml/README.md
    # section 3.5): this dataset's raw URLs are ~99.5% schemeless, so any real
    # user pasting a full "https://..." URL got systematically pushed toward
    # "phishing" purely by those extra 8 characters, regardless of content.
    content = normalized.split("://", 1)[1]

    return {
        "url_length": len(content),
        "host_length": len(host),
        "subdomain_count": subdomain_count,
        "is_ip_address": is_ip,
        "has_https": parsed.scheme == "https",
        "special_char_count": sum(content.count(c) for c in ["@", "-", "_", "%", "="]),
        "digit_count": sum(c.isdigit() for c in content),
        # a bare trailing "/" and no path at all are the same "no path" case --
        # counting them differently just encodes which dataset a URL string came
        # from (whether it was stored with a trailing slash), not real signal.
        "path_length": len(parsed.path.rstrip("/")),
        "query_length": len(parsed.query),
        "brand_keyword_count": len(brand_hits),
        "brand_keyword_outside_domain": bool(brand_hits) and not brand_in_domain,
        "suspicious_tld": tld in _SUSPICIOUS_TLDS,
        "has_at_symbol": "@" in url,
    }
