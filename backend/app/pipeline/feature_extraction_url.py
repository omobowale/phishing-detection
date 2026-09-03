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
    host = parsed.netloc.split(":")[0]
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

    return {
        "url_length": len(url),
        "host_length": len(host),
        "subdomain_count": subdomain_count,
        "is_ip_address": is_ip,
        "has_https": parsed.scheme == "https",
        "special_char_count": sum(url.count(c) for c in ["@", "-", "_", "%", "="]),
        "digit_count": sum(c.isdigit() for c in url),
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
        "brand_keyword_count": len(brand_hits),
        "brand_keyword_outside_domain": bool(brand_hits) and not brand_in_domain,
        "suspicious_tld": tld in _SUSPICIOUS_TLDS,
        "has_at_symbol": "@" in url,
    }
