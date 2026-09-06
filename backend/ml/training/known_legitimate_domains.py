"""Root domains used to clean provably-wrong "phishing" labels out of the raw
Kaggle dataset before training (see ml/README.md section 3.4 for the full
writeup and evidence).

Rationale: a "phishing" label on the literal apex domain (or a subdomain) of a
corporate/editorial site with no public user-generated-content or free-hosting
feature is not an edge case or a debatable judgment call -- it is impossible.
An attacker cannot publish a credential-harvesting page at
`research.microsoft.com/~cmbishop/` or `github.com/joyent/node/wiki` without a
full domain compromise, which is not what mass phishing datasets are
capturing. This is a stronger claim than "domain membership implies the label
is wrong" -- it specifically requires the domain to have no plausible
attacker-controlled hosting surface at all, checked domain by domain below.

Audit performed (not just a couple of examples): every one of the ~30 domains
below was individually queried against the raw CSV and 3+ sample dropped URLs
per domain were read manually, not just the two largest offenders. Findings:
`microsoft.com` is 74.7% labeled "phishing" (366/490 rows), `github.com` is
90.9% (30/33 rows, though github.com is NOT in this list -- see exclusions
below), and every one of the ~1,785 rows this drops across all 30 domains that
was inspected was an ordinary real page: product pages, support docs, press
releases, API references, encyclopedia articles. One caveat found during this
audit and worth flagging rather than hiding: `salesforce.com`'s 4 dropped rows
are on `*.my.salesforce.com`, which is a per-customer CRM instance subdomain --
not free public hosting like Google Forms, but also not purely static
corporate content the way microsoft.com/wikipedia.org are. Kept in the list
since provisioning one requires an actual paid org account (not something a
random attacker can freely spin up), but it's the least clear-cut entry here.

Deliberately EXCLUDED from this list: domains with a genuine free-hosting or
user-generated-content surface where trusted-domain abuse is a real, known
phishing technique (Google Forms/Sites/Docs, GitHub user pages, WordPress.com,
Blogspot, Weebly, Wix, Tumblr, Dropbox share links, etc.). Rows labeled
"phishing" on those hosts are NOT auto-cleaned -- they may be genuine examples
of exactly the attack pattern this app should learn to be cautious about, and
indiscriminately stripping them would remove real signal, not just noise.

This list only removes false "phishing" labels; it does not add anything back
as a whitelist or otherwise change runtime behavior. It has no relationship to
the admin-managed Whitelist feature (app/models/whitelist.py) -- that's a
separate, live, admin-curated mechanism for a different purpose.
"""

NO_PHISHING_POSSIBLE_DOMAINS = [
    # major tech / corporate, no public content-hosting surface
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "ibm.com",
    "oracle.com",
    "intel.com",
    "nvidia.com",
    "samsung.com",
    "sony.com",
    "dell.com",
    "hp.com",
    "cisco.com",
    "adobe.com",
    "salesforce.com",
    # encyclopedic / editorial / reference, moderated content only
    "wikipedia.org",
    "wikimedia.org",
    "nytimes.com",
    "bbc.com",
    "cnn.com",
    "forbes.com",
    "bloomberg.com",
    "espn.com",
    "imdb.com",
    # government / international orgs
    "irs.gov",
    "usa.gov",
    "nasa.gov",
    "who.int",
    "un.org",
    # video/media platforms (streaming/viewing, not open publishing of arbitrary pages)
    "youtube.com",
    "netflix.com",
    "spotify.com",
]

# NOTE: google.com, bing.com, yahoo.com, github.com, wordpress.com, blogspot.com
# etc. are deliberately NOT in this list even though they're major legitimate
# domains -- they have real free-hosting/user-content subdomains (Google
# Forms/Sites/Docs, GitHub user pages, etc.) where trusted-domain-abuse phishing
# is a genuine, documented technique, not a labeling error. A host-suffix match
# on "google.com" would also match "docs.google.com" and wrongly strip real
# phishing examples along with any noise.
