"""Authoritative Nigerian petroleum sources used to bias web retrieval.

The model is weak on Nigeria-specific facts, so we steer Tavily toward verifiable,
citable sources. Tier 1 = official/regulatory, Tier 2 = reference/intergovernmental,
Tier 3 = reputable industry news. Lower tier number = higher trust (used for ranking
and shown in citations).
"""
from __future__ import annotations

from urllib.parse import urlparse

# domain -> (human label, tier)
NIGERIAN_SOURCES: dict[str, tuple[str, int]] = {
    # --- Tier 1: official / regulatory ---
    "nuprc.gov.ng":        ("Nigerian Upstream Petroleum Regulatory Commission", 1),
    "nmdpra.gov.ng":       ("Nigerian Midstream & Downstream Petroleum Regulatory Authority", 1),
    "nnpcgroup.com":       ("NNPC Limited", 1),
    "neiti.gov.ng":        ("Nigeria Extractive Industries Transparency Initiative", 1),
    "ncdmb.gov.ng":        ("Nigerian Content Development & Monitoring Board", 1),
    "pia.gov.ng":          ("Petroleum Industry Act 2021 (official)", 1),
    "dpr.gov.ng":          ("Department of Petroleum Resources (legacy)", 1),
    "petroleumindustrybill.com": ("Petroleum Industry Act resources", 1),
    # --- Tier 2: reference / intergovernmental ---
    "eia.gov":             ("U.S. Energy Information Administration", 2),
    "opec.org":            ("OPEC", 2),
    "iea.org":             ("International Energy Agency", 2),
    "worldbank.org":       ("World Bank", 2),
    "onepetro.org":        ("OnePetro (SPE)", 2),
    "spe.org":             ("Society of Petroleum Engineers", 2),
    "spenigeriacouncil.org": ("SPE Nigeria Council", 2),
    # --- Tier 3: reputable industry / national news ---
    "africaoilgasreport.com": ("Africa Oil+Gas Report", 3),
    "punchng.com":         ("The Punch (Energy)", 3),
    "vanguardngr.com":     ("Vanguard (Energy)", 3),
    "businessday.ng":      ("BusinessDay", 3),
    "thisdaylive.com":     ("ThisDay", 3),
    "premiumtimesng.com":  ("Premium Times", 3),
    "oilprice.com":        ("OilPrice.com", 3),
}

# Broad institutions that publish across every sector. A domain-restricted search
# always returns its best match, so asking one of these about a petroleum topic
# returns whatever it has that is closest, which for a development bank is a blog
# post about survey methodology. They stay in the tier map above, so they are
# labelled correctly when they legitimately turn up in the open web pass, but they
# are not searched directly. This had put roughly 28 World Bank documents into the
# store, of which one was about petroleum.
BROAD_INSTITUTIONS: frozenset = frozenset({"worldbank.org"})

# Domains preferred for the initial Tavily pass (Tier 1 + 2). News (tier 3) is
# allowed on the broad fallback pass so fresh developments are still captured.
PREFERRED_DOMAINS: list[str] = [
    d for d, (_, t) in NIGERIAN_SOURCES.items() if t <= 2 and d not in BROAD_INSTITUTIONS
]

# A document has to actually be about this subject. Domain-restricted search, a
# mis-parsed PDF, or a page that merely mentions Nigeria will otherwise be stored
# and retrieved forever.
_PETROLEUM_TERMS = (
    "petroleum", "oil", "gas", "reservoir", "drilling", "well", "wellbore", "hydrocarbon",
    "crude", "upstream", "downstream", "midstream", "refinery", "refining", "opec",
    "permeability", "porosity", "seismic", "completion", "production", "barrel", "bopd",
    "formation", "pipeline", "lng", "flaring", "eor", "pvt", "geology", "subsurface",
    "offshore", "onshore", "licence", "license", "lease", "royalty", "condensate",
)


def is_petroleum_relevant(text: str, title: str = "", sample: int = 200_000,
                          min_terms: int = 4) -> bool:
    """True when the text is plausibly about petroleum, gas or its regulation.

    Deliberately generous: a real regulatory or engineering document trips this
    many times over, while a survey-methodology manual or a blog post about
    remittances trips almost none.

    It reads the whole document, not the opening. Article pages routinely begin
    with a sign-in menu, social links and a cookie banner, so a head-only sample
    judged a JPT paper on Nigerian deepwater fiscal terms and an IEA analysis of
    enhanced oil recovery to be off topic, which is exactly backwards.
    """
    body = f"{title}\n{(text or '')[:sample]}".lower()
    return sum(1 for term in _PETROLEUM_TERMS if term in body) >= min_terms


# Domains that are never worth citing in an engineering answer: video, homework
# and answer-mill sites, SEO calculator pages, and social platforms. The broad
# open-web Tavily pass surfaces these, and they had already put YouTube listings
# and a Chegg "Solved..." page into the knowledge base.
BLOCKED_DOMAINS: frozenset = frozenset({
    "youtube.com", "youtu.be", "vimeo.com", "tiktok.com", "facebook.com",
    "instagram.com", "x.com", "twitter.com", "reddit.com", "pinterest.com",
    "linkedin.com", "medium.com", "quora.com",
    "chegg.com", "coursehero.com", "quizlet.com", "studocu.com", "scribd.com",
    "brainly.com", "numerade.com", "toolgrit.com", "calculator.net",
    "slideshare.net", "academia.edu", "researchgate.net",
})


def is_blocked(url: str) -> bool:
    """True for sources that should never enter the knowledge base."""
    host = domain_of(url)
    if not host:
        return True
    return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)


# Characters from scripts this assistant does not answer in. A document full of
# them is a translated edition that pollutes retrieval with unreadable context.
_NON_LATIN = (
    (0x0400, 0x04FF),    # Cyrillic
    (0x0590, 0x06FF),    # Hebrew, Arabic
    (0x3040, 0x30FF),    # Hiragana, Katakana
    (0x3400, 0x4DBF),    # CJK extension A
    (0x4E00, 0x9FFF),    # CJK unified
    (0xAC00, 0xD7AF),    # Hangul
)


def is_english(text: str, sample: int = 4000, threshold: float = 0.03) -> bool:
    """Cheap script check: reject text with a meaningful share of non-Latin script.

    A Chinese edition of the SPE reserves guidelines was ingested and became the
    single largest document in the store, so it was retrieved for ordinary English
    questions and fed back as context the model could not use.
    """
    head = (text or "")[:sample]
    letters = [c for c in head if c.isalpha()]
    if not letters:
        return False
    hits = sum(1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in _NON_LATIN))
    return (hits / len(letters)) < threshold


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def classify(url: str) -> tuple[str, int]:
    """Return (label, tier) for a URL, matching on registered/parent domain."""
    host = domain_of(url)
    if not host:
        return ("Unknown source", 3)
    for dom, (label, tier) in NIGERIAN_SOURCES.items():
        if host == dom or host.endswith("." + dom):
            return (label, tier)
    # Government / academic heuristics for unseen but trustworthy hosts.
    if host.endswith(".gov.ng"):
        return (f"Nigerian government ({host})", 1)
    if host.endswith(".edu") or host.endswith(".edu.ng") or host.endswith(".ac.uk"):
        return (f"Academic ({host})", 2)
    return (host or "Unknown source", 3)


def is_nigeria_relevant(text: str) -> bool:
    """Cheap heuristic: does a query look Nigeria-specific?"""
    t = (text or "").lower()
    keywords = (
        "nigeria", "nigerian", "niger delta", "nnpc", "nuprc", "nmdpra", "neiti",
        "ncdmb", "pia 2021", "petroleum industry act", "dangote", "bonny", "forcados",
        "qua iboe", "escravos", "opl", "oml", "naija",
    )
    return any(k in t for k in keywords)
