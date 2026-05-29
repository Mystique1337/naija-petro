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

# Domains preferred for the initial Tavily pass (Tier 1 + 2). News (tier 3) is
# allowed on the broad fallback pass so fresh developments are still captured.
PREFERRED_DOMAINS: list[str] = [d for d, (_, t) in NIGERIAN_SOURCES.items() if t <= 2]


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
