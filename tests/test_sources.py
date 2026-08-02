"""Source classification used to tier and label citations."""
from __future__ import annotations

import pytest

from app.rag.sources import (
    NIGERIAN_SOURCES,
    PREFERRED_DOMAINS,
    classify,
    domain_of,
    is_nigeria_relevant,
)


# --------------------------------------------------------------------------- #
# domain_of
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url,expected", [
    ("https://nuprc.gov.ng/data", "nuprc.gov.ng"),
    ("https://www.nuprc.gov.ng/data", "nuprc.gov.ng"),          # www is stripped
    ("http://WWW.OPEC.ORG/Nigeria", "opec.org"),                 # and case-folded
    ("https://opec.org", "opec.org"),
    ("https://sub.www.example.com/x", "sub.www.example.com"),    # only a leading www
    ("https://example.com:8443/x", "example.com:8443"),
])
def test_domain_of(url, expected):
    assert domain_of(url) == expected


@pytest.mark.parametrize("junk", ["", "not a url", "   ", "://///", "javascript:alert(1)", None, 42])
def test_domain_of_survives_junk_input(junk):
    assert domain_of(junk) == ""


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #
def test_classify_tier_1_official_source():
    assert classify("https://www.nuprc.gov.ng/regulations/2024") == (
        "Nigerian Upstream Petroleum Regulatory Commission", 1)


def test_classify_tier_2_reference_source():
    assert classify("https://opec.org/opec_web/en/about_us/167.htm") == ("OPEC", 2)


def test_classify_tier_3_news_source():
    assert classify("https://punchng.com/energy/story") == ("The Punch (Energy)", 3)


@pytest.mark.parametrize("url", [
    "https://something.nuprc.gov.ng/a",
    "https://data.portal.nuprc.gov.ng/a",
    "https://www.something.nuprc.gov.ng/a",
])
def test_classify_matches_subdomains_of_a_known_domain(url):
    assert classify(url) == ("Nigerian Upstream Petroleum Regulatory Commission", 1)


def test_classify_does_not_match_a_lookalike_suffix():
    # notnuprc.gov.ng must not match nuprc.gov.ng, but it is still a .gov.ng host.
    label, tier = classify("https://notnuprc.gov.ng/a")
    assert label != "Nigerian Upstream Petroleum Regulatory Commission"
    assert tier == 1


@pytest.mark.parametrize("url,host", [
    ("https://fmpr.gov.ng/policy", "fmpr.gov.ng"),
    ("https://www.energy.gov.ng/", "energy.gov.ng"),
])
def test_gov_ng_heuristic_is_tier_1(url, host):
    assert classify(url) == (f"Nigerian government ({host})", 1)


@pytest.mark.parametrize("url,host", [
    ("https://stanford.edu/paper", "stanford.edu"),
    ("https://www.unilag.edu.ng/paper", "unilag.edu.ng"),
    ("https://ox.ac.uk/paper", "ox.ac.uk"),
])
def test_academic_heuristic_is_tier_2(url, host):
    assert classify(url) == (f"Academic ({host})", 2)


def test_unknown_host_defaults_to_tier_3_labelled_by_host():
    assert classify("https://www.randomblog.example/post") == ("randomblog.example", 3)


@pytest.mark.parametrize("url", ["", "not a url", None])
def test_unparseable_url_is_an_unknown_tier_3_source(url):
    assert classify(url) == ("Unknown source", 3)


def test_every_known_source_classifies_to_its_own_tier():
    for domain, (label, tier) in NIGERIAN_SOURCES.items():
        assert classify(f"https://{domain}/some/path") == (label, tier)
        assert tier in (1, 2, 3)


# --------------------------------------------------------------------------- #
# PREFERRED_DOMAINS
# --------------------------------------------------------------------------- #
def test_preferred_domains_hold_only_tier_1_and_2():
    assert PREFERRED_DOMAINS, "the preferred list must not be empty"
    for domain in PREFERRED_DOMAINS:
        assert NIGERIAN_SOURCES[domain][1] <= 2
    tier3 = [d for d, (_l, t) in NIGERIAN_SOURCES.items() if t == 3]
    assert tier3, "expected some tier 3 sources to exist"
    assert not set(tier3) & set(PREFERRED_DOMAINS)
    assert len(PREFERRED_DOMAINS) == len(set(PREFERRED_DOMAINS))


# --------------------------------------------------------------------------- #
# is_nigeria_relevant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "What does NUPRC say about flaring?",
    "what does nuprc say about flaring?",
    "Niger Delta production outlook",
    "NIGERIA crude exports",
    "Tell me about the Petroleum Industry Act",
    "OML 118 water injection",
    "Dangote refinery throughput",
])
def test_is_nigeria_relevant_is_case_insensitive(text):
    assert is_nigeria_relevant(text) is True
    assert is_nigeria_relevant(text.upper()) is True
    assert is_nigeria_relevant(text.lower()) is True


@pytest.mark.parametrize("text", [
    "How does a centrifugal pump work?",
    "Explain relative permeability curves",
    "",
    None,
])
def test_is_nigeria_relevant_rejects_generic_text(text):
    assert is_nigeria_relevant(text) is False
