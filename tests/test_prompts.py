"""Context assembly and message building."""
from __future__ import annotations

from app.config import SYSTEM_PROMPT
from app.rag.prompts import (
    CITATION_INSTRUCTIONS,
    REASONING_DIRECTIVE,
    RetrievedChunk,
    build_context,
    build_messages,
)


def chunk(marker: str, url: str, title: str = "", tier: int = 1, size: int = 60) -> RetrievedChunk:
    body = f"{marker} " + ("x" * max(0, size - len(marker) - 1))
    return RetrievedChunk(
        content=body,
        source_url=url,
        title=title or f"Title for {url}",
        domain=url.split("/")[2] if "//" in url else url,
        source_tier=tier,
        similarity=0.9,
    )


A = "https://www.nuprc.gov.ng/report"
B = "https://opec.org/nigeria"
C = "https://businessday.ng/oil"


# --------------------------------------------------------------------------- #
# build_context
# --------------------------------------------------------------------------- #
def test_no_chunks_gives_empty_context_and_no_sources():
    assert build_context([]) == ("", [])


def test_citations_are_numbered_by_unique_source_url():
    chunks = [chunk("a1", A), chunk("a2", A), chunk("b1", B)]
    context, sources = build_context(chunks)

    assert [s["n"] for s in sources] == [1, 2]
    assert [s["url"] for s in sources] == [A, B]
    # Both chunks from source A carry the same citation number.
    assert context.count("[1]") == 2
    assert context.count("[2]") == 1


def test_sources_are_diversified_one_chunk_per_source_first():
    chunks = [chunk("a1", A), chunk("a2", A), chunk("b1", B), chunk("c1", C)]
    context, sources = build_context(chunks)

    order = [context.index(m) for m in ("a1", "b1", "c1", "a2")]
    assert order == sorted(order), "expected one chunk per source before the extras"
    assert [s["url"] for s in sources] == [A, B, C]


def test_citation_dicts_have_the_shape_the_ui_expects():
    _context, sources = build_context([chunk("a1", A, title="NUPRC 2024 report", tier=1)])
    assert sources == [{
        "n": 1,
        "title": "NUPRC 2024 report",
        "url": A,
        "domain": "www.nuprc.gov.ng",
        "tier": 1,
    }]


def test_citation_title_falls_back_to_domain_then_url():
    c = RetrievedChunk(content="body text", source_url=A, title="", domain="nuprc.gov.ng", source_tier=1)
    _context, sources = build_context([c])
    assert sources[0]["title"] == "nuprc.gov.ng"


def test_character_budget_is_respected():
    chunks = [chunk(f"m{i}", f"https://example{i}.com/a", size=100) for i in range(10)]
    context, sources = build_context(chunks, budget=250)

    assert len(context) < 600          # 3 chunks of 100 plus headers, not all 10
    assert len(sources) < 10
    assert "m0" in context


def test_at_least_one_chunk_survives_an_impossibly_small_budget():
    chunks = [chunk("m0", A, size=500), chunk("m1", B, size=500)]
    context, sources = build_context(chunks, budget=1)
    assert len(sources) == 1
    assert "m0" in context


def test_chunks_without_a_url_are_keyed_by_title():
    c1 = RetrievedChunk(content="first body", source_url="", title="Uploaded doc", domain="upload", source_tier=2)
    c2 = RetrievedChunk(content="second body", source_url="", title="Uploaded doc", domain="upload", source_tier=2)
    _context, sources = build_context([c1, c2])
    assert len(sources) == 1
    assert sources[0]["title"] == "Uploaded doc"


# --------------------------------------------------------------------------- #
# build_messages
# --------------------------------------------------------------------------- #
def test_build_messages_starts_with_the_system_prompt_and_ends_with_the_user_turn():
    messages, sources = build_messages("What is the PIA?", [chunk("a1", A)])
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[-1]["role"] == "user"
    assert "What is the PIA?" in messages[-1]["content"]
    assert CITATION_INSTRUCTIONS in messages[-1]["content"]
    assert "# Sources" in messages[-1]["content"]
    assert len(sources) == 1


def test_build_messages_appends_history_between_system_and_user():
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    messages, _sources = build_messages("follow up", [chunk("a1", A)], history=history)
    assert messages[1:3] == history
    assert len(messages) == 4


def test_build_messages_without_history():
    messages, _sources = build_messages("q", [chunk("a1", A)], history=None)
    assert len(messages) == 2


def test_reasoning_directive_only_when_reasoning_is_true():
    on, _ = build_messages("q", [chunk("a1", A)], reasoning=True)
    off, _ = build_messages("q", [chunk("a1", A)], reasoning=False)
    assert on[-1]["content"].startswith(REASONING_DIRECTIVE)
    assert REASONING_DIRECTIVE not in off[-1]["content"]
    assert "<think>" not in off[-1]["content"]


def test_no_sources_variant_when_nothing_was_retrieved():
    messages, sources = build_messages("q", [], reasoning=False)
    assert sources == []
    content = messages[-1]["content"]
    assert "No external sources were retrieved" in content
    assert CITATION_INSTRUCTIONS not in content
    assert "# Question\nq" in content


def test_no_sources_variant_still_honours_the_reasoning_flag():
    messages, _ = build_messages("q", [], reasoning=True)
    assert messages[-1]["content"].startswith(REASONING_DIRECTIVE)
