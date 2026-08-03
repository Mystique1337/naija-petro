"""Context assembly and message building."""
from __future__ import annotations

from app.config import SYSTEM_PROMPT
from app.rag.prompts import (
    CITATION_INSTRUCTIONS,
    REASONING_DIRECTIVE,
    RetrievedChunk,
    _window_budget,
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
# Citation numbering must not collide
# --------------------------------------------------------------------------- #
def test_untitled_sources_from_different_places_do_not_share_a_citation():
    """Keying on `source_url or title` made every untitled, url-less chunk the
    same source, so one document was cited as another."""
    c1 = RetrievedChunk(content="alpha body", source_url="", title="", domain="alpha.ng", source_tier=2)
    c2 = RetrievedChunk(content="beta body", source_url="", title="", domain="beta.ng", source_tier=2)
    context, sources = build_context([c1, c2])
    assert len(sources) == 2
    assert "[1]" in context and "[2]" in context


def test_two_documents_sharing_a_url_are_cited_separately():
    c1 = RetrievedChunk(content="2023 edition", source_url=A, title="NUPRC report 2023",
                        domain="www.nuprc.gov.ng", source_tier=1)
    c2 = RetrievedChunk(content="2024 edition", source_url=A, title="NUPRC report 2024",
                        domain="www.nuprc.gov.ng", source_tier=1)
    _context, sources = build_context([c1, c2])
    assert [s["title"] for s in sources] == ["NUPRC report 2023", "NUPRC report 2024"]


def test_citation_numbers_first_appear_in_ascending_order():
    """Diversification reorders the chunks, so check numbering follows it."""
    chunks = [chunk("a1", A), chunk("a2", A), chunk("b1", B), chunk("c1", C), chunk("b2", B)]
    context, sources = build_context(chunks)
    firsts = [context.index(f"[{s['n']}]") for s in sources]
    assert firsts == sorted(firsts)
    assert [s["n"] for s in sources] == [1, 2, 3]


def test_empty_chunks_do_not_take_a_citation_number():
    empty = RetrievedChunk(content="   ", source_url=A, title="Empty doc", domain="d", source_tier=1)
    context, sources = build_context([empty, chunk("b1", B)])
    assert len(sources) == 1
    assert "b1" in context and "Empty doc" not in context


# --------------------------------------------------------------------------- #
# The context cannot overflow the model window
# --------------------------------------------------------------------------- #
def test_an_oversized_first_chunk_is_clipped_to_the_budget():
    """The first chunk is always included, but a single 80k character chunk from
    a document ingested before chunking bounded tables must not fill the window."""
    huge = RetrievedChunk(content="x " * 40000, source_url=A, title="Huge table",
                          domain="www.nuprc.gov.ng", source_tier=1)
    context, sources = build_context([huge], budget=5000)
    assert len(sources) == 1
    assert len(context) <= 5200
    assert "truncated" in context


def test_the_context_never_exceeds_what_the_window_can_hold():
    chunks = [chunk(f"m{i}", f"https://example{i}.com/a", size=4000) for i in range(60)]
    context, _sources = build_context(chunks, budget=10 ** 7)
    assert len(context) <= _window_budget() + 4200      # at most one chunk of overshoot


def test_a_long_history_shrinks_the_context_instead_of_overflowing():
    chunks = [chunk(f"m{i}", f"https://example{i}.com/a", size=3000) for i in range(20)]
    long_history = [{"role": "user", "content": "q" * 30000},
                    {"role": "assistant", "content": "a" * 30000}]
    with_history, _ = build_messages("q", chunks, history=long_history)
    without_history, _ = build_messages("q", chunks)
    assert len(with_history[-1]["content"]) < len(without_history[-1]["content"])


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


# --------------------------------------------------------------------------- #
# Citation instructions
# --------------------------------------------------------------------------- #
def test_the_prompt_names_the_markers_that_actually_exist():
    """Naming the range is what stops the model inventing [7] out of four sources."""
    one, _ = build_messages("q", [chunk("a1", A)])
    three, _ = build_messages("q", [chunk("a1", A), chunk("b1", B), chunk("c1", C)])
    assert "The only valid marker is [1]." in one[-1]["content"]
    assert "The only valid markers are [1] to [3]." in three[-1]["content"]
    assert "valid marker" not in build_messages("q", [])[0][-1]["content"]


def test_the_citation_rules_are_explicit_and_show_an_example():
    assert "[2]." in CITATION_INSTRUCTIONS, "the rules carry a worked example marker"
    assert "## Sources" in CITATION_INSTRUCTIONS
    assert len(CITATION_INSTRUCTIONS) < 1200, "rules must stay short enough to be read"
