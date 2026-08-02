"""Structure-aware markdown chunking."""
from __future__ import annotations

import pytest

from app.rag.chunking import chunk_text


def _longest_overlap(a: str, b: str) -> int:
    """Length of the longest suffix of `a` that is also a prefix of `b`."""
    for k in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:k]):
            return k
    return 0


def _sentences(n: int, start: int = 1) -> str:
    """Distinct sentences (unique numbers) so overlap detection is unambiguous."""
    return " ".join(
        f"Section {i} of the report discusses Nigerian reservoir behaviour in detail."
        for i in range(start, start + n)
    )


# --------------------------------------------------------------------------- #
# Degenerate input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["", "   ", "\n\n\t  \n", None])
def test_empty_or_whitespace_only_input_returns_no_chunks(text):
    assert chunk_text(text) == []


def test_short_input_returns_a_single_chunk():
    text = ("The Petroleum Industry Act 2021 restructured the Nigerian oil and gas "
            "sector and created the NUPRC as the upstream regulator.")
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


# --------------------------------------------------------------------------- #
# Headings
# --------------------------------------------------------------------------- #
def test_heading_breadcrumb_is_prefixed_onto_the_chunk():
    md = (
        "# Nigerian Upstream\n\n"
        "## PIA 2021\n\n"
        "The Act created the NUPRC and the NMDPRA as separate upstream and "
        "midstream regulators for the Nigerian petroleum industry."
    )
    chunks = chunk_text(md, max_chars=400, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].startswith("Nigerian Upstream > PIA 2021\n\n")
    assert "The Act created the NUPRC" in chunks[0]


def test_breadcrumb_tracks_the_current_section_and_drops_siblings():
    body_a = _sentences(3, start=1)
    body_b = _sentences(3, start=11)
    md = f"# Regulation\n\n## NUPRC\n\n{body_a}\n\n## NMDPRA\n\n{body_b}"
    chunks = chunk_text(md, max_chars=220, overlap=20)
    assert len(chunks) >= 2
    assert chunks[0].startswith("Regulation > NUPRC")
    # The sibling heading replaces NUPRC rather than accumulating next to it.
    tail = [c for c in chunks if c.startswith("Regulation > NMDPRA")]
    assert tail, chunks
    assert not any(c.startswith("Regulation > NUPRC > NMDPRA") for c in chunks)


def test_heading_with_no_body_still_produces_content():
    md = "# Title only\n\n## Sub\n\n" + _sentences(4)
    chunks = chunk_text(md, max_chars=500, overlap=40)
    assert chunks
    assert all("Title only > Sub" in c for c in chunks)


# --------------------------------------------------------------------------- #
# Atomic blocks
# --------------------------------------------------------------------------- #
TABLE = (
    "| Field | Operator | Peak bopd |\n"
    "|---|---|---|\n"
    "| Bonga | Shell | 225000 |\n"
    "| Egina | TotalEnergies | 200000 |\n"
    "| Agbami | Chevron | 250000 |"
)


def test_markdown_table_is_never_split_across_chunks():
    md = f"{_sentences(6)}\n\n{TABLE}\n\n{_sentences(6, start=21)}"
    chunks = chunk_text(md, max_chars=200, overlap=20)
    assert len(chunks) > 1, "expected the prose to force several chunks"
    holders = [c for c in chunks if TABLE in c]
    assert len(holders) == 1, "the table must survive intact in exactly one chunk"
    # No chunk may contain a partial table without the whole of it.
    for c in chunks:
        if TABLE in c:
            continue
        assert "| Bonga | Shell | 225000 |" not in c


def test_fenced_code_block_stays_whole():
    code = (
        "```python\n"
        "def hydrostatic(mw_ppg, tvd_ft):\n"
        "    # 0.052 converts ppg x ft to psi\n"
        "    return 0.052 * mw_ppg * tvd_ft\n"
        "```"
    )
    md = f"{_sentences(6)}\n\n{code}\n\n{_sentences(6, start=31)}"
    chunks = chunk_text(md, max_chars=200, overlap=20)
    holders = [c for c in chunks if code in c]
    assert len(holders) == 1, "the fenced block must survive intact in exactly one chunk"


def test_tilde_fenced_code_block_stays_whole():
    code = "~~~\nqi = 1000\nDi = 0.1\n~~~"
    md = f"{_sentences(4)}\n\n{code}\n\n{_sentences(4, start=41)}"
    chunks = chunk_text(md, max_chars=200, overlap=20)
    assert sum(1 for c in chunks if code in c) == 1


# --------------------------------------------------------------------------- #
# Sizing, sentence fallback and overlap
# --------------------------------------------------------------------------- #
def test_oversized_prose_falls_back_to_sentence_splitting():
    max_chars, overlap = 200, 20
    md = _sentences(40)
    chunks = chunk_text(md, max_chars=max_chars, overlap=overlap)
    assert len(chunks) > 5
    # A chunk may overshoot only by the one sentence that was appended to an
    # almost-empty buffer, and sentences here are well under max_chars.
    longest_sentence = max(len(s) for s in md.split(". "))
    for c in chunks:
        assert len(c) <= max_chars + longest_sentence + 4, c
    # Nothing is lost: every sentence number is still present somewhere.
    joined = " ".join(chunks)
    for i in range(1, 41):
        assert f"Section {i} of the report" in joined


def test_a_single_very_long_sentence_is_hard_wrapped():
    max_chars = 120
    md = "word " * 200            # no sentence terminators at all
    chunks = chunk_text(md.strip(), max_chars=max_chars, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= max_chars + 20 for c in chunks)


def test_overlap_carries_text_forward_at_a_word_boundary():
    max_chars, overlap = 240, 60
    chunks = chunk_text(_sentences(30), max_chars=max_chars, overlap=overlap)
    assert len(chunks) > 2

    for first, second in zip(chunks, chunks[1:]):
        k = _longest_overlap(first, second)
        assert k > 0, "expected the tail of one chunk to start the next"
        assert k <= overlap
        # The carried text starts right after a space, never mid-word.
        assert first[-(k + 1)] == " ", repr(first[-(k + 5):])


def test_zero_overlap_does_not_repeat_text():
    chunks = chunk_text(_sentences(30), max_chars=200, overlap=0)
    assert len(chunks) > 2
    for first, second in zip(chunks, chunks[1:]):
        assert _longest_overlap(first, second) == 0
