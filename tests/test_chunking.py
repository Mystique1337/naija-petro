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


def test_a_trailing_heading_does_not_emit_a_chunk_made_only_of_overlap():
    """A heading with nothing after it used to flush the carried tail as a chunk,
    a verbatim copy of the end of the previous chunk under a new breadcrumb."""
    chunks = chunk_text(f"# Regulation\n\n{_sentences(20)}\n\n## Appendix", max_chars=300, overlap=100)
    assert chunks
    assert not any("Appendix" in c for c in chunks)
    assert len(set(chunks)) == len(chunks)


def test_overlap_never_carries_table_rows_into_the_next_chunk():
    """The tail used to be taken from the joined body, so a chunk ending in a
    table pasted its last rows into the head of the next one."""
    md = f"{_sentences(6)}\n\n{TABLE}\n\n{_sentences(6, start=21)}"
    chunks = chunk_text(md, max_chars=200, overlap=120)
    assert sum(1 for c in chunks if TABLE in c) == 1
    for c in chunks:
        if TABLE in c:
            continue
        assert "| Bonga" not in c and "| Egina" not in c and "| Agbami" not in c


def test_overlap_never_carries_code_lines_into_the_next_chunk():
    code = "```python\ndef hydrostatic(mw_ppg, tvd_ft):\n    return 0.052 * mw_ppg * tvd_ft\n```"
    md = f"{_sentences(6)}\n\n{code}\n\n{_sentences(6, start=61)}"
    chunks = chunk_text(md, max_chars=200, overlap=120)
    assert sum(1 for c in chunks if code in c) == 1
    for c in chunks:
        if code in c:
            continue
        assert "0.052 * mw_ppg" not in c and "```" not in c


# --------------------------------------------------------------------------- #
# Atomic blocks are kept whole but bounded
# --------------------------------------------------------------------------- #
def test_a_huge_table_is_split_into_bounded_parts_with_whole_rows():
    head = "| Well | Field | Oil rate bopd |\n|---|---|---|"
    rows = [f"| W{i:03d} | Field {i % 11} | {900 + i} |" for i in range(300)]
    chunks = chunk_text(f"{head}\n" + "\n".join(rows), max_chars=600, overlap=100)

    assert len(chunks) > 5, "a 300 row table must not be one chunk"
    assert max(len(c) for c in chunks) <= 800
    for c in chunks:
        assert c.startswith("| Well | Field | Oil rate bopd |"), "each part repeats the header"
    for r in rows:
        assert sum(1 for c in chunks if r in c) == 1, f"row split or duplicated: {r}"


def test_a_huge_fenced_block_is_split_into_bounded_fenced_parts():
    lines = [f"q{i} = decline(qi={900 + i}, di=0.12, t={i})" for i in range(200)]
    chunks = chunk_text("```python\n" + "\n".join(lines) + "\n```", max_chars=600, overlap=100)

    assert len(chunks) > 3
    assert max(len(c) for c in chunks) <= 800
    for c in chunks:
        assert c.startswith("```python") and c.rstrip().endswith("```")
    for ln in lines:
        assert sum(1 for c in chunks if ln in c) == 1


def test_an_unterminated_fence_is_bounded_too():
    md = "```\n" + "\n".join(f"log line {i} from the daily drilling report" for i in range(300))
    chunks = chunk_text(md, max_chars=600, overlap=100)
    assert len(chunks) > 3
    assert max(len(c) for c in chunks) <= 800


# --------------------------------------------------------------------------- #
# Low-information chunks (page furniture)
# --------------------------------------------------------------------------- #
RUNNING_HEADER = "NIGERIAN OIL AND GAS REPORT 2024"
FURNITURE = "\n".join([RUNNING_HEADER] * 9 + ["Page 41", "Page 42", "Page 43"])
TOC = "\n".join([
    "Table of Contents",
    "1. Introduction ............................................ 1",
    "2. Upstream licensing rounds .............................. 17",
    "3. Midstream gas infrastructure ........................... 43",
    "4. Downstream deregulation ................................ 71",
    "5. Host community development trusts ..................... 106",
])
COOKIE = ("We use cookies to give you the best experience on this website. By continuing "
          "to browse the site you agree to our use of cookies. Read our Privacy Policy "
          "and our Terms of Use for more detail.")


def test_repeated_running_headers_are_dropped_while_the_body_survives():
    md = (f"# Body\n\n{_sentences(4)}\n\n"
          f"# Furniture\n\n{FURNITURE}\n\n"
          f"# More body\n\n{_sentences(4, start=51)}")
    chunks = chunk_text(md, max_chars=400, overlap=0)
    assert not any(RUNNING_HEADER in c for c in chunks)
    assert any("Section 1 of the report" in c for c in chunks)
    assert any("Section 51 of the report" in c for c in chunks)


def test_a_table_of_contents_is_dropped():
    md = f"# Front matter\n\n{TOC}\n\n# Chapter one\n\n{_sentences(5)}"
    chunks = chunk_text(md, max_chars=500, overlap=0)
    assert not any("Upstream licensing rounds" in c for c in chunks)
    assert any("Section 1 of the report" in c for c in chunks)


def test_a_cookie_and_privacy_notice_is_dropped():
    md = f"# Notice\n\n{COOKIE}\n\n# Content\n\n{_sentences(5)}"
    chunks = chunk_text(md, max_chars=500, overlap=0)
    assert not any("cookies" in c for c in chunks)
    assert any("Section 1 of the report" in c for c in chunks)


def test_furniture_does_not_ride_along_inside_a_good_chunk():
    """Furniture short enough to stay in the buffer used to be glued onto the
    next section instead of being dropped."""
    md = f"# Notice\n\n{COOKIE}\n\n# Content\n\n{_sentences(5)}"
    chunks = chunk_text(md, max_chars=4000, overlap=200)
    assert len(chunks) == 1
    assert "cookies" not in chunks[0]


def test_a_document_that_is_only_furniture_yields_no_chunks():
    assert chunk_text(f"# Privacy\n\n{COOKIE}\n\n{FURNITURE}\n", max_chars=1500, overlap=200) == []


def test_a_document_of_headings_only_yields_no_chunks():
    assert chunk_text("# One\n\n## Two\n\n### Three", max_chars=400, overlap=200) == []


def test_a_long_breadcrumb_cannot_keep_a_near_empty_chunk_alive():
    """The length floor applies to the body: the breadcrumb is navigation, and a
    deep one is long enough on its own to pass a 100 character filter."""
    md = ("# Nigerian Upstream Petroleum Regulation Handbook\n\n"
          "## Chapter Four: Midstream and Downstream Licensing\n\n"
          "### Section 12 Definitions\n\n"
          "See above.\n\n"
          f"# Substantive part\n\n{_sentences(6)}")
    chunks = chunk_text(md, max_chars=600, overlap=10)
    assert chunks
    assert not any("Section 12 Definitions" in c for c in chunks)
    assert any("Substantive part" in c for c in chunks)


# --------------------------------------------------------------------------- #
# The furniture filter must not eat real technical content
# --------------------------------------------------------------------------- #
def test_a_digit_heavy_markdown_table_is_never_treated_as_noise():
    numbers = ("| Year | bopd |\n|---|---|\n| 2019 | 1800000 |\n| 2020 | 1700000 |\n"
               "| 2021 | 1300000 |\n| 2022 | 1200000 |\n| 2023 | 1400000 |")
    chunks = chunk_text(f"# Production\n\n{numbers}", max_chars=500, overlap=0)
    assert len(chunks) == 1 and numbers in chunks[0]


def test_a_table_extracted_without_pipes_is_not_treated_as_noise():
    plain = "\n".join([
        "Year  Crude production bopd  Exports bopd",
        "2019  1,800,000  1,600,000",
        "2020  1,700,000  1,500,000",
        "2021  1,300,000  1,100,000",
        "2022  1,200,000  1,000,000",
        "2023  1,400,000  1,200,000",
    ])
    chunks = chunk_text(f"# Production history\n\n{plain}\n\n# Next\n\n{_sentences(4)}",
                        max_chars=500, overlap=0)
    assert any("1,800,000" in c for c in chunks)


def test_an_equation_only_section_is_not_treated_as_noise():
    md = ("# Darcy radial flow\n\n"
          "\\[ q = \\frac{k h (p_e - p_{wf})}{141.2 B \\mu \\ln(r_e / r_w)} \\]\n"
          "\\[ J = q / (p_r - p_{wf}) \\]\n\n"
          f"# Next\n\n{_sentences(4)}")
    chunks = chunk_text(md, max_chars=500, overlap=0)
    assert any("141.2" in c for c in chunks)
