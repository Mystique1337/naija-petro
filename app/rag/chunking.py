"""Structure-aware markdown chunking for RAG.

Keeps headings attached to their section as a breadcrumb prefix, never splits
tables or fenced code mid-structure, packs paragraphs up to a size budget with
overlap, and falls back to sentence/word splitting only for oversized prose.
This preserves technical detail (tables, equations, regulation clauses) far
better than naive fixed-size splitting, so retrieval does not lose context.

Two extra jobs, both driven by what the live knowledge base looks like (long
PDFs: a 706 chunk Petroleum Industry Act, several 200-400 chunk magazines, an
EIA model manual):

1. Page furniture is dropped. Running headers, bare page numbers, tables of
   contents and cookie/privacy notices otherwise get embedded and rank like
   real content. See `_is_low_information`, which is deliberately structural
   and never judges a chunk that holds a table or fenced code.
2. Atomic blocks are bounded. A 600 row table used to become one chunk that on
   its own exceeded the whole context budget; it is now broken at row
   boundaries, with the header repeated, into pieces that fit a chunk.
"""
from __future__ import annotations

import re

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\[])")
_TABLE_RULE = re.compile(r"^\|[\s:|-]+$")

# --------------------------------------------------------------------------- #
# Low-information (page furniture) detection
# --------------------------------------------------------------------------- #
# A "word" here is a run of at least three letters: enough to exclude page
# numbers, figure indices, dot leaders and single-letter column labels.
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
# "3.2 Reservoir management .......... 47"
_TOC_LINE = re.compile(r"\.{3,}\s*\d{1,4}\s*$|(?:\.\s){3,}\.?\s*\d{1,4}\s*$")
# "12", "Page 12", "page 12 of 340", "xiv"
_PAGE_LINE = re.compile(r"^(?:page\s*)?(?:\d{1,4}|[ivxlcdm]{1,7})(?:\s*of\s*\d{1,4})?$", re.I)
_BOILERPLATE = (
    "privacy policy", "cookie", "terms of use", "terms and conditions",
    "all rights reserved", "newsletter", "skip to content", "skip to main content",
    "javascript is disabled", "enable javascript", "advertisement",
    "follow us on", "share this article",
)
_NUMBER = re.compile(r"\d[\d,.]*")
_MIN_WORDS = 5          # below this a chunk cannot say anything
_MIN_ALPHA = 0.45       # letters as a share of non-space characters


def _looks_tabular(lines: list[str]) -> bool:
    """Rows of aligned figures, i.e. a table an extractor emitted without pipes.

    Production/reserves tables are the most valuable content in this corpus and
    are legitimately almost all digits, so they are exempt from the density rule.
    """
    if len(lines) < 3:
        return False
    numeric = sum(1 for ln in lines if len(_NUMBER.findall(ln)) >= 2)
    return numeric * 2 >= len(lines)


def _is_low_information(body: str, min_words: int = _MIN_WORDS) -> bool:
    """True when a body is page furniture rather than retrievable content.

    Conservative on purpose: every rule needs a structural signal, and a body
    that carries a markdown table or a fenced block is never judged at all,
    since those are legitimately dense in digits and punctuation.

    `min_words` is lowered to 0 when judging a single block, where a short line
    ("PART III", "See Table 4") may still belong with the text around it; the
    full floor applies to a finished chunk, which has to stand on its own.
    """
    s = (body or "").strip()
    if not s:
        return True
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    # Tables, fenced blocks and LaTeX are legitimately dense in digits and
    # punctuation, so they are never judged as furniture.
    if ("```" in s or "~~~" in s or any(ln.startswith("|") for ln in lines)
            or "\\[" in s or "\\(" in s or "$$" in s):
        return False

    words = _WORD.findall(s)
    if len(words) < min_words:                      # page numbers, stray captions, nav labels
        return True

    dense = "".join(s.split())
    alpha = sum(1 for ch in dense if ch.isalpha())
    # Mostly digits/punctuation *and* thin on words: numeric page furniture,
    # index rows, extraction artefacts. Prose stays well above one word per 20
    # characters, so equation-heavy passages are not caught by this.
    if (dense and alpha < _MIN_ALPHA * len(dense) and len(words) * 20 < len(s)
            and not _looks_tabular(lines)):
        return True

    # Half or more of the body is lines that appear again verbatim: a running
    # header or footer repeated once per PDF page.
    if len(lines) >= 6:
        seen: dict[str, int] = {}
        for ln in lines:
            seen[ln] = seen.get(ln, 0) + 1
        repeated = sum(len(ln) for ln in lines if seen[ln] > 1)
        if repeated * 2 >= sum(len(ln) for ln in lines):
            return True

    if len(lines) >= 4:
        furniture = sum(1 for ln in lines if _TOC_LINE.search(ln) or _PAGE_LINE.match(ln))
        if furniture * 5 >= len(lines) * 3:         # 60%+ table-of-contents / page rows
            return True

    low = s.lower()
    if len(words) < 80 and sum(1 for marker in _BOILERPLATE if marker in low) >= 2:
        return True                                 # cookie banner, footer, subscribe wall

    return False


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
def _split_table(rows: list[str], limit: int) -> list[str]:
    """Break an oversized table at row boundaries, repeating the header row.

    Rows are never split: a part may overshoot `limit` if a single row does.
    """
    whole = "\n".join(rows)
    if len(whole) <= limit or len(rows) < 4:
        return [whole]
    head = rows[:2] if _TABLE_RULE.match(rows[1].strip()) else []
    data = rows[len(head):]
    head_len = sum(len(r) + 1 for r in head)
    parts: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for r in data:
        if cur and head_len + cur_len + len(r) + 1 > limit:
            parts.append("\n".join(head + cur))
            cur, cur_len = [], 0
        cur.append(r)
        cur_len += len(r) + 1
    if cur:
        parts.append("\n".join(head + cur))
    return parts or [whole]


def _split_code(opener: str, body: list[str], closer: str | None, fence: str,
                limit: int) -> list[str]:
    """Break an oversized fenced block at line boundaries, re-fencing each part."""
    close = closer if closer is not None else fence
    whole = "\n".join([opener, *body, close]) if body or closer is not None else opener
    if len(whole) <= limit or not body:
        return [whole]
    frame = len(opener) + len(close) + 2
    parts: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for ln in body:
        if cur and frame + cur_len + len(ln) + 1 > limit:
            parts.append("\n".join([opener, *cur, close]))
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += len(ln) + 1
    if cur:
        parts.append("\n".join([opener, *cur, close]))
    return parts or [whole]


def _atomic_blocks(md: str, limit: int = 1500) -> list[tuple]:
    """Split markdown into blocks: ('heading',level,title) | ('code',t) | ('table',t) | ('text',t).

    Tables and fenced code stay atomic, but one whose text exceeds `limit` is
    emitted as several bounded blocks (whole rows / whole lines) so a single
    huge table or an unterminated fence cannot swallow the context budget.
    """
    lines = md.split("\n")
    blocks: list[tuple] = []
    buf: list[str] = []
    i, n = 0, len(lines)

    def flush():
        if buf:
            t = "\n".join(buf).strip()
            if t:
                blocks.append(("text", t))
            buf.clear()

    while i < n:
        s = lines[i]
        st = s.strip()
        if st.startswith("```") or st.startswith("~~~"):       # fenced code: keep whole
            flush()
            fence = st[:3]
            i += 1
            code: list[str] = []
            while i < n and not lines[i].strip().startswith(fence):
                code.append(lines[i]); i += 1
            closer = None
            if i < n:
                closer = lines[i]; i += 1
            for part in _split_code(s, code, closer, fence, limit):
                blocks.append(("code", part))
            continue
        m = _HEADING.match(s)
        if m:
            flush()
            blocks.append(("heading", len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        if st.startswith("|"):                                  # table: keep rows whole
            flush()
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i]); i += 1
            for part in _split_table(tbl, limit):
                blocks.append(("table", part))
            continue
        buf.append(s); i += 1
    flush()
    return blocks


def _sentences(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for p in _SENT.split(text):
        if len(p) <= limit:
            out.append(p)
        else:                                                   # hard-wrap a very long sentence
            cur = ""
            for word in p.split(" "):
                if len(cur) + len(word) + 1 > limit and cur:
                    out.append(cur); cur = word
                else:
                    cur = f"{cur} {word}".strip()
            if cur:
                out.append(cur)
    return out


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    blocks = _atomic_blocks(text, max_chars)
    chunks: list[str] = []
    cur: list[str] = []
    kinds: list[str] = []
    cur_len = 0
    fresh = False                       # anything new added since the last emit?
    heads: dict[int, str] = {}
    # The floor applies to the body: a breadcrumb is navigation, not content, and
    # a deep one is long enough on its own to keep an otherwise empty chunk alive.
    min_body = max(40, overlap // 2)

    def breadcrumb() -> str:
        return " > ".join(heads[k] for k in sorted(heads))

    def reset():
        nonlocal cur, kinds, cur_len, fresh
        cur, kinds, cur_len, fresh = [], [], 0, False

    def emit():
        nonlocal cur, kinds, cur_len
        body = "\n\n".join(cur).strip()
        # `fresh` is False when the buffer holds nothing but the overlap carried
        # over from the previous chunk (a trailing heading used to emit that as a
        # chunk of its own, a verbatim copy of the previous chunk's tail).
        if not body or not fresh:
            reset()
            return
        if len(body) <= min_body or _is_low_information(body):
            reset()                     # furniture: drop it, and do not carry it forward
            return
        bc = breadcrumb()
        chunks.append(f"{bc}\n\n{body}" if bc else body)
        # Overlap comes from the trailing prose only. Taking it from the joined
        # body pasted the last rows of a table, or the tail of a code block, into
        # the head of the next chunk as an orphaned fragment.
        tail_src = ""
        for item, kind in zip(reversed(cur), reversed(kinds)):
            if kind != "text":
                break
            tail_src = item if not tail_src else f"{item}\n\n{tail_src}"
        tail_src = tail_src.strip()
        tail = tail_src[-overlap:] if (overlap and len(tail_src) > overlap) else ""
        if tail and " " in tail:                                # start overlap at a word boundary
            tail = tail[tail.find(" ") + 1:]
        reset()
        if tail:
            cur, kinds, cur_len = [tail], ["text"], len(tail)

    for b in blocks:
        if b[0] == "heading":
            level, title = b[1], b[2]
            if cur_len > overlap:                               # close at a section boundary
                emit()
            heads = {k: v for k, v in heads.items() if k < level}
            heads[level] = title
            continue
        kind, body = b[0], b[1]
        # Drop furniture at the block level too, not only once a chunk is built:
        # a table of contents or a cookie banner that is short enough to merge
        # with the next section would otherwise ride along inside a good chunk.
        if kind == "text" and _is_low_information(body, min_words=0):
            continue
        if kind == "text" and len(body) > max_chars:
            for sent in _sentences(body, max_chars):
                if cur_len + len(sent) + 2 > max_chars and cur_len > overlap:
                    emit()
                cur.append(sent); kinds.append("text"); cur_len += len(sent) + 2
                fresh = True
        else:                                                   # text fits, or atomic table/code
            if cur_len + len(body) + 2 > max_chars and cur_len > overlap:
                emit()
            cur.append(body); kinds.append(kind); cur_len += len(body) + 2
            fresh = True
    emit()
    if chunks:
        return chunks
    # Nothing survived. Keep the document whole rather than lose it, unless it is
    # only headings or only furniture, in which case it should not be embedded.
    if not any(b[0] != "heading" for b in blocks) or _is_low_information(text):
        return []
    return [text]
