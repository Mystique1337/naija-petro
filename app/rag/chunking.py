"""Structure-aware markdown chunking for RAG.

Keeps headings attached to their section as a breadcrumb prefix, never splits
tables or fenced code, packs paragraphs up to a size budget with overlap, and
falls back to sentence/word splitting only for oversized prose. This preserves
technical detail (tables, equations, regulation clauses) far better than naive
fixed-size splitting, so retrieval does not lose context.
"""
from __future__ import annotations

import re

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\[])")


def _atomic_blocks(md: str) -> list[tuple]:
    """Split markdown into blocks: ('heading',level,title) | ('code',t) | ('table',t) | ('text',t)."""
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
            code = [s]
            i += 1
            while i < n and not lines[i].strip().startswith(fence):
                code.append(lines[i]); i += 1
            if i < n:
                code.append(lines[i]); i += 1
            blocks.append(("code", "\n".join(code)))
            continue
        m = _HEADING.match(s)
        if m:
            flush()
            blocks.append(("heading", len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        if st.startswith("|"):                                  # table: keep whole
            flush()
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i]); i += 1
            blocks.append(("table", "\n".join(tbl)))
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
    blocks = _atomic_blocks(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    heads: dict[int, str] = {}

    def breadcrumb() -> str:
        return " > ".join(heads[k] for k in sorted(heads))

    def emit():
        nonlocal cur, cur_len
        body = "\n\n".join(cur).strip()
        if not body:
            cur, cur_len = [], 0
            return
        bc = breadcrumb()
        chunks.append(f"{bc}\n\n{body}" if bc else body)
        tail = body[-overlap:] if (overlap and len(body) > overlap) else ""
        if tail and " " in tail:                                # start overlap at a word boundary
            tail = tail[tail.find(" ") + 1:]
        cur = [tail] if tail else []
        cur_len = len(tail)

    for b in blocks:
        if b[0] == "heading":
            level, title = b[1], b[2]
            if cur_len > overlap:                               # close at a section boundary
                emit()
            heads = {k: v for k, v in heads.items() if k < level}
            heads[level] = title
            continue
        body = b[1]
        if b[0] == "text" and len(body) > max_chars:
            for sent in _sentences(body, max_chars):
                if cur_len + len(sent) + 2 > max_chars and cur_len > overlap:
                    emit()
                cur.append(sent); cur_len += len(sent) + 2
        else:                                                   # text fits, or atomic table/code kept whole
            if cur_len + len(body) + 2 > max_chars and cur_len > overlap:
                emit()
            cur.append(body); cur_len += len(body) + 2
    emit()
    return [c for c in chunks if len(c.strip()) > max(40, overlap // 2)] or ([text] if text else [])
