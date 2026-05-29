"""Lightweight, dependency-free text chunking for markdown documents.

Splits on the largest natural boundary that keeps chunks under `max_chars`
(paragraphs → lines → sentences → words), with a character overlap so context
isn't lost across chunk edges.
"""
from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?])\s+")


def _split(text: str, sep: str) -> list[str]:
    if sep == "":
        return list(text)
    if sep == "sent":
        return _SENT.split(text)
    return text.split(sep)


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Greedily pack the finest-grained pieces into <= max_chars windows.
    pieces: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
        else:
            for sent in _SENT.split(para):
                sent = sent.strip()
                if len(sent) <= max_chars:
                    pieces.append(sent)
                else:  # very long sentence: hard-wrap on words
                    buf = ""
                    for word in sent.split(" "):
                        if len(buf) + len(word) + 1 > max_chars:
                            pieces.append(buf.strip())
                            buf = word
                        else:
                            buf = f"{buf} {word}".strip()
                    if buf:
                        pieces.append(buf)

    chunks: list[str] = []
    cur = ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur.strip())
            # carry an overlap tail into the next chunk
            tail = cur[-overlap:] if overlap else ""
            cur = f"{tail} {p}".strip() if tail else p
        else:
            cur = f"{cur}\n\n{p}".strip() if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks
