"""Prompt construction: turn retrieved chunks into a cited context block."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import SYSTEM_PROMPT, settings

CITATION_INSTRUCTIONS = (
    "Use the numbered sources below to ground your answer in verifiable, Nigeria-specific "
    "facts. Cite sources inline as [1], [2], etc. immediately after the claim they support. "
    "Prefer official/regulatory sources (NUPRC, NMDPRA, NNPC, NEITI, the PIA 2021) over news. "
    "If the sources do not cover something, answer from your general petroleum-engineering "
    "knowledge and say so explicitly — never invent a citation. End with a '## Sources' list."
)


@dataclass
class RetrievedChunk:
    content: str
    source_url: str
    title: str
    domain: str
    source_tier: int
    score: float = 0.0
    similarity: float = 0.0


def _dedupe_sources(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Number citations by unique source URL, preserving order."""
    seen: dict[str, int] = {}
    ordered: list[RetrievedChunk] = []
    for c in chunks:
        key = c.source_url or c.title
        if key not in seen:
            seen[key] = len(ordered) + 1
            ordered.append(c)
    return ordered


def build_context(chunks: list[RetrievedChunk], budget: int | None = None) -> tuple[str, list[dict]]:
    """Return (context_block, sources). Retrieval can be broad, but the context is
    capped to a character budget so the window and cost stay bounded."""
    if not chunks:
        return "", []

    budget = budget or settings.context_char_budget
    # Diversify: take one chunk per unique source first (in relevance order), then
    # extras, so more distinct sources are shown within the character budget.
    seen_src: set = set()
    first, rest = [], []
    for c in chunks:
        key = c.source_url or c.title
        (first if key not in seen_src else rest).append(c)
        seen_src.add(key)
    ordered = first + rest

    included: list[RetrievedChunk] = []
    total = 0
    for c in ordered:
        seg = (c.content or "").strip()
        if included and total + len(seg) > budget:
            break
        included.append(c)
        total += len(seg) + 20

    sources = _dedupe_sources(included)
    url_to_n = {(s.source_url or s.title): i + 1 for i, s in enumerate(sources)}

    blocks = []
    for c in included:
        n = url_to_n[c.source_url or c.title]
        header = f"[{n}] {c.title or c.domain}".strip()
        blocks.append(f"{header}\n{c.content.strip()}")
    context = "\n\n---\n\n".join(blocks)

    citation_list = [
        {
            "n": i + 1,
            "title": s.title or s.domain or s.source_url,
            "url": s.source_url,
            "domain": s.domain,
            "tier": s.source_tier,
        }
        for i, s in enumerate(sources)
    ]
    return context, citation_list


REASONING_DIRECTIVE = (
    "First reason through your approach inside <think> and </think> tags (your private "
    "working: what the question needs, which sources are relevant, the key steps). Then, "
    "after </think>, write the final answer. Keep the reasoning concise.\n\n"
)


def build_messages(query: str, chunks: list[RetrievedChunk], history: list[dict] | None = None,
                   reasoning: bool = False) -> tuple[list[dict], list[dict]]:
    """Build chat messages + the source list for the UI."""
    context, sources = build_context(chunks)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []

    prefix = REASONING_DIRECTIVE if reasoning else ""
    if context:
        user = (
            f"{prefix}{CITATION_INSTRUCTIONS}\n\n"
            f"# Sources\n{context}\n\n"
            f"# Question\n{query}"
        )
    else:
        user = (
            f"{prefix}No external sources were retrieved. Answer from your general "
            "petroleum-engineering knowledge and note that the answer is not "
            "grounded in Nigeria-specific sources.\n\n"
            f"# Question\n{query}"
        )
    messages.append({"role": "user", "content": user})
    return messages, sources
