"""Prompt construction: turn retrieved chunks into a cited context block."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import SYSTEM_PROMPT, settings

# Numbered, imperative and with one worked example: the model complied only
# sometimes with a prose paragraph. The rules it breaks most often (a marker on
# every borrowed claim, and never inventing a number) are stated first and last.
CITATION_INSTRUCTIONS = (
    "Answer from the numbered sources below and cite them inline. Citation rules:\n"
    "1. Put the marker at the end of the sentence it supports, before the full stop, "
    "like this: Nigeria produced about 1.4 million bopd in 2023 [2].\n"
    "2. Every claim taken from a source needs a marker. An uncited borrowed fact is an error.\n"
    "3. Use only the numbers listed under '# Sources'. Never invent a marker, a title or a URL.\n"
    "4. Where sources disagree, prefer official ones (NUPRC, NMDPRA, NNPC, NEITI, PIA 2021) "
    "over news, and say which you followed.\n"
    "5. If the sources do not cover a point, say so in that sentence and answer it from your "
    "general petroleum-engineering knowledge, with no marker.\n"
    "6. Close with a '## Sources' list of the numbers you actually cited."
)

# Rough characters per token for this corpus (LaTeX, tables and unit strings
# tokenise worse than plain English), plus what the rest of the prompt costs:
# system prompt, citation rules, question and separators.
_CHARS_PER_TOKEN = 3.2
_RESERVE_TOKENS = 700
_MIN_CONTEXT_CHARS = 2000       # never squeeze the context below this
_TRUNCATED = "\n[... source truncated]"


@dataclass
class RetrievedChunk:
    content: str
    source_url: str
    title: str
    domain: str
    source_tier: int
    score: float = 0.0
    similarity: float = 0.0


def _source_key(c: RetrievedChunk) -> tuple:
    """Identity of the document a chunk came from.

    Keyed on url + title + domain, not on `source_url or title`: two documents
    can share a URL (a page re-ingested after its title changed) and uploads can
    have neither URL nor title, and both cases used to collapse onto one key, so
    a chunk was cited as a document it did not come from.
    """
    return (
        (c.source_url or "").strip().lower(),
        (c.title or "").strip().lower(),
        (c.domain or "").strip().lower(),
    )


def _dedupe_sources(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Number citations by unique source document, preserving order."""
    seen: set = set()
    ordered: list[RetrievedChunk] = []
    for c in chunks:
        key = _source_key(c)
        if key not in seen:
            seen.add(key)
            ordered.append(c)
    return ordered


def _window_budget() -> int:
    """Characters of context the model's own window can actually take."""
    room = settings.max_model_len - settings.max_new_tokens - _RESERVE_TOKENS
    return max(_MIN_CONTEXT_CHARS, int(room * _CHARS_PER_TOKEN))


def _clip(text: str, limit: int) -> str:
    """Cut `text` to `limit` characters at a line or word boundary."""
    if len(text) <= limit:
        return text
    cut = text[: max(1, limit - len(_TRUNCATED))]
    stop = max(cut.rfind("\n"), cut.rfind(" "))
    if stop > len(cut) // 2:
        cut = cut[:stop]
    return cut.rstrip() + _TRUNCATED


def build_context(chunks: list[RetrievedChunk], budget: int | None = None) -> tuple[str, list[dict]]:
    """Return (context_block, sources). Retrieval can be broad, but the context is
    capped to a character budget so the window and cost stay bounded."""
    if not chunks:
        return "", []

    budget = settings.context_char_budget if budget is None else budget
    # The configured budget is a request; the window is the hard limit.
    budget = max(1, min(budget, _window_budget()))
    # The first chunk is always included so a question is never left with no
    # context at all, but it is clipped: an oversized chunk (a whole table from a
    # document ingested before chunking bounded them) would otherwise fill the
    # window on its own.
    first_cap = max(budget, _MIN_CONTEXT_CHARS)

    # Diversify: take one chunk per unique source first (in relevance order), then
    # extras, so more distinct sources are shown within the character budget.
    # Citation numbers are assigned from this same order, so they still appear in
    # ascending order in the context; the extras only repeat earlier numbers.
    seen_src: set = set()
    first, rest = [], []
    for c in chunks:
        key = _source_key(c)
        (first if key not in seen_src else rest).append(c)
        seen_src.add(key)
    ordered = first + rest

    included: list[RetrievedChunk] = []
    bodies: list[str] = []
    total = 0
    for c in ordered:
        seg = (c.content or "").strip()
        if not seg:
            continue                    # an empty chunk must not burn a citation number
        if not included:
            seg = _clip(seg, first_cap)
        # Count what the block actually costs: header line plus separator.
        cost = len(seg) + len(c.title or c.domain or "") + 16
        if included and total + cost > budget:
            break
        included.append(c)
        bodies.append(seg)
        total += cost

    if not included:
        return "", []

    sources = _dedupe_sources(included)
    key_to_n = {_source_key(s): i + 1 for i, s in enumerate(sources)}

    blocks = []
    for c, seg in zip(included, bodies):
        header = f"[{key_to_n[_source_key(c)]}] {c.title or c.domain}".strip()
        blocks.append(f"{header}\n{seg}")
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


def _marker_scope(n: int) -> str:
    """Name the markers that exist, which is what stops invented ones."""
    if n <= 0:
        return ""
    if n == 1:
        return "\nThe only valid marker is [1]."
    return f"\nThe only valid markers are [1] to [{n}]."


def build_messages(query: str, chunks: list[RetrievedChunk], history: list[dict] | None = None,
                   reasoning: bool = False) -> tuple[list[dict], list[dict]]:
    """Build chat messages + the source list for the UI."""
    history = history or []
    # History shares the window with the context, so a long conversation shrinks
    # the retrieved context rather than overflowing the model.
    spent = sum(len(m.get("content") or "") for m in history) + len(query or "")
    budget = min(settings.context_char_budget, max(_MIN_CONTEXT_CHARS, _window_budget() - spent))
    context, sources = build_context(chunks, budget=budget)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history

    prefix = REASONING_DIRECTIVE if reasoning else ""
    if context:
        user = (
            f"{prefix}{CITATION_INSTRUCTIONS}{_marker_scope(len(sources))}\n\n"
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
