"""Retrieval orchestration: the heart of the dynamic RAG.

    embed query → hybrid search → assess coverage → (if weak) fetch live &
    re-retrieve → rerank → top-k cited chunks.

Embedding and reranking are injected (async) so they can run in Modal GPU
containers while this runs in the web container.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.config import settings
from app.rag import db, ingest
from app.rag.prompts import RetrievedChunk

EmbedFn = Callable[[list[str], str], Awaitable[list[list[float]]]]
RerankFn = Callable[[str, list[str]], Awaitable[list[float]]]


@dataclass
class RetrieveResult:
    chunks: list[RetrievedChunk]
    coverage: float = 0.0           # max semantic similarity seen locally
    candidates: int = 0             # raw hits before rerank
    enriched: bool = False          # did we fetch live this turn?
    enrichment: dict = field(default_factory=dict)
    reranked: bool = False


def _rows_to_chunks(rows: list[dict]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            content=r["content"],
            source_url=r.get("source_url") or "",
            title=r.get("title") or "",
            domain=r.get("domain") or "",
            source_tier=r.get("source_tier") or 3,
            score=float(r.get("score") or 0.0),
            similarity=float(r.get("similarity") or 0.0),
        )
        for r in rows
    ]


def _coverage(rows: list[dict]) -> float:
    return max((float(r.get("similarity") or 0.0) for r in rows), default=0.0)


def _cap_per_source(rows: list[dict], max_per_source: int, final_k: int) -> list[dict]:
    """Keep the best `final_k` rows, but no more than `max_per_source` per document.

    A 700 chunk PDF has 700 chances to rank and routinely took every slot, so the
    answer cited one magazine or manual repeatedly no matter what was asked. Rows
    stay in relevance order; over-quota chunks are held back and only used to fill
    the tail if there is nothing else, so a genuinely single-source question still
    gets a full context.
    """
    if max_per_source <= 0:
        return rows[:final_k]
    counts: dict = {}
    kept, overflow = [], []
    for r in rows:
        key = r.get("document_id") or r.get("source_url") or r.get("title") or ""
        if counts.get(key, 0) < max_per_source:
            counts[key] = counts.get(key, 0) + 1
            kept.append(r)
            if len(kept) == final_k:
                return kept
        else:
            overflow.append(r)
    return (kept + overflow)[:final_k]


def _distinct_sources(rows: list[dict], limit: int) -> int:
    """How many different documents the best `limit` hits come from."""
    return len({(r.get("document_id") or r.get("source_url") or r.get("title") or "")
                for r in rows[:limit]})


def is_weak(rows: list[dict]) -> bool:
    """Local knowledge is insufficient, so fetch live.

    Coverage alone is the maximum similarity of any single chunk, so one loosely
    matching page scores high and suppresses enrichment even when the store holds
    nothing else on the subject. A question answered from a single document is
    exactly where this assistant invents specifics, so breadth counts too.
    """
    if len(rows) < settings.min_chunks:
        return True
    if _coverage(rows) < settings.coverage_threshold:
        return True
    return _distinct_sources(rows, settings.final_k) < settings.min_sources


async def _search(query: str, embed_fn: EmbedFn) -> list[dict]:
    qvec = (await embed_fn([query], "query"))[0]
    return await db.hybrid_search(query, qvec, match_count=settings.top_k)


async def retrieve(query: str, embed_fn: EmbedFn, rerank_fn: RerankFn | None = None) -> RetrieveResult:
    rows = await _search(query, embed_fn)
    coverage = _coverage(rows)
    enriched, enrichment = False, {}

    # Dynamic step: if the store can't answer confidently, fetch verifiable
    # sources now and re-retrieve so this very turn benefits.
    if is_weak(rows):
        enrichment = await ingest.ingest_query(query, embed_fn)
        enriched = enrichment.get("ingested_docs", 0) > 0
        if enriched:
            rows = await _search(query, embed_fn)
            coverage = _coverage(rows)

    candidates = len(rows)

    # Rerank the candidate set with a cross-encoder for precision, then trim.
    reranked = False
    if rerank_fn and rows:
        scores = await rerank_fn(query, [r["content"] for r in rows])
        if scores and len(scores) == len(rows):
            order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
            rows = [rows[i] for i in order]
            reranked = True

    top = _cap_per_source(rows, settings.max_per_source, settings.final_k)
    return RetrieveResult(
        chunks=_rows_to_chunks(top),
        coverage=coverage,
        candidates=candidates,
        enriched=enriched,
        enrichment=enrichment,
        reranked=reranked,
    )
