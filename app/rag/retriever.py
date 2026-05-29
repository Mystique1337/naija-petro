"""Retrieval orchestration — the heart of the dynamic RAG.

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


def is_weak(rows: list[dict]) -> bool:
    """Local knowledge is insufficient → we should fetch live."""
    if len(rows) < settings.min_chunks:
        return True
    return _coverage(rows) < settings.coverage_threshold


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

    top = rows[: settings.final_k]
    return RetrieveResult(
        chunks=_rows_to_chunks(top),
        coverage=coverage,
        candidates=candidates,
        enriched=enriched,
        enrichment=enrichment,
        reranked=reranked,
    )
