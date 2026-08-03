"""Retrieval orchestration: coverage assessment, enrichment and reranking.

Everything the retriever touches (embedding, the store, live ingestion) is
injected or monkeypatched, so these tests run offline in milliseconds.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from app.rag import db, ingest, retriever
from app.rag.retriever import _cap_per_source
from app.rag.prompts import RetrievedChunk


@dataclass(frozen=True)
class FakeSettings:
    """Only the knobs the retriever reads, pinned so the env cannot move them."""
    min_chunks: int = 3
    coverage_threshold: float = 0.55
    top_k: int = 10
    final_k: int = 4
    max_per_source: int = 3
    min_sources: int = 3


@pytest.fixture(autouse=True)
def pinned_settings(monkeypatch):
    fake = FakeSettings()
    monkeypatch.setattr(retriever, "settings", fake)
    return fake


def row(i: int, similarity: float) -> dict:
    return {
        "content": f"chunk {i}",
        "source_url": f"https://example{i}.com/doc",
        "title": f"Doc {i}",
        "domain": f"example{i}.com",
        "source_tier": 1,
        "score": 0.5,
        "similarity": similarity,
    }


async def fake_embed(texts, mode):
    assert mode in ("query", "document")
    return [[0.1, 0.2, 0.3] for _ in texts]


def install_search(monkeypatch, batches):
    """Serve `batches` (a list of row-lists) from successive hybrid_search calls."""
    calls = []

    async def _search(query_text, query_embedding, match_count=20):
        calls.append({"query": query_text, "match_count": match_count,
                      "embedding": query_embedding})
        return batches[min(len(calls) - 1, len(batches) - 1)]

    monkeypatch.setattr(db, "hybrid_search", _search)
    return calls


def install_ingest(monkeypatch, result):
    calls = []

    async def _ingest(query, embed_fn, max_results=None):
        calls.append(query)
        return result

    monkeypatch.setattr(ingest, "ingest_query", _ingest)
    return calls


# --------------------------------------------------------------------------- #
# is_weak
# --------------------------------------------------------------------------- #
def test_is_weak_when_too_few_rows(pinned_settings):
    rows = [row(i, 0.99) for i in range(pinned_settings.min_chunks - 1)]
    assert retriever.is_weak(rows) is True


def test_is_weak_on_no_rows_at_all():
    assert retriever.is_weak([]) is True


def test_is_weak_when_similarity_is_below_the_threshold(pinned_settings):
    rows = [row(i, pinned_settings.coverage_threshold - 0.01) for i in range(5)]
    assert retriever.is_weak(rows) is True


def test_not_weak_with_enough_rows_and_good_similarity(pinned_settings):
    rows = [row(i, 0.2) for i in range(5)]
    rows[2]["similarity"] = pinned_settings.coverage_threshold + 0.01   # max is what counts
    assert retriever.is_weak(rows) is False


def test_missing_similarity_counts_as_zero():
    rows = [{"content": "c"} for _ in range(9)]
    assert retriever.is_weak(rows) is True


# --------------------------------------------------------------------------- #
# retrieve
# --------------------------------------------------------------------------- #
def test_strong_coverage_skips_enrichment(monkeypatch, pinned_settings):
    rows = [row(i, 0.8) for i in range(6)]
    searches = install_search(monkeypatch, [rows])
    ingests = install_ingest(monkeypatch, {"ingested_docs": 1, "new_chunks": 9})

    result = asyncio.run(retriever.retrieve("nigerian gas flaring rules", fake_embed))

    assert ingests == [], "strong local coverage must not trigger a live fetch"
    assert len(searches) == 1
    assert searches[0]["match_count"] == pinned_settings.top_k
    assert result.enriched is False
    assert result.enrichment == {}
    assert result.reranked is False
    assert result.coverage == pytest.approx(0.8)
    assert result.candidates == 6
    assert len(result.chunks) == pinned_settings.final_k
    assert isinstance(result.chunks[0], RetrievedChunk)
    assert result.chunks[0].content == "chunk 0"
    assert result.chunks[0].source_tier == 1


def test_weak_coverage_enriches_and_searches_again(monkeypatch):
    weak = [row(0, 0.10)]
    strong = [row(i, 0.77) for i in range(5)]
    searches = install_search(monkeypatch, [weak, strong])
    ingests = install_ingest(monkeypatch, {"ingested_docs": 2, "new_chunks": 17, "searched": 4})

    result = asyncio.run(retriever.retrieve("obscure nigerian marginal field", fake_embed))

    assert ingests == ["obscure nigerian marginal field"]
    assert len(searches) == 2, "the enriched store must be searched again this turn"
    assert result.enriched is True
    assert result.enrichment["new_chunks"] == 17
    assert result.coverage == pytest.approx(0.77)
    assert result.candidates == 5


def test_weak_coverage_that_ingests_nothing_keeps_the_first_result(monkeypatch):
    weak = [row(0, 0.10)]
    searches = install_search(monkeypatch, [weak, [row(i, 0.9) for i in range(5)]])
    install_ingest(monkeypatch, {"ingested_docs": 0, "new_chunks": 0, "searched": 3})

    result = asyncio.run(retriever.retrieve("nothing new on the web", fake_embed))

    assert len(searches) == 1, "no re-search when nothing was ingested"
    assert result.enriched is False
    assert result.enrichment["searched"] == 3
    assert result.coverage == pytest.approx(0.10)
    assert result.candidates == 1


def test_rerank_reorders_rows(monkeypatch, pinned_settings):
    rows = [row(i, 0.8) for i in range(5)]
    install_search(monkeypatch, [rows])
    install_ingest(monkeypatch, {"ingested_docs": 0})
    seen = {}

    async def rerank(query, passages):
        seen["query"] = query
        seen["passages"] = passages
        return [float(i) for i in range(len(passages))]        # last row scores highest

    result = asyncio.run(retriever.retrieve("q", fake_embed, rerank))

    assert result.reranked is True
    assert seen["query"] == "q"
    assert seen["passages"] == [f"chunk {i}" for i in range(5)]
    assert [c.content for c in result.chunks] == [
        f"chunk {i}" for i in (4, 3, 2, 1)
    ][: pinned_settings.final_k]
    assert result.candidates == 5, "candidates counts the pre-rerank hits"


def test_rerank_with_a_mismatched_score_count_is_ignored(monkeypatch):
    rows = [row(i, 0.8) for i in range(5)]
    install_search(monkeypatch, [rows])

    async def rerank(query, passages):
        return [1.0, 2.0]                                       # wrong length

    result = asyncio.run(retriever.retrieve("q", fake_embed, rerank))

    assert result.reranked is False
    assert [c.content for c in result.chunks] == ["chunk 0", "chunk 1", "chunk 2", "chunk 3"]


def test_rerank_is_skipped_when_there_are_no_rows(monkeypatch):
    install_search(monkeypatch, [[]])
    install_ingest(monkeypatch, {"ingested_docs": 0})
    called = []

    async def rerank(query, passages):
        called.append(query)
        return [1.0]

    result = asyncio.run(retriever.retrieve("q", fake_embed, rerank))

    assert called == []
    assert result.chunks == []
    assert result.candidates == 0
    assert result.coverage == 0.0


def test_rows_are_mapped_with_safe_defaults(monkeypatch):
    install_search(monkeypatch, [[{"content": "bare row"}] * 5])
    install_ingest(monkeypatch, {"ingested_docs": 0})

    result = asyncio.run(retriever.retrieve("q", fake_embed))
    c = result.chunks[0]
    assert (c.content, c.source_url, c.title, c.domain) == ("bare row", "", "", "")
    assert c.source_tier == 3
    assert c.score == 0.0 and c.similarity == 0.0


# --------------------------------------------------------------------------- #
# _cap_per_source: a long PDF has hundreds of chances to rank and used to take
# every slot, so answers cited one magazine or manual over and over
# --------------------------------------------------------------------------- #
def _row(doc_id, i):
    return {"document_id": doc_id, "content": f"chunk {i}", "source_url": f"http://x/{doc_id}"}


def test_cap_per_source_limits_one_document():
    rows = [_row("A", i) for i in range(12)]
    kept = _cap_per_source(rows, 3, 12)
    assert sum(1 for r in kept if r["document_id"] == "A") == 12  # overflow backfills
    assert [r["content"] for r in kept[:3]] == ["chunk 0", "chunk 1", "chunk 2"]


def test_cap_per_source_prefers_diverse_documents():
    rows = [_row("A", 0), _row("A", 1), _row("A", 2), _row("A", 3), _row("B", 0), _row("C", 0)]
    kept = _cap_per_source(rows, 2, 4)
    docs = [r["document_id"] for r in kept]
    assert docs.count("A") == 2
    assert "B" in docs and "C" in docs


def test_cap_per_source_returns_final_k_at_most():
    rows = [_row(str(i), 0) for i in range(20)]
    assert len(_cap_per_source(rows, 3, 12)) == 12


def test_cap_per_source_disabled_when_zero():
    rows = [_row("A", i) for i in range(5)]
    assert _cap_per_source(rows, 0, 3) == rows[:3]


def test_cap_per_source_falls_back_to_source_url():
    rows = [{"source_url": "http://a", "content": str(i)} for i in range(5)]
    kept = _cap_per_source(rows, 2, 3)
    assert len(kept) == 3


# --------------------------------------------------------------------------- #
# is_weak also demands breadth: one loosely matching page used to score high
# enough to suppress enrichment, and single-source answers are where the model
# invents specifics
# --------------------------------------------------------------------------- #
def test_single_source_is_weak_even_with_high_similarity():
    rows = [{"document_id": "A", "similarity": 0.95} for _ in range(6)]
    assert retriever.is_weak(rows) is True


def test_enough_distinct_sources_is_not_weak():
    rows = [{"document_id": d, "similarity": 0.8} for d in ("A", "B", "C", "D")]
    assert retriever.is_weak(rows) is False


def test_low_similarity_is_still_weak_with_many_sources():
    rows = [{"document_id": d, "similarity": 0.1} for d in ("A", "B", "C", "D")]
    assert retriever.is_weak(rows) is True
