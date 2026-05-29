"""Async Postgres access for the vector store (asyncpg + pgvector).

Talks to the database over the direct Postgres connection string
(`SUPABASE_DB_URL`), which works for a Railway-hosted Supabase/Postgres as
well as a full self-hosted Supabase.
"""
from __future__ import annotations

import os

from app.config import settings

_pool: "asyncpg.Pool | None" = None


def _ssl_modes() -> list:
    """Ordered SSL options to try, mirroring libpq sslmode=prefer.

    `SUPABASE_DB_SSL` forces a single mode (require/disable). Otherwise we try
    TLS first then fall back to plaintext on an SSL-specific failure — which is
    what a real Supabase (needs TLS) and a Railway TCP proxy (rejects TLS) each
    require, without hardcoding host patterns.
    """
    override = os.environ.get("SUPABASE_DB_SSL", "auto").lower()
    if override in ("0", "false", "disable", "off"):
        return [False]
    if override in ("1", "true", "require", "on"):
        return [True]
    return [True, False]


async def _init_conn(conn) -> None:
    from pgvector.asyncpg import register_vector

    await register_vector(conn)


async def get_pool() -> "asyncpg.Pool":
    global _pool
    if _pool is None:
        import asyncpg

        if not settings.supabase_db_url:
            raise RuntimeError("SUPABASE_DB_URL is not set")
        last_err = None
        for ssl_opt in _ssl_modes():
            pool = None
            try:
                pool = await asyncpg.create_pool(
                    dsn=settings.supabase_db_url,
                    ssl=ssl_opt,
                    min_size=1,
                    max_size=8,
                    init=_init_conn,
                    command_timeout=30,
                )
                await pool.fetchval("SELECT 1")  # force a real connection now
                _pool = pool
                break
            except Exception as e:  # fall back to plaintext only on SSL errors
                last_err = e
                if pool is not None:
                    try:
                        await pool.close()
                    except Exception:
                        pass
                if "ssl" in str(e).lower():
                    continue
                raise
        if _pool is None:
            raise last_err
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
async def document_exists(content_hash: str) -> bool:
    pool = await get_pool()
    row = await pool.fetchval("SELECT 1 FROM documents WHERE content_hash = $1", content_hash)
    return row is not None


async def upsert_document(doc: dict, chunks: list[dict]) -> dict:
    """Insert a document + its chunks. Idempotent on content_hash.

    doc:    title, source_url, domain, source_tier, published_date, content,
            content_hash, metadata
    chunks: list of {content, embedding (list[float]), token_count, metadata}
    Returns {"inserted": bool, "document_id": str|None, "chunk_count": int}.
    """
    import json

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            doc_id = await conn.fetchval(
                """
                INSERT INTO documents
                    (title, source_url, domain, source_tier, published_date,
                     content, content_hash, metadata)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id
                """,
                doc.get("title"), doc.get("source_url"), doc.get("domain"),
                doc.get("source_tier", 3), doc.get("published_date"),
                doc["content"], doc["content_hash"],
                json.dumps(doc.get("metadata", {})),
            )
            if doc_id is None:  # already present → skip (dedup)
                return {"inserted": False, "document_id": None, "chunk_count": 0}

            await conn.executemany(
                """
                INSERT INTO document_chunks
                    (document_id, chunk_index, content, embedding, token_count, metadata)
                VALUES ($1,$2,$3,$4,$5,$6)
                ON CONFLICT (document_id, chunk_index) DO NOTHING
                """,
                [
                    (doc_id, i, c["content"], c["embedding"],
                     c.get("token_count"), json.dumps(c.get("metadata", {})))
                    for i, c in enumerate(chunks)
                ],
            )
    return {"inserted": True, "document_id": str(doc_id), "chunk_count": len(chunks)}


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
async def hybrid_search(query_text: str, query_embedding: list[float], match_count: int = 20) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM hybrid_search($1, $2, $3)",
        query_text, query_embedding, match_count,
    )
    return [dict(r) for r in rows]


async def match_documents(query_embedding: list[float], match_count: int = 10, threshold: float = 0.0) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM match_documents($1, $2, $3)",
        query_embedding, match_count, threshold,
    )
    return [dict(r) for r in rows]


async def kb_stats() -> dict:
    pool = await get_pool()
    row = await pool.fetchrow("SELECT * FROM kb_stats()")
    return dict(row) if row else {"documents": 0, "chunks": 0, "last_ingest": None}
