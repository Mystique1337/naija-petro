"""Data access for the RAG store over the Supabase REST (PostgREST) API.

The store lives in a self-hosted Supabase; we reach it over HTTPS (PostgREST +
RPC) rather than a direct Postgres connection, because the REST gateway is the
publicly reachable interface (works from Modal) and needs no exposed database
port. All calls use the service-role key server-side and are pinned to the app's
own schema (`SUPABASE_DB_SCHEMA`, e.g. naija_petro) via the Accept/Content-Profile
headers.

Vector similarity (match_documents / hybrid_search) and the kb_stats counts run
inside Postgres functions exposed as RPC; everything else is plain table
reads/writes. Function signatures and return shapes match the previous asyncpg
implementation, so callers are unchanged.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.config import settings

_client: "httpx.AsyncClient | None" = None
_lock = asyncio.Lock()


def _rest_base() -> str:
    if not settings.supabase_url:
        raise RuntimeError("SUPABASE_URL is not set")
    if not settings.supabase_service_key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is not set")
    return f"{settings.supabase_url}/rest/v1"


def _headers(*, write: bool = False, prefer: str | None = None) -> dict:
    key = settings.supabase_service_key
    schema = settings.supabase_db_schema or "public"
    h = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept-Profile": schema,       # schema to read from
        "Content-Profile": schema,      # schema to write to / call RPC in
    }
    if write:
        h["Content-Type"] = "application/json"
    if prefer:
        h["Prefer"] = prefer
    return h


async def get_client() -> "httpx.AsyncClient":
    global _client
    if _client is None:
        async with _lock:
            if _client is None:
                import httpx

                _client = httpx.AsyncClient(base_url=_rest_base(), timeout=30.0)
    return _client


async def close_pool() -> None:  # name kept for backwards compatibility
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _vec(embedding: list[float]) -> str:
    """pgvector text literal: PostgREST casts it to vector on the far side."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


def _start_of_day_utc() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


async def _get(path: str, params: dict, *, count: bool = False):
    c = await get_client()
    h = _headers(prefer="count=exact") if count else _headers()
    r = await c.get(path, params=params, headers=h)
    r.raise_for_status()
    if count:
        total = None
        cr = r.headers.get("content-range", "")
        if "/" in cr and cr.rsplit("/", 1)[1].isdigit():
            total = int(cr.rsplit("/", 1)[1])
        return r.json(), total
    return r.json()


async def _post(path: str, json, *, prefer: str = "return=representation"):
    c = await get_client()
    r = await c.post(path, json=json, headers=_headers(write=True, prefer=prefer))
    r.raise_for_status()
    if r.status_code == 204 or not r.content:
        return []
    return r.json()


async def _rpc(name: str, args: dict):
    c = await get_client()
    r = await c.post(f"/rpc/{name}", json=args, headers=_headers(write=True))
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
async def document_exists(content_hash: str) -> bool:
    rows = await _get("/documents", {"content_hash": f"eq.{content_hash}", "select": "id", "limit": 1})
    return bool(rows)


async def upsert_document(doc: dict, chunks: list[dict]) -> dict:
    """Insert a document + its chunks. Idempotent on content_hash.

    doc:    title, source_url, domain, source_tier, published_date, content,
            content_hash, metadata
    chunks: list of {content, embedding (list[float]), token_count, metadata}
    Returns {"inserted": bool, "document_id": str|None, "chunk_count": int}.
    """
    body = {
        "title": doc.get("title"),
        "source_url": doc.get("source_url"),
        "domain": doc.get("domain"),
        "source_tier": doc.get("source_tier", 3),
        "published_date": doc.get("published_date"),
        "content": doc["content"],
        "content_hash": doc["content_hash"],
        "metadata": doc.get("metadata", {}),
    }
    # ignore-duplicates: a conflicting content_hash yields an empty representation.
    rows = await _post(
        "/documents?on_conflict=content_hash",
        body,
        prefer="resolution=ignore-duplicates,return=representation",
    )
    if not rows:  # already present -> dedup
        return {"inserted": False, "document_id": None, "chunk_count": 0}

    doc_id = rows[0]["id"]
    if chunks:
        payload = [
            {
                "document_id": doc_id,
                "chunk_index": i,
                "content": c["content"],
                "embedding": _vec(c["embedding"]),
                "token_count": c.get("token_count"),
                "metadata": c.get("metadata", {}),
            }
            for i, c in enumerate(chunks)
        ]
        await _post(
            "/document_chunks?on_conflict=document_id,chunk_index",
            payload,
            prefer="resolution=ignore-duplicates,return=minimal",
        )
    return {"inserted": True, "document_id": str(doc_id), "chunk_count": len(chunks)}


# --------------------------------------------------------------------------- #
# Reads (vector search + counts run inside Postgres functions via RPC)
# --------------------------------------------------------------------------- #
async def hybrid_search(query_text: str, query_embedding: list[float], match_count: int = 20) -> list[dict]:
    return await _rpc(
        "hybrid_search",
        {"query_text": query_text, "query_embedding": _vec(query_embedding), "match_count": match_count},
    )


async def match_documents(query_embedding: list[float], match_count: int = 10, threshold: float = 0.0) -> list[dict]:
    return await _rpc(
        "match_documents",
        {"query_embedding": _vec(query_embedding), "match_count": match_count, "similarity_threshold": threshold},
    )


async def kb_stats() -> dict:
    rows = await _rpc("kb_stats", {})
    if rows:
        return rows[0]
    return {"documents": 0, "chunks": 0, "last_ingest": None}


# --------------------------------------------------------------------------- #
# Analytics + feedback
# --------------------------------------------------------------------------- #
async def log_usage(e: dict) -> None:
    cols = ("session_id", "user_id", "ip_hash", "country", "query", "answer_chars",
            "n_sources", "coverage", "enriched", "kb_added", "reasoning", "latency_ms")
    await _post("/usage_events", {k: e.get(k) for k in cols}, prefer="return=minimal")


async def log_feedback(f: dict) -> None:
    await _post(
        "/feedback",
        {
            "session_id": f.get("session_id"),
            "user_id": f.get("user_id"),
            "query": f.get("query"),
            "rating": f.get("rating"),
            "trace_id": f.get("trace_id"),
            "comment": f.get("comment"),
            "answer": f.get("answer"),
            "sources": f.get("sources"),  # jsonb; pass the object as-is
        },
        prefer="return=minimal",
    )


async def subscribe(email: str, wants_updates: bool = False, source: str | None = None) -> None:
    await _post(
        "/subscribers?on_conflict=email",
        {"email": email, "wants_updates": wants_updates, "source": source},
        prefer="resolution=merge-duplicates,return=minimal",
    )


# --------------------------------------------------------------------------- #
# Feature requests
# --------------------------------------------------------------------------- #
async def add_feature(text: str, email: str | None, session_id: str | None) -> None:
    await _post("/feature_requests", {"text": text, "email": email, "session_id": session_id},
                prefer="return=minimal")


async def list_features(limit: int = 20) -> list[dict]:
    return await _get("/feature_requests", {
        "select": "text,votes,created_at",
        "order": "votes.desc,created_at.desc",
        "limit": limit,
    })


# --------------------------------------------------------------------------- #
# Saved chat history (anonymous)
# --------------------------------------------------------------------------- #
async def save_turns(user_id: str, session_id: str, turns: list[tuple]) -> None:
    if not turns:
        return
    payload = [
        {"user_id": user_id, "session_id": session_id, "role": role, "content": content}
        for role, content in turns
    ]
    await _post("/conversations", payload, prefer="return=minimal")


async def list_sessions(user_id: str, limit: int = 20) -> list[dict]:
    """One row per session: session_id, last activity, and first user message as title.

    Aggregated client-side over this user's turns (a single user's history is
    small); mirrors the previous GROUP BY / array_agg query.
    """
    rows = await _get("/conversations", {
        "user_id": f"eq.{user_id}",
        "select": "session_id,role,content,created_at",
        "order": "created_at.asc",
        "limit": 5000,
    })
    sessions: dict[str, dict] = {}
    for r in rows:
        sid = r["session_id"]
        s = sessions.setdefault(sid, {"session_id": sid, "last": None, "title": None})
        s["last"] = r["created_at"]  # rows are ascending, so this ends on the max
        if s["title"] is None and r["role"] == "user":
            s["title"] = r["content"]
    out = sorted(sessions.values(), key=lambda s: s["last"] or "", reverse=True)
    return out[:limit]


async def load_session(session_id: str, limit: int = 100) -> list[dict]:
    return await _get("/conversations", {
        "session_id": f"eq.{session_id}",
        "select": "role,content",
        "order": "created_at.asc",
        "limit": limit,
    })


# --------------------------------------------------------------------------- #
# Access tokens + daily limit
# --------------------------------------------------------------------------- #
async def token_active(token: str) -> bool:
    if not token:
        return False
    rows = await _get("/access_tokens", {
        "token": f"eq.{token}", "active": "is.true", "select": "id", "limit": 1,
    })
    return bool(rows)


async def daily_ip_count(ip_hash: str) -> int:
    _, total = await _get("/usage_events", {
        "ip_hash": f"eq.{ip_hash}",
        "created_at": f"gte.{_start_of_day_utc()}",
        "select": "id",
        "limit": 1,
    }, count=True)
    return int(total or 0)


async def list_tokens() -> list[dict]:
    return await _get("/access_tokens", {
        "select": "id,token,label,kind,active,created_at",
        "order": "kind.asc,id.asc",
    })


async def set_token_active(token_id: int, active: bool) -> None:
    c = await get_client()
    r = await c.patch(f"/access_tokens?id=eq.{token_id}",
                      json={"active": active},
                      headers=_headers(write=True, prefer="return=minimal"))
    r.raise_for_status()


async def create_token(token: str, label: str, kind: str = "secondary") -> None:
    await _post(
        "/access_tokens?on_conflict=token",
        {"token": token, "label": label, "kind": kind, "active": True},
        prefer="resolution=ignore-duplicates,return=minimal",
    )


async def count_tokens_by_kind() -> dict:
    rows = await _get("/access_tokens", {"select": "kind"})
    out: dict[str, int] = {}
    for r in rows:
        out[r["kind"]] = out.get(r["kind"], 0) + 1
    return out


async def usage_overview(days: int = 14) -> dict:
    """Headline usage numbers + recent daily breakdown for the admin panel."""
    summary_rows = await _get("/usage_summary", {"select": "*", "limit": 1})
    daily = await _get("/usage_daily", {"select": "*", "order": "day.desc", "limit": days})
    _, today = await _get("/usage_events", {
        "created_at": f"gte.{_start_of_day_utc()}", "select": "id", "limit": 1,
    }, count=True)
    return {
        "summary": summary_rows[0] if summary_rows else {},
        "today": int(today or 0),
        "daily": daily,
    }
