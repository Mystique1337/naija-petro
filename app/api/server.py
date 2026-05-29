"""FastAPI front door for the Naija-Petro RAG assistant.

`create_app(deps)` builds the app with GPU-backed callables injected, so this
module never imports Modal — keeping it cycle-free and unit-testable with a stub.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import observability as obs
from app.config import settings
from app.rag import db
from app.rag.prompts import build_messages
from app.rag.retriever import retrieve

FRONTEND_DIR = Path(os.environ.get("FRONTEND_DIR") or (Path(__file__).resolve().parent.parent / "frontend"))

FOLLOWUP_SYS = (
    "You suggest follow-up questions for a petroleum-engineering assistant focused on "
    "Nigeria. Output exactly three short, distinct questions (max 12 words each). "
    "Plain text only: one question per line, no numbering, no markdown, no bold, no "
    "preamble, no em-dashes. Do not use <think> tags."
)


def _clean_followup(line: str) -> str:
    line = line.replace("*", "").replace("#", "").replace("`", "")
    return line.strip(" -•\t0123456789.)").strip()


@dataclass
class Deps:
    """GPU/Modal-backed callables injected by the deployment layer."""
    embed: Callable[[list[str], str], Awaitable[list[list[float]]]]
    llm_stream: Callable[[list[dict], dict], "Awaitable"]      # async generator
    rerank: Optional[Callable[[str, list[str]], Awaitable[list[float]]]] = None
    spawn_enrich: Optional[Callable[[str], None]] = None
    llm_complete: Optional[Callable[[list[dict], dict], Awaitable[str]]] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    history: Optional[list[dict]] = None
    reasoning: bool = True


class FeedbackRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    query: Optional[str] = None
    rating: int = 0          # 1 = up, -1 = down
    trace_id: Optional[str] = None
    comment: Optional[str] = None
    answer: Optional[str] = None        # full exchange stored for training/preference data
    sources: Optional[list] = None


class SubscribeRequest(BaseModel):
    email: str
    wants_updates: bool = False


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _sse(kind: str, **data) -> str:
    return f"data: {json.dumps({'type': kind, **data})}\n\n"


# --- simple in-memory rate limiter (per web container) ---
_hits: dict = defaultdict(deque)


def _rate_ok(key: str, max_n: int, window_s: int) -> bool:
    now = time.time()
    dq = _hits[key]
    while dq and dq[0] < now - window_s:
        dq.popleft()
    if len(dq) >= max_n:
        return False
    dq.append(now)
    return True


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "0.0.0.0"


def _hash_ip(ip: str) -> str:
    return hashlib.sha256((ip + settings.ip_salt).encode()).hexdigest()[:32]


def create_app(deps: Deps) -> FastAPI:
    api = FastAPI(title="Naija-Petro", version="0.2.0")

    @api.get("/healthz")
    async def healthz():
        from app import __version__
        return {"status": "ok", "version": __version__}

    @api.get("/kb/stats")
    async def kb_stats():
        try:
            return await db.kb_stats()
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    @api.post("/feedback")
    async def feedback(fb: FeedbackRequest):
        try:
            await db.log_feedback(fb.model_dump())
        except Exception:
            pass
        try:
            client = obs.get_client()
            if client and fb.trace_id and fb.rating:
                client.create_score(name="user_feedback", value=fb.rating,
                                    trace_id=fb.trace_id, data_type="NUMERIC")
                client.flush()
        except Exception:
            pass
        return {"ok": True}

    @api.post("/subscribe")
    async def subscribe(req: SubscribeRequest):
        email = (req.email or "").strip().lower()
        if not _EMAIL_RE.match(email) or len(email) > 254:
            return JSONResponse({"error": "Please enter a valid email."}, status_code=400)
        try:
            await db.subscribe(email, bool(req.wants_updates), source="app")
        except Exception:
            return JSONResponse({"error": "Could not save right now."}, status_code=503)
        return {"ok": True}

    @api.post("/chat")
    async def chat(req: ChatRequest, request: Request):
        # --- gates (checked before we start streaming) ---
        if settings.access_key:
            key = request.headers.get("x-access-key") or request.query_params.get("key", "")
            if key != settings.access_key:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
        msg = (req.message or "").strip()
        if not msg:
            return JSONResponse({"error": "Empty message"}, status_code=400)
        if len(msg) > settings.max_query_chars:
            return JSONResponse({"error": f"Message too long (max {settings.max_query_chars})"}, status_code=400)

        ip_hash = _hash_ip(_client_ip(request))
        sess = req.session_id or ip_hash
        country = request.headers.get("cf-ipcountry") or request.headers.get("x-vercel-ip-country")
        if not (_rate_ok(f"m:{ip_hash}", settings.rate_limit_max, settings.rate_limit_window_s)
                and _rate_ok(f"h:{ip_hash}", settings.rate_limit_max_hour, 3600)
                and _rate_ok(f"m:{sess}", settings.rate_limit_max, settings.rate_limit_window_s)):
            return JSONResponse({"error": "Rate limit exceeded. Please wait a moment."}, status_code=429)

        sampling = {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_new_tokens,
            "reasoning": req.reasoning,
        }

        async def gen():
            t0 = time.time()
            result = None
            answer = ""
            yield _sse("status", stage="starting")
            try:
                with obs.trace("chat", session_id=sess, user_id=req.user_id, input=msg) as tr:
                    rerank_fn = deps.rerank if (settings.enable_rerank and deps.rerank) else None

                    # Retrieve, heartbeating while we wait (Encoders may be cold).
                    rtask = asyncio.ensure_future(retrieve(msg, deps.embed, rerank_fn))
                    while True:
                        try:
                            result = await asyncio.wait_for(asyncio.shield(rtask), timeout=8)
                            break
                        except asyncio.TimeoutError:
                            yield _sse("status", stage="searching")
                    messages, sources = build_messages(msg, result.chunks, req.history, reasoning=req.reasoning)

                    tr.update(metadata={
                        "coverage": round(result.coverage, 3),
                        "candidates": result.candidates,
                        "enriched": result.enriched,
                        "reranked": result.reranked,
                        "kb_added": result.enrichment.get("new_chunks", 0),
                    })

                    if settings.always_enrich and not result.enriched and deps.spawn_enrich:
                        try:
                            deps.spawn_enrich(msg)
                        except Exception:
                            pass

                    yield _sse(
                        "meta",
                        sources=sources,
                        coverage=round(result.coverage, 3),
                        enriched=result.enriched,
                        kb_added=result.enrichment.get("new_chunks", 0),
                        reranked=result.reranked,
                        trace_id=obs.current_trace_id(),
                    )

                    # Stream tokens, heartbeating until the first token (LLM may be cold).
                    parts: list[str] = []
                    agen = deps.llm_stream(messages, sampling).__aiter__()
                    nxt = asyncio.ensure_future(agen.__anext__())
                    while True:
                        try:
                            tok = await asyncio.wait_for(asyncio.shield(nxt), timeout=8)
                        except asyncio.TimeoutError:
                            yield _sse("status", stage="reasoning")
                            continue
                        except StopAsyncIteration:
                            break
                        parts.append(tok)
                        yield _sse("token", t=tok)
                        nxt = asyncio.ensure_future(agen.__anext__())

                    answer = "".join(parts)
                    tr.generation("llm-generation", settings.model_repo, messages, answer)

                    # Suggested follow-up questions (best-effort, non-streaming).
                    if settings.enable_followups and deps.llm_complete and answer:
                        try:
                            fu_msgs = [
                                {"role": "system", "content": FOLLOWUP_SYS},
                                {"role": "user", "content": f"Question: {msg}\n\nAnswer: {answer[:1500]}\n\nThree follow-up questions:"},
                            ]
                            fu = await deps.llm_complete(fu_msgs, {"max_tokens": 140, "temperature": 0.5, "reasoning": False})
                            qs = [_clean_followup(l) for l in fu.splitlines() if l.strip()]
                            qs = [q for q in qs if 8 < len(q) <= 120][:3]
                            if qs:
                                yield _sse("followups", items=qs)
                        except Exception:
                            pass

                    yield _sse("done", chars=len(answer))
            except Exception as e:  # surface errors to the client cleanly
                yield _sse("error", message=str(e))
            finally:
                # Analytics (best-effort, never blocks or raises into the stream).
                try:
                    await db.log_usage({
                        "session_id": sess, "user_id": req.user_id, "ip_hash": ip_hash,
                        "country": country, "query": msg[:1000], "answer_chars": len(answer),
                        "n_sources": (len(result.chunks) if result else 0),
                        "coverage": (round(result.coverage, 4) if result else 0.0),
                        "enriched": (bool(result.enriched) if result else False),
                        "kb_added": (result.enrichment.get("new_chunks", 0) if result else 0),
                        "reasoning": req.reasoning, "latency_ms": int((time.time() - t0) * 1000),
                    })
                except Exception:
                    pass

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Static frontend. no-store on index so UI updates aren't stuck in browser cache.
    if FRONTEND_DIR.exists():
        @api.get("/")
        async def index():
            return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-store"})

        api.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    return api
