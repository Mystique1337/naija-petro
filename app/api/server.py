"""FastAPI front door for the Naija-Petro RAG assistant.

`create_app(deps)` builds the app with GPU-backed callables injected, so this
module never imports Modal — keeping it cycle-free and unit-testable with a stub.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import observability as obs
from app.config import settings
from app.rag import db
from app.rag.prompts import build_messages
from app.rag.retriever import retrieve

FRONTEND_DIR = Path(os.environ.get("FRONTEND_DIR") or (Path(__file__).resolve().parent.parent / "frontend"))


@dataclass
class Deps:
    """GPU/Modal-backed callables injected by the deployment layer."""
    embed: Callable[[list[str], str], Awaitable[list[list[float]]]]
    llm_stream: Callable[[list[dict], dict], "Awaitable"]      # async generator
    rerank: Optional[Callable[[str, list[str]], Awaitable[list[float]]]] = None
    spawn_enrich: Optional[Callable[[str], None]] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    history: Optional[list[dict]] = None


def _sse(kind: str, **data) -> str:
    return f"data: {json.dumps({'type': kind, **data})}\n\n"


def create_app(deps: Deps) -> FastAPI:
    api = FastAPI(title="Naija-Petro", version="0.1.0")

    @api.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @api.get("/kb/stats")
    async def kb_stats():
        try:
            return await db.kb_stats()
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    @api.post("/chat")
    async def chat(req: ChatRequest):
        sampling = {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_new_tokens,
        }

        async def gen():
            try:
                with obs.trace(
                    "chat",
                    session_id=req.session_id,
                    user_id=req.user_id,
                    input=req.message,
                ) as tr:
                    rerank_fn = deps.rerank if (settings.enable_rerank and deps.rerank) else None
                    result = await retrieve(req.message, deps.embed, rerank_fn)
                    messages, sources = build_messages(req.message, result.chunks, req.history)

                    tr.update(metadata={
                        "coverage": round(result.coverage, 3),
                        "candidates": result.candidates,
                        "enriched": result.enriched,
                        "reranked": result.reranked,
                        "kb_added": result.enrichment.get("new_chunks", 0),
                    })

                    # Self-update after every query: if we didn't already fetch
                    # inline, kick off a non-blocking background enrichment.
                    if settings.always_enrich and not result.enriched and deps.spawn_enrich:
                        try:
                            deps.spawn_enrich(req.message)
                        except Exception:
                            pass

                    yield _sse(
                        "meta",
                        sources=sources,
                        coverage=round(result.coverage, 3),
                        enriched=result.enriched,
                        kb_added=result.enrichment.get("new_chunks", 0),
                        reranked=result.reranked,
                    )

                    parts: list[str] = []
                    async for tok in deps.llm_stream(messages, sampling):
                        parts.append(tok)
                        yield _sse("token", t=tok)

                    answer = "".join(parts)
                    tr.generation("llm-generation", settings.model_repo, messages, answer)
                    yield _sse("done", chars=len(answer))
            except Exception as e:  # surface errors to the client cleanly
                yield _sse("error", message=str(e))

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Static frontend (index.html at /, assets under /static).
    if FRONTEND_DIR.exists():
        @api.get("/")
        async def index():
            return FileResponse(FRONTEND_DIR / "index.html")

        api.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    return api
