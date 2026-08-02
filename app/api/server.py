"""FastAPI front door for the Naija-Petro RAG assistant.

`create_app(deps)` builds the app with GPU-backed callables injected, so this
module never imports Modal, keeping it cycle-free and unit-testable with a stub.
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

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import observability as obs
from app.config import CONTINUE_INSTRUCTION, STREAM_TRUNCATED, SYSTEM_PROMPT, settings
from app.rag import db
from app.rag.prompts import build_messages
from app.rag.retriever import retrieve
from app.tools.calculators import CALCULATORS, TOOL_MENU, TOOL_SPECS, needs_tools, run_tool

TOOL_SELECT_SYS = (
    "You pick a calculator for a petroleum-engineering question. Given the list and the "
    "question, if exactly one calculator clearly applies and its arguments can be filled "
    "with numbers from the question, reply with ONLY a JSON object "
    '{"tool": "<name>", "args": {...}} using numeric values. If none applies or required '
    "numbers are missing, reply with exactly {}. No prose, no markdown, no <think> tags."
)


def _parse_json_obj(text: str):
    import json
    s = (text or "").strip()
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        return json.loads(s[a:b + 1])
    except Exception:
        return None

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
    units: str = "field"          # "field" or "si"
    continuation: bool = False    # continue a previously truncated answer (no new retrieval)


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


class ToolRunRequest(BaseModel):
    name: str
    args: dict = {}


class FeatureRequest(BaseModel):
    text: str
    email: Optional[str] = None
    session_id: Optional[str] = None


class TokenToggleRequest(BaseModel):
    id: int
    active: bool


class TokenCreateRequest(BaseModel):
    label: Optional[str] = None
    kind: str = "secondary"


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _sse(kind: str, **data) -> str:
    return f"data: {json.dumps({'type': kind, **data})}\n\n"


async def _pump_tokens(llm_stream, messages, sampling):
    """Stream the model, yielding ('token', text) for content and ('status',
    'reasoning') heartbeats while waiting (the GPU may be cold). Ends with
    ('truncated', bool) so the caller can tell whether the answer was cut off.
    """
    agen = llm_stream(messages, sampling).__aiter__()
    nxt = asyncio.ensure_future(agen.__anext__())
    truncated = False
    while True:
        try:
            tok = await asyncio.wait_for(asyncio.shield(nxt), timeout=8)
        except asyncio.TimeoutError:
            yield ("status", "reasoning")
            continue
        except StopAsyncIteration:
            break
        if tok == STREAM_TRUNCATED:
            truncated = True
        else:
            yield ("token", tok)
        nxt = asyncio.ensure_future(agen.__anext__())
    yield ("truncated", truncated)


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
    api = FastAPI(title="Naija-Petro", version="0.3.0")

    # The interface is open to test. Anonymous visitors get settings.free_daily_limit
    # queries per calendar day; a valid, active access token lifts that limit. The
    # /admin panel (token management) is guarded by settings.admin_token.
    def _admin_ok(request: Request) -> bool:
        tok = request.headers.get("x-admin-token") or request.query_params.get("admin", "")
        return bool(settings.admin_token) and tok == settings.admin_token

    def _req_token(request: Request) -> str:
        return (request.headers.get("x-access-token") or request.query_params.get("token", "")).strip()

    @api.get("/healthz")
    async def healthz():
        from app import __version__
        return {"status": "ok", "version": __version__, "open": True,
                "daily_limit": settings.free_daily_limit, "admin": bool(settings.admin_token)}

    @api.get("/token/check")
    async def token_check(token: str = ""):
        try:
            return {"valid": await db.token_active(token.strip())}
        except Exception:
            return {"valid": False}

    # ----- Admin panel API (guarded by X-Admin-Token) -----
    @api.get("/admin/api/auth")
    async def admin_auth(request: Request):
        return {"ok": _admin_ok(request)}

    @api.get("/admin/api/tokens")
    async def admin_tokens(request: Request):
        if not _admin_ok(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        try:
            return {"tokens": await db.list_tokens()}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    @api.post("/admin/api/tokens/toggle")
    async def admin_toggle(req: TokenToggleRequest, request: Request):
        if not _admin_ok(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        try:
            await db.set_token_active(int(req.id), bool(req.active))
            return {"ok": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    @api.post("/admin/api/tokens/create")
    async def admin_create(req: TokenCreateRequest, request: Request):
        if not _admin_ok(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        kind = "primary" if (req.kind or "").lower() == "primary" else "secondary"
        cap = 3 if kind == "primary" else 7
        try:
            counts = await db.count_tokens_by_kind()
            if counts.get(kind, 0) >= cap:
                return JSONResponse({"error": f"Limit reached: max {cap} {kind} tokens."}, status_code=400)
            import secrets as _secrets
            token = f"np-{'pri' if kind == 'primary' else 'sec'}-{_secrets.token_urlsafe(12)}"
            label = (req.label or "").strip()[:60] or f"{kind.title()} token"
            await db.create_token(token, label, kind)
            return {"ok": True, "token": token, "label": label, "kind": kind}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    @api.get("/admin/api/stats")
    async def admin_stats(request: Request):
        if not _admin_ok(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        try:
            return await db.usage_overview(14)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)

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

    @api.get("/tools")
    async def tools_list():
        return {"calculators": [
            {"name": n, "label": spec["label"], "args": spec["args"],
             "description": (CALCULATORS[n][1] if n in CALCULATORS else "")}
            for n, spec in TOOL_SPECS.items()
        ]}

    @api.post("/tools/run")
    async def tools_run(req: ToolRunRequest):
        return run_tool(req.name, req.args)

    @api.post("/feature")
    async def feature(req: FeatureRequest):
        text = (req.text or "").strip()
        if len(text) < 3:
            return JSONResponse({"error": "Please describe the feature."}, status_code=400)
        try:
            await db.add_feature(text[:1000], (req.email or None), req.session_id)
        except Exception:
            return JSONResponse({"error": "Could not save right now."}, status_code=503)
        return {"ok": True}

    @api.get("/features")
    async def features():
        try:
            return {"features": await db.list_features(20)}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    @api.get("/history")
    async def history(user_id: str = ""):
        if not user_id:
            return {"sessions": []}
        try:
            return {"sessions": await db.list_sessions(user_id, 25)}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    @api.get("/history/{session_id}")
    async def history_session(session_id: str):
        try:
            return {"turns": await db.load_session(session_id, 100)}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    @api.post("/upload")
    async def upload(file: UploadFile = File(...), session_id: str = Form(""), user_id: str = Form("")):
        from app.rag import ingest
        data = await file.read()
        if len(data) > 8_000_000:
            return JSONResponse({"error": "File too large (max 8 MB)."}, status_code=400)
        try:
            text = await asyncio.to_thread(ingest.extract_upload, file.filename or "upload", data)
            if len((text or "").strip()) < 50:
                return JSONResponse({"error": "Could not extract readable text from this file."}, status_code=400)
            res = await ingest.ingest_text(
                text, (file.filename or "Uploaded document"), deps.embed, source_label="upload",
                metadata={"filename": file.filename, "session_id": session_id, "user_id": user_id},
            )
            return {"ok": True, "filename": file.filename,
                    "chunks": res.get("chunk_count", 0), "inserted": res.get("inserted", False)}
        except Exception as e:
            return JSONResponse({"error": f"Upload failed: {e}"}, status_code=500)

    @api.post("/chat")
    async def chat(req: ChatRequest, request: Request):
        # --- gates (checked before we start streaming) ---
        msg = (req.message or "").strip()
        if not msg:
            return JSONResponse({"error": "Empty message"}, status_code=400)
        if len(msg) > settings.max_query_chars:
            return JSONResponse({"error": f"Message too long (max {settings.max_query_chars})"}, status_code=400)

        ip_hash = _hash_ip(_client_ip(request))
        sess = req.session_id or ip_hash
        country = request.headers.get("cf-ipcountry") or request.headers.get("x-vercel-ip-country")

        # Burst limit (anti-abuse) applies to everyone.
        if not (_rate_ok(f"m:{ip_hash}", settings.rate_limit_max, settings.rate_limit_window_s)
                and _rate_ok(f"h:{ip_hash}", settings.rate_limit_max_hour, 3600)):
            return JSONResponse({"error": "You are sending requests too quickly. Please wait a moment."}, status_code=429)

        # Daily free limit: anonymous visitors get settings.free_daily_limit queries
        # per calendar day. A valid, active access token lifts the limit.
        token = _req_token(request)
        try:
            has_token = await db.token_active(token) if token else False
        except Exception:
            has_token = False
        if not has_token and not req.continuation and settings.free_daily_limit > 0:
            try:
                used = await db.daily_ip_count(ip_hash)
            except Exception:
                used = 0
            if used >= settings.free_daily_limit:
                return JSONResponse(
                    {"error": f"You have used your {settings.free_daily_limit} free questions for today. "
                              "Come back tomorrow, or enter an access token to keep going.",
                     "limit": "daily", "used": used, "max": settings.free_daily_limit},
                    status_code=429,
                )

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
            truncated = False

            # Continuation: finish a previously cut-off answer. No new retrieval,
            # sources, tools, or follow-ups; just keep generating from the history.
            if req.continuation:
                hist = [
                    {"role": h.get("role"), "content": h.get("content", "")}
                    for h in (req.history or []) if h.get("role") in ("user", "assistant")
                ]
                cont_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + hist + \
                    [{"role": "user", "content": CONTINUE_INSTRUCTION}]
                cont_sampling = {**sampling, "reasoning": False}
                try:
                    parts: list[str] = []
                    async for kind, val in _pump_tokens(deps.llm_stream, cont_messages, cont_sampling):
                        if kind == "token":
                            parts.append(val)
                            yield _sse("token", t=val)
                        elif kind == "status":
                            yield _sse("status", stage=val)
                        elif kind == "truncated":
                            truncated = val
                    answer = "".join(parts)
                    yield _sse("done", chars=len(answer), truncated=truncated)
                except Exception as e:
                    yield _sse("error", message=str(e))
                return

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
                    if (req.units or "field").lower() == "si":
                        messages[-1]["content"] = (
                            "Present all numeric results in SI units (metres, kPa or MPa, m3, sm3, "
                            "kg/m3); convert from field units where needed and show the converted value.\n\n"
                            + messages[-1]["content"]
                        )

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

                    # Tool pre-pass: for computational questions, the model picks a
                    # calculator (JSON), we compute exact figures and inject them so the
                    # streamed answer uses verified numbers, not estimates.
                    if needs_tools(msg) and deps.llm_complete:
                        try:
                            sel = await deps.llm_complete(
                                [{"role": "system", "content": TOOL_SELECT_SYS},
                                 {"role": "user", "content": f"Calculators:\n{TOOL_MENU}\n\nQuestion: {msg}\n\nJSON:"}],
                                {"max_tokens": 160, "temperature": 0.0, "reasoning": False},
                            )
                            obj = _parse_json_obj(sel)
                            if obj and obj.get("tool") in CALCULATORS:
                                res = run_tool(obj["tool"], obj.get("args", {}))
                                if "error" not in res:
                                    yield _sse("tool", name=obj["tool"], args=obj.get("args", {}), result=res)
                                    note = (
                                        "A verified engineering calculator has already computed the exact result "
                                        "for this question:\n"
                                        f"{obj['tool']}({json.dumps(obj.get('args', {}))}) = {json.dumps(res)}\n"
                                        "Report these exact figures as the answer. You may show the governing "
                                        "formula and explain the inputs, but state the verified numeric result "
                                        "above verbatim and do NOT redo the arithmetic or produce a different "
                                        "number.\n\n"
                                    )
                                    messages[-1]["content"] = note + messages[-1]["content"]
                        except Exception:
                            pass

                    # Stream tokens, heartbeating until the first token (LLM may be cold).
                    parts: list[str] = []
                    async for kind, val in _pump_tokens(deps.llm_stream, messages, sampling):
                        if kind == "token":
                            parts.append(val)
                            yield _sse("token", t=val)
                        elif kind == "status":
                            yield _sse("status", stage=val)
                        elif kind == "truncated":
                            truncated = val

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

                    yield _sse("done", chars=len(answer), truncated=truncated)
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
                # Saved history (best-effort)
                try:
                    if req.user_id and answer:
                        await db.save_turns(req.user_id, sess, [("user", msg), ("assistant", answer)])
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

        @api.get("/admin")
        async def admin_page():
            return FileResponse(FRONTEND_DIR / "admin.html", headers={"Cache-Control": "no-store"})

        api.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    return api
