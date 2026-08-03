"""FastAPI front door for the Naija-Petro RAG assistant.

`create_app(deps)` builds the app with GPU-backed callables injected, so this
module never imports Modal, keeping it cycle-free and unit-testable with a stub.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
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

log = logging.getLogger("naija_petro")

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


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _strip_think(text: str) -> str:
    """Drop a reasoning block that came back inline rather than in its own field."""
    text = _THINK_RE.sub("", text or "")
    return text.replace("<think>", "").replace("</think>", "").strip()


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
    warm_llm: Optional[Callable[[], None]] = None      # pre-boot the GPU, returns at once


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

# Caller-supplied opaque ids (session_id, user_id) are echoed into PostgREST
# filters, Langfuse attributes and analytics rows, so they are bounded here.
_ID_MAX_CHARS = 128
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")

# `history` is caller-supplied and goes straight into the prompt, so it is
# bounded in count, size and role before it can reach the model. The UI sends at
# most 8 turns of at most ~4k characters (max_new_tokens is 1024), so these caps
# never clip real traffic; they only stop a hand-written request from filling the
# context window or smuggling in a system turn.
_HISTORY_MAX_TURNS = 12
_HISTORY_MAX_CHARS_PER_TURN = 6000
# Derived from the model window rather than fixed: at roughly 3.2 characters per
# token, a flat 24000 was ~7500 tokens of history alone, more than the whole 8192
# window before a single retrieved source or generated token. Quarter of the
# window leaves room for the sources, which are the point of the app.
_HISTORY_MAX_CHARS_TOTAL = max(4000, int(settings.max_model_len * 3.2 * 0.25))
_HISTORY_ROLES = ("user", "assistant")

# Free continuations per IP per day, as a multiple of the daily question limit:
# finishing a cut-off answer can legitimately take a few rounds, but not an
# unbounded number. Token holders are exempt.
_CONTINUATIONS_PER_DAY = 3


# --- /upload limits ---
_UPLOAD_MAX_BYTES = 8_000_000
_UPLOAD_CHUNK_BYTES = 1 << 20
_UPLOAD_MAX_PER_HOUR = 10
# extract_upload only knows how to read a PDF or decode text, and anything else
# decodes to mojibake that still passes the 50-character check and gets embedded
# into the shared knowledge base, so the accepted set is spelled out.
_UPLOAD_SUFFIXES = (".pdf", ".txt", ".text", ".md", ".markdown", ".csv", ".tsv", ".json", ".log")
_UPLOAD_TYPES = ("application/pdf", "application/json", "application/x-ndjson")


def _clean_filename(name) -> str:
    """A display-safe, bounded filename.

    It is echoed back to the caller, stored as the document title and used to
    build `upload://<title>`, and multipart headers are not length-limited.
    """
    if not isinstance(name, str):
        return ""
    name = _CTRL_RE.sub("", name).replace("\\", "/").rsplit("/", 1)[-1].strip()
    return name[:200]


def _upload_type_ok(filename: str, content_type) -> bool:
    ctype = (content_type or "").split(";")[0].strip().lower()
    if filename.lower().endswith(_UPLOAD_SUFFIXES):
        return True
    return ctype.startswith("text/") or ctype in _UPLOAD_TYPES


def _json_safe(value, _depth: int = 0):
    """Replace non-finite floats with null so a payload can always be encoded.

    Starlette renders JSONResponse with allow_nan=False, so a single inf or nan
    anywhere in a calculator result turns a 200 into an unhandled 500; in an SSE
    frame json.dumps emits `Infinity`, which is not JSON and which the browser
    drops on the floor. Overflowing but perfectly legal inputs reach both paths
    (hydrostatic_pressure with 1e308 ppg, or a "nan" string argument).
    """
    if _depth > 20:            # caller-supplied args can nest; do not recurse forever
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, _depth + 1) for v in value]
    return value


def _clean_id(value, limit: int = _ID_MAX_CHARS) -> str:
    """Bound an opaque, client-generated id (session_id / user_id).

    Nothing server-side ever validated these, so an oversized or control-character
    id silently broke every downstream write (PostgREST filter, analytics row,
    saved history) inside a best-effort swallow.
    """
    if not isinstance(value, str):
        return ""
    return _CTRL_RE.sub("", value).strip()[:limit]


def _clean_history(raw) -> list[dict]:
    """Bound and role-filter caller-supplied conversation history.

    Only user/assistant turns survive: build_messages splices history straight
    into the message list, so an unfiltered request could inject its own system
    turn and replace SYSTEM_PROMPT. Content must be a string (a dict or int there
    blows up the chat template as a 500-equivalent error event) and the whole
    block is capped so a large history cannot push the prompt past the context
    window. Over-long turns keep their tail, which is what a continuation needs.
    """
    if not isinstance(raw, list):
        return []
    turns: list[dict] = []
    for item in raw[-_HISTORY_MAX_TURNS * 4:]:     # bound the scan itself
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in _HISTORY_ROLES or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if len(content) > _HISTORY_MAX_CHARS_PER_TURN:
            content = content[-_HISTORY_MAX_CHARS_PER_TURN:]
        turns.append({"role": role, "content": content})

    turns = turns[-_HISTORY_MAX_TURNS:]
    total = 0
    kept: list[dict] = []
    for turn in reversed(turns):                   # drop the oldest turns first
        total += len(turn["content"])
        if kept and total > _HISTORY_MAX_CHARS_TOTAL:
            break
        kept.append(turn)
    kept.reverse()
    return kept


def _sse(kind: str, **data) -> str:
    return f"data: {json.dumps(_json_safe({'type': kind, **data}))}\n\n"


def _store_error(where: str, exc: Exception, status: int = 503) -> JSONResponse:
    """Log the real failure, hand the caller a generic one.

    str(exc) on an httpx/PostgREST error carries the Supabase host, the schema
    name and the database's own message. Several of these endpoints are
    unauthenticated, so returning it was an infrastructure disclosure to anyone
    who could make a store call fail, and nothing was written to the log either.
    """
    log.warning("%s failed: %s: %s", where, type(exc).__name__, exc)
    return JSONResponse({"error": "The knowledge store is unavailable right now."},
                        status_code=status)


async def _pump_tokens(llm_stream, messages, sampling):
    """Stream the model, yielding ('token', text) for content and ('status',
    'reasoning') heartbeats while waiting (the GPU may be cold). Ends with
    ('truncated', bool) so the caller can tell whether the answer was cut off.
    """
    agen = llm_stream(messages, sampling).__aiter__()
    nxt = asyncio.ensure_future(agen.__anext__())
    truncated = False
    try:
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
    finally:
        # A client that hangs up mid-answer closes this generator. Without a
        # teardown the in-flight read was just abandoned: the task pins the
        # backend generator, so the stream stayed open until that read resolved
        # (on a cold GPU, the whole time to first token) and was then only
        # released by the garbage collector, and a read that failed instead
        # surfaced as a bare "Task exception was never retrieved" with no
        # request context attached.
        if nxt.done():
            # Suspended on a yield: the read already resolved, so the generator
            # is idle and can be closed properly.
            try:
                await agen.aclose()
            except Exception as e:
                log.warning("llm stream close failed: %s: %s", type(e).__name__, e)
        else:
            # Suspended on a heartbeat with a read still in flight; cancelling
            # the read is what unwinds the generator and drops the connection.
            nxt.cancel()


# Strong refs to in-flight warm-ups: a bare task can be garbage collected mid-run.
_warm_tasks: set = set()

# Same, for analytics writes detached after a client disconnect.
_record_tasks: set = set()


async def _run_warm(warm_llm) -> None:
    """Spawn the GPU pre-boot off the event loop, swallowing everything.

    The spawn is a blocking control-plane call, so it goes through a thread; a
    warm-up is a pure optimisation and must never slow down or fail a request.
    """
    try:
        await asyncio.to_thread(warm_llm)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Swallowed on purpose (a warm-up must never affect a request), but not
        # silently: a broken pre-boot only shows up as every answer being slow.
        log.warning("gpu warm-up failed: %s: %s", type(e).__name__, e)


# --- simple in-memory rate limiter (per web container) ---
# Entries are expiry stamps (hit time + window), not hit times, so the sweep can
# retire a key without knowing which window it was counted under.
_hits: dict = defaultdict(deque)
_last_prune = 0.0
_PRUNE_EVERY_S = 300
# Backstop for a flood of one-shot IPs between sweeps. ~100k deques is a few MB.
_MAX_HIT_KEYS = 100_000


def _prune_hits(now: float) -> None:
    """Drop keys whose window has fully elapsed.

    `_hits` only ever grew: one deque per hashed IP the container had ever seen,
    kept for the life of the container. Memory tracked total unique visitors
    rather than active ones, so a long-lived web container leaked steadily.
    """
    global _last_prune
    _last_prune = now
    for key in list(_hits.keys()):
        dq = _hits.get(key)
        if dq is None:
            continue
        while dq and dq[0] <= now:
            dq.popleft()
        if not dq:
            _hits.pop(key, None)


def _rate_ok(key: str, max_n: int, window_s: int) -> bool:
    now = time.time()
    # Sweep on a timer, or sooner under a key flood, but never more than once
    # every few seconds: the sweep is O(keys) and runs on the request path.
    if (now - _last_prune > _PRUNE_EVERY_S
            or (len(_hits) > _MAX_HIT_KEYS and now - _last_prune > 5)):
        over = len(_hits) > _MAX_HIT_KEYS
        _prune_hits(now)
        if over and len(_hits) > _MAX_HIT_KEYS:
            log.warning("rate-limiter holding %d live keys after a sweep", len(_hits))
    dq = _hits[key]
    while dq and dq[0] <= now:
        dq.popleft()
    if len(dq) >= max_n:
        return False
    dq.append(now + window_s)
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
        # compare_digest, not ==: a plain comparison returns as soon as it finds a
        # differing byte, which leaks the token prefix by prefix to anyone willing
        # to time the responses. The endpoint is public and unthrottled. Compare
        # bytes, because compare_digest raises on non-ASCII str input.
        return bool(settings.admin_token) and hmac.compare_digest(
            tok.encode("utf-8", "ignore"), settings.admin_token.encode("utf-8"))

    def _req_token(request: Request) -> str:
        # Bounded: it is a query parameter that becomes a PostgREST filter.
        return (request.headers.get("x-access-token")
                or request.query_params.get("token", "")).strip()[:_ID_MAX_CHARS]

    def _fire_warm() -> bool:
        """Trigger a GPU pre-boot, never blocking and never raising. Returns
        whether a warm-up was actually fired.
        """
        if not settings.enable_warm or not deps.warm_llm:
            return False
        try:
            task = asyncio.ensure_future(_run_warm(deps.warm_llm))
            _warm_tasks.add(task)
            task.add_done_callback(_warm_tasks.discard)
        except Exception as e:
            log.warning("could not schedule gpu warm-up: %s: %s", type(e).__name__, e)
            return False
        return True

    @api.api_route("/warm", methods=["GET", "POST"])
    async def warm(request: Request):
        """Boot the GPU ahead of a question so the cold start is not in the wait.

        Fire-and-forget: it never touches the GPU response and never raises.

        This is the only path that can start a GPU without a question being
        asked, so it is capped twice: once per IP, and once globally per hour so
        rotating IPs cannot hold an L4 alive indefinitely. Every trigger keeps
        the GPU billing for at least LLM_SCALEDOWN_WINDOW seconds.
        """
        try:
            if not settings.enable_warm:
                return {"warming": False}
            # One boot per visitor per window: repeat calls add nothing while the
            # container is already warm, they only extend the idle bill.
            if not _rate_ok(f"w:{_hash_ip(_client_ip(request))}", 1, settings.warm_per_ip_window_s):
                return {"warming": False}
            if not _rate_ok("w:global", settings.warm_max_per_hour, 3600):
                return {"warming": False}
            return {"warming": _fire_warm()}
        except Exception as e:
            log.warning("/warm failed: %s: %s", type(e).__name__, e)
            return {"warming": False}

    @api.get("/healthz")
    async def healthz():
        from app import __version__
        return {"status": "ok", "version": __version__, "open": True,
                "daily_limit": settings.free_daily_limit, "admin": bool(settings.admin_token)}

    @api.get("/token/check")
    async def token_check(token: str = ""):
        try:
            return {"valid": await db.token_active(token.strip()[:_ID_MAX_CHARS])}
        except Exception as e:
            # Fails closed on purpose, but a store outage makes every valid token
            # look invalid, so it must not be invisible.
            log.warning("token check failed: %s: %s", type(e).__name__, e)
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
            return _store_error("admin token list", e)

    @api.post("/admin/api/tokens/toggle")
    async def admin_toggle(req: TokenToggleRequest, request: Request):
        if not _admin_ok(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        try:
            await db.set_token_active(int(req.id), bool(req.active))
            return {"ok": True}
        except Exception as e:
            return _store_error("admin token toggle", e)

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
            return _store_error("admin token create", e)

    @api.get("/admin/api/stats")
    async def admin_stats(request: Request):
        if not _admin_ok(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        try:
            return await db.usage_overview(14)
        except Exception as e:
            return _store_error("admin stats", e)

    @api.get("/kb/stats")
    async def kb_stats():
        try:
            return await db.kb_stats()
        except Exception as e:
            return _store_error("kb stats", e)

    @api.post("/feedback")
    async def feedback(fb: FeedbackRequest):
        # Public, unauthenticated write into the preference/training table, so the
        # payload is bounded and the rating is forced to the three values the UI
        # can actually send (it also becomes a Langfuse numeric score).
        payload = fb.model_dump()
        payload["rating"] = max(-1, min(1, int(fb.rating or 0)))
        payload["session_id"] = _clean_id(fb.session_id)
        payload["user_id"] = _clean_id(fb.user_id)
        payload["trace_id"] = _clean_id(fb.trace_id)
        payload["query"] = (fb.query or "")[:settings.max_query_chars]
        payload["comment"] = (fb.comment or "")[:2000]
        payload["answer"] = (fb.answer or "")[:40000]
        payload["sources"] = (fb.sources or [])[:50]
        try:
            await db.log_feedback(payload)
        except Exception as e:
            # Best-effort, but this is the preference data the fine-tune feeds on:
            # a silent swallow means the pipeline can be dead for months.
            log.warning("feedback write failed: %s: %s", type(e).__name__, e)
        try:
            client = obs.get_client()
            if client and payload["trace_id"] and payload["rating"]:
                client.create_score(name="user_feedback", value=payload["rating"],
                                    trace_id=payload["trace_id"], data_type="NUMERIC")
                await asyncio.to_thread(client.flush)
        except Exception as e:
            log.warning("feedback score failed: %s: %s", type(e).__name__, e)
        return {"ok": True}

    @api.post("/subscribe")
    async def subscribe(req: SubscribeRequest):
        email = (req.email or "").strip().lower()
        if not _EMAIL_RE.match(email) or len(email) > 254:
            return JSONResponse({"error": "Please enter a valid email."}, status_code=400)
        try:
            await db.subscribe(email, bool(req.wants_updates), source="app")
        except Exception as e:
            log.warning("subscribe failed: %s: %s", type(e).__name__, e)
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
        # run_tool already turns a bad argument set into {"error": ...}, but it
        # cannot stop a well-formed argument from overflowing to inf (1e308 ppg)
        # or arriving as "nan". Starlette renders JSON with allow_nan=False, so
        # those results used to come back as an unhandled 500.
        return _json_safe(run_tool(req.name, req.args))

    @api.post("/feature")
    async def feature(req: FeatureRequest):
        text = (req.text or "").strip()
        if len(text) < 3:
            return JSONResponse({"error": "Please describe the feature."}, status_code=400)
        try:
            await db.add_feature(text[:1000], ((req.email or "").strip()[:254] or None),
                                 _clean_id(req.session_id))
        except Exception as e:
            log.warning("feature request write failed: %s: %s", type(e).__name__, e)
            return JSONResponse({"error": "Could not save right now."}, status_code=503)
        return {"ok": True}

    @api.get("/features")
    async def features():
        try:
            return {"features": await db.list_features(20)}
        except Exception as e:
            return _store_error("feature list", e)

    @api.get("/history")
    async def history(user_id: str = ""):
        # Bounded before it becomes a PostgREST filter in a request URL.
        user_id = _clean_id(user_id)
        if not user_id:
            return {"sessions": []}
        try:
            return {"sessions": await db.list_sessions(user_id, 25)}
        except Exception as e:
            return _store_error("session list", e)

    @api.get("/history/{session_id}")
    async def history_session(session_id: str):
        session_id = _clean_id(session_id)
        if not session_id:
            return {"turns": []}
        try:
            return {"turns": await db.load_session(session_id, 100)}
        except Exception as e:
            return _store_error("session load", e)

    @api.post("/upload")
    async def upload(request: Request, file: UploadFile = File(...),
                     session_id: str = Form(""), user_id: str = Form("")):
        from app.rag import ingest

        # Unauthenticated, and every accepted file spends embedding GPU and lands
        # in the knowledge base every visitor then reads, so it gets its own cap.
        ip_hash = _hash_ip(_client_ip(request))
        if not _rate_ok(f"u:{ip_hash}", _UPLOAD_MAX_PER_HOUR, 3600):
            return JSONResponse({"error": "Too many uploads. Please try again later."}, status_code=429)

        filename = _clean_filename(file.filename)
        if not _upload_type_ok(filename, file.content_type):
            return JSONResponse(
                {"error": "Unsupported file type. Upload a PDF, text, markdown or CSV file."},
                status_code=400)

        # Read incrementally: `await file.read()` pulled the entire body into the
        # web container before the size was ever checked, so the 8 MB limit did
        # nothing to stop a 2 GB post from taking the container down with it.
        chunks: list[bytes] = []
        size = 0
        while True:
            block = await file.read(_UPLOAD_CHUNK_BYTES)
            if not block:
                break
            size += len(block)
            if size > _UPLOAD_MAX_BYTES:
                return JSONResponse({"error": "File too large (max 8 MB)."}, status_code=400)
            chunks.append(block)
        data = b"".join(chunks)
        if not data:
            return JSONResponse({"error": "The file is empty."}, status_code=400)

        try:
            text = await asyncio.to_thread(ingest.extract_upload, filename or "upload", data)
        except Exception as e:
            # Extraction runs on caller-controlled bytes, so a parser blowing up is
            # expected traffic, not an incident: report it as a bad file and keep
            # the parser's message (which can name internal paths) in the log.
            log.warning("upload extraction failed for %r: %s: %s", filename, type(e).__name__, e)
            return JSONResponse({"error": "Could not read this file."}, status_code=400)
        if len((text or "").strip()) < 50:
            return JSONResponse({"error": "Could not extract readable text from this file."}, status_code=400)

        try:
            res = await ingest.ingest_text(
                text, (filename or "Uploaded document"), deps.embed, source_label="upload",
                metadata={"filename": filename, "session_id": _clean_id(session_id),
                          "user_id": _clean_id(user_id)},
            )
        except Exception as e:
            log.warning("upload ingest failed for %r: %s: %s", filename, type(e).__name__, e)
            return JSONResponse({"error": "Could not index this file right now."}, status_code=503)
        return {"ok": True, "filename": filename,
                "chunks": res.get("chunk_count", 0), "inserted": res.get("inserted", False)}

    @api.post("/chat")
    async def chat(req: ChatRequest, request: Request):
        # --- gates (checked before we start streaming) ---
        msg = (req.message or "").strip()
        if not msg:
            return JSONResponse({"error": "Empty message"}, status_code=400)
        if len(msg) > settings.max_query_chars:
            return JSONResponse({"error": f"Message too long (max {settings.max_query_chars})"}, status_code=400)

        ip_hash = _hash_ip(_client_ip(request))
        sess = _clean_id(req.session_id) or ip_hash
        user_id = _clean_id(req.user_id)
        history = _clean_history(req.history)
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
        except Exception as e:
            # Fails closed: a store outage rate-limits paying token holders, which
            # is the safe direction but must be visible in the logs.
            log.warning("token lookup failed: %s: %s", type(e).__name__, e)
            has_token = False

        # A continuation deliberately skips the daily check and returns before the
        # analytics write, so it neither consumes nor counts a credit. That also
        # makes it the one unmetered generation path: `continuation: true` plus a
        # hand-written history is a free answer to any question, repeatable for
        # ever, and only the burst limit stood in the way (200 GPU generations an
        # hour per IP). Cap it per IP per day so the escape hatch stays one.
        if req.continuation and not has_token and settings.free_daily_limit > 0:
            if not _rate_ok(f"c:{ip_hash}", _CONTINUATIONS_PER_DAY * settings.free_daily_limit, 86400):
                return JSONResponse(
                    {"error": "You have used your free continuations for today. "
                              "Come back tomorrow, or enter an access token to keep going.",
                     "limit": "daily"},
                    status_code=429,
                )

        if not has_token and not req.continuation and settings.free_daily_limit > 0:
            try:
                used = await db.daily_ip_count(ip_hash)
            except Exception as e:
                # Fails open on purpose (a store outage must not close the app),
                # but that silently disables the daily limit, so it gets a log.
                log.warning("daily count failed, letting the request through: %s: %s",
                            type(e).__name__, e)
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
            # Hoisted so the analytics `finally` can still see a partial answer
            # when the model dies mid-stream or the client hangs up: previously
            # the join never ran on those paths and the turn was logged as zero
            # characters even though the tokens had been generated and paid for.
            parts: list[str] = []
            disconnected = False

            # Continuation: finish a previously cut-off answer. No new retrieval,
            # sources, tools, or follow-ups; just keep generating from the history.
            if req.continuation:
                cont_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + \
                    [{"role": "user", "content": CONTINUE_INSTRUCTION}]
                cont_sampling = {**sampling, "reasoning": False}
                try:
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
                    log.warning("continuation stream failed: %s: %s", type(e).__name__, e)
                    yield _sse("error", message=str(e))
                return

            yield _sse("status", stage="starting")
            try:
                with obs.trace("chat", session_id=sess, user_id=user_id, input=msg) as tr:
                    rerank_fn = deps.rerank if (settings.enable_rerank and deps.rerank) else None

                    # Boot the GPU now, not after retrieval: the weights load while
                    # sources are searched instead of strictly afterwards.
                    _fire_warm()

                    # Retrieve, heartbeating while we wait (Encoders may be cold).
                    rtask = asyncio.ensure_future(retrieve(msg, deps.embed, rerank_fn))
                    while True:
                        try:
                            result = await asyncio.wait_for(asyncio.shield(rtask), timeout=8)
                            break
                        except asyncio.TimeoutError:
                            yield _sse("status", stage="searching")
                    messages, sources = build_messages(msg, result.chunks, history, reasoning=req.reasoning)
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
                        except Exception as e:
                            # The self-updating knowledge base is exactly the kind
                            # of background feature that can be dead for months
                            # behind a bare `pass`.
                            log.warning("enrichment spawn failed: %s: %s", type(e).__name__, e)

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
                                # Sanitised here as well as at the SSE frame. json.loads
                                # accepts `Infinity` and `NaN`, so the model can hand
                                # back a non-finite argument, and a finite one can still
                                # overflow inside the calculator. Either way the literal
                                # `Infinity` must not be written into the prompt as a
                                # "verified" figure for the model to quote.
                                args = _json_safe(obj.get("args") or {})
                                if not isinstance(args, dict):
                                    args = {}
                                res = _json_safe(run_tool(obj["tool"], args))
                                if "error" not in res:
                                    yield _sse("tool", name=obj["tool"], args=args, result=res)
                                    # Plot points go to the UI but never into the prompt:
                                    # a "series" can be dozens of values of pure context bloat.
                                    lean = {k: v for k, v in res.items() if k != "series"}
                                    note = (
                                        "A verified engineering calculator has already computed the exact result "
                                        "for this question:\n"
                                        f"{obj['tool']}({json.dumps(args)}) = {json.dumps(lean)}\n"
                                        "Report these exact figures as the answer. You may show the governing "
                                        "formula and explain the inputs, but state the verified numeric result "
                                        "above verbatim and do NOT redo the arithmetic or produce a different "
                                        "number.\n\n"
                                    )
                                    messages[-1]["content"] = note + messages[-1]["content"]
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            # Best-effort: an unusable pre-pass must not cost the
                            # user an answer. But a bare `pass` here is how a whole
                            # feature stays dead without anyone noticing.
                            log.warning("tool pre-pass failed: %s: %s", type(e).__name__, e)

                    # Stream tokens, heartbeating until the first token (LLM may be cold).
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

                    # Close the answer first: follow-ups need another generation and
                    # the user should not watch a finished answer wait on it.
                    yield _sse("done", chars=len(answer), truncated=truncated)

                    # Suggested follow-up questions (best-effort, non-streaming).
                    if settings.enable_followups and deps.llm_complete and answer:
                        try:
                            fu_msgs = [
                                {"role": "system", "content": FOLLOWUP_SYS},
                                {"role": "user", "content": f"Question: {msg}\n\nAnswer: {answer[:1500]}\n\nThree follow-up questions:"},
                            ]
                            # 220 tokens, because the model ignores the 12 word
                            # request and writes 80 to 200 character questions; at 140
                            # the third one was cut off mid-sentence.
                            fu = await deps.llm_complete(fu_msgs, {"max_tokens": 220, "temperature": 0.5, "reasoning": False})
                            qs = [_clean_followup(l) for l in _strip_think(fu).splitlines() if l.strip()]
                            # Keep questions, nothing else. The model sometimes answers
                            # instead of asking, and a length filter alone let markdown
                            # table rows through as "suggestions". Trailing question
                            # mark is the reliable signal; emitting nothing beats
                            # emitting noise.
                            qs = [q for q in qs if q.endswith("?") and len(q) > 8][:3]
                            qs = [q if len(q) <= 160 else q[:157].rstrip() + "..." for q in qs]
                            if qs:
                                yield _sse("followups", items=qs)
                            else:
                                log.warning("follow-ups produced nothing usable from %r", (fu or "")[:200])
                        except asyncio.CancelledError:
                            log.warning("follow-up generation cancelled")
                            raise
                        except Exception as e:
                            # Best-effort, but not silent: swallowing this is how the
                            # follow-up feature stayed broken without anyone noticing.
                            log.warning("follow-up generation failed: %s: %s", type(e).__name__, e)
            except asyncio.CancelledError:
                # The client hung up mid-stream. Nothing can be yielded any more,
                # and every await from here on re-raises, so the analytics below
                # are detached instead of awaited: a disconnect used to burn a
                # full generation that was never counted against the daily limit.
                disconnected = True
                log.warning("chat stream cancelled after %d chars (client disconnect)",
                            len("".join(parts)))
                raise
            except Exception as e:  # surface errors to the client cleanly
                log.warning("chat stream failed: %s: %s", type(e).__name__, e)
                yield _sse("error", message=str(e))
            finally:
                # An error or a disconnect skips the join above, so recover what
                # was actually generated rather than logging the turn as empty.
                if not answer and parts:
                    answer = "".join(parts)

                async def _record() -> None:
                    # Analytics (best-effort, never raises into the stream). Note
                    # db.daily_ip_count reads these rows, so a silent failure here
                    # also silently disables the daily limit.
                    try:
                        await db.log_usage({
                            "session_id": sess, "user_id": user_id or None, "ip_hash": ip_hash,
                            "country": country, "query": msg[:1000], "answer_chars": len(answer),
                            "n_sources": (len(result.chunks) if result else 0),
                            "coverage": (round(result.coverage, 4) if result else 0.0),
                            "enriched": (bool(result.enriched) if result else False),
                            "kb_added": (result.enrichment.get("new_chunks", 0) if result else 0),
                            "reasoning": req.reasoning, "latency_ms": int((time.time() - t0) * 1000),
                        })
                    except Exception as e:
                        log.warning("usage write failed: %s: %s", type(e).__name__, e)
                    # Saved history (best-effort)
                    try:
                        if user_id and answer:
                            await db.save_turns(user_id, sess, [("user", msg), ("assistant", answer)])
                    except Exception as e:
                        log.warning("history write failed: %s: %s", type(e).__name__, e)

                if disconnected:
                    task = asyncio.ensure_future(_record())
                    _record_tasks.add(task)          # a bare task can be collected mid-run
                    task.add_done_callback(_record_tasks.discard)
                else:
                    await _record()

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
