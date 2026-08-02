"""Local development entrypoint: the whole app on a laptop, no Modal, no cloud GPU.

modal_app.py is the production wiring: it injects Modal-backed callables into the
`Deps` dataclass and serves app/api/server.py as an ASGI app. This module fills the
same `Deps` with local implementations, so the API, the RAG pipeline, and the
frontend can be exercised without deploying anything:

  * llm     -> any OpenAI-compatible server on localhost (Ollama by default)
  * embed   -> sentence-transformers on CPU, loaded lazily on first use
  * rerank  -> optional cross-encoder, only when ENABLE_RERANK is on
  * enrich  -> a background asyncio task instead of Modal's `enrich.spawn`

    python local_app.py                  # real local model + real embeddings
    python local_app.py --fake-llm       # canned answers, real retrieval
    python local_app.py --fake           # no models at all (frontend work)

It reads and writes the same Supabase over the REST API as production, so there is
no local database to run.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import os
import re
import struct
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent

# app/config.py reads the environment at import time, so the .env has to be in
# place before anything under app/ is imported.
load_dotenv(REPO_ROOT / ".env")

# A FRONTEND_DIR left over from the Modal image (/assets/frontend) would serve a
# blank UI here; drop it so server.py's package-relative fallback wins.
if os.environ.get("FRONTEND_DIR") and not Path(os.environ["FRONTEND_DIR"]).exists():
    os.environ.pop("FRONTEND_DIR")

from app.config import STREAM_TRUNCATED, settings  # noqa: E402  (must follow load_dotenv)

# Ollama's OpenAI-compatible endpoint, and the published GGUF build of the 8B.
# For LM Studio use http://localhost:1234/v1 and whatever model id it lists.
LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "hf.co/Shinzmann/naija-petro-8b-GGUF:Q4_K_M")
LLM_API_KEY = os.environ.get("LOCAL_LLM_API_KEY", "local")

# Embeddings can come from the same OpenAI-compatible server rather than a second
# local copy of the model. Set LOCAL_EMBED_MODEL to the server's embedding model id
# (LM Studio ships nomic-embed-text-v1.5, the exact model this project uses) and the
# 550 MB sentence-transformers download is skipped entirely.
EMBED_MODEL_API = os.environ.get("LOCAL_EMBED_MODEL", "")
EMBED_BASE_URL = os.environ.get("LOCAL_EMBED_BASE_URL", "") or LLM_BASE_URL

# Written as code points, not literals, so no dash character appears in this file.
# em-dash, en-dash, unicode hyphen, non-breaking hyphen.
UNICODE_DASHES = (chr(0x2014), chr(0x2013), chr(0x2010), chr(0x2011))


def _strip_dashes(text: str) -> str:
    """Normalise unicode dashes to a plain hyphen (mirrors modal_app.LLMService)."""
    for ch in UNICODE_DASHES:
        text = text.replace(ch, "-")
    return text


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# LLM: any OpenAI-compatible server on localhost (Ollama, llama.cpp, LM Studio, vLLM)
# --------------------------------------------------------------------------- #
_client = None


def _openai_client():
    """One shared AsyncOpenAI client, built on first use."""
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _client


async def llm_stream(messages: list[dict], sampling: dict | None = None) -> AsyncIterator[str]:
    s = sampling or {}
    # reasoning on -> Qwen3 emits <think>...</think> first (the UI renders it as a
    # collapsible trace). Servers that do not understand the extra body ignore it.
    stream = await _openai_client().chat.completions.create(
        model=LLM_MODEL, messages=messages, stream=True,
        temperature=s.get("temperature", settings.temperature),
        top_p=s.get("top_p", settings.top_p),
        max_tokens=s.get("max_tokens", settings.max_new_tokens),
        extra_body={"chat_template_kwargs": {"enable_thinking": bool(s.get("reasoning", True))},
                    "repetition_penalty": s.get("repetition_penalty", 1.1)},
    )
    finish = None
    async for chunk in stream:
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        if ch.delta and ch.delta.content:
            yield _strip_dashes(ch.delta.content)
        if ch.finish_reason:
            finish = ch.finish_reason
    if finish == "length":      # hit max_tokens -> tell the UI the answer was cut off
        yield STREAM_TRUNCATED


async def llm_complete(messages: list[dict], sampling: dict | None = None) -> str:
    s = sampling or {}
    resp = await _openai_client().chat.completions.create(
        model=LLM_MODEL, messages=messages, stream=False,
        temperature=s.get("temperature", settings.temperature),
        top_p=s.get("top_p", settings.top_p),
        max_tokens=s.get("max_tokens", settings.max_new_tokens),
        extra_body={"chat_template_kwargs": {"enable_thinking": bool(s.get("reasoning", False))}},
    )
    return _strip_dashes(resp.choices[0].message.content or "")


# --------------------------------------------------------------------------- #
# Encoders: sentence-transformers on CPU, loaded on first use (they download)
# --------------------------------------------------------------------------- #
_embedder = None
_embedder_lock = asyncio.Lock()
_reranker = None
_reranker_lock = asyncio.Lock()


async def _get_embedder():
    global _embedder
    if _embedder is None:
        async with _embedder_lock:
            if _embedder is None:
                from app.rag.embeddings import EmbeddingModel

                print(f"[local] loading embeddings: {settings.embed_model} on CPU "
                      "(first use, may download weights)")
                _embedder = await asyncio.to_thread(EmbeddingModel, settings.embed_model, "cpu")
    return _embedder


async def embed(texts: list[str], mode: str = "document") -> list[list[float]]:
    if EMBED_MODEL_API:
        return await api_embed(texts, mode)
    model = await _get_embedder()
    # encode() is blocking CPU work: keep it off the event loop.
    return await asyncio.to_thread(model.encode, texts, mode)


_embed_client = None
_embed_dim_checked = False


async def api_embed(texts: list[str], mode: str = "document") -> list[list[float]]:
    """Embed through an OpenAI-compatible /v1/embeddings endpoint.

    nomic-embed-text-v1.5 is asymmetric: the task prefix decides whether a text is
    treated as a query or a stored passage, and the server does not add it for us,
    so it is applied here exactly as app/rag/embeddings.py does. Vectors are
    L2-normalised for parity with the stored ones; pgvector's cosine operator does
    not need it, but it keeps coverage scores comparable with production.
    """
    global _embed_client, _embed_dim_checked
    import math

    from app.rag.embeddings import DOC_PREFIX, QUERY_PREFIX

    if _embed_client is None:
        from openai import AsyncOpenAI

        _embed_client = AsyncOpenAI(base_url=EMBED_BASE_URL, api_key=LLM_API_KEY)

    prefix = QUERY_PREFIX if mode == "query" else DOC_PREFIX
    resp = await _embed_client.embeddings.create(
        model=EMBED_MODEL_API, input=[prefix + (t or "") for t in texts]
    )
    vectors = [list(d.embedding) for d in sorted(resp.data, key=lambda d: d.index)]

    if vectors and not _embed_dim_checked:
        _embed_dim_checked = True
        got = len(vectors[0])
        if got != settings.embed_dim:
            print(f"  ! embedding dimension mismatch: {EMBED_MODEL_API} returned {got}, "
                  f"the store expects {settings.embed_dim}. Retrieval will fail. "
                  "Use a matching model or unset LOCAL_EMBED_MODEL.", flush=True)

    out = []
    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


async def _get_reranker():
    global _reranker
    if _reranker is None:
        async with _reranker_lock:
            if _reranker is None:
                from app.rag.embeddings import Reranker

                print(f"[local] loading reranker: {settings.rerank_model} on CPU "
                      "(first use, may download weights)")
                _reranker = await asyncio.to_thread(Reranker, settings.rerank_model)
    return _reranker


async def rerank(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    model = await _get_reranker()
    return await asyncio.to_thread(model.score, query, passages)


# --------------------------------------------------------------------------- #
# Fakes: exercise the UI and the retrieval path with nothing installed
# --------------------------------------------------------------------------- #
FAKE_ANSWER = """## Local fake answer

This response comes from `local_app.py --fake-llm`, so no model was called. It is
here to exercise the streaming UI: markdown, inline citations, tables, and math.

Nigerian export grades are compared mainly on API gravity and sulphur content [1],
while the fiscal terms that apply to them are set by the Petroleum Industry Act
2021 [2].

| Grade | API gravity | Sulphur (wt %) |
|---|---|---|
| Bonny Light | 32.9 | 0.16 |
| Qua Iboe | 36.0 | 0.12 |
| Forcados | 29.7 | 0.20 |

Hydrostatic pressure at depth follows \\( p = 0.052 \\times \\rho \\times h \\), with
\\( \\rho \\) in ppg and \\( h \\) in feet, so 10.5 ppg mud at 8,000 ft gives about
4,368 psi.

Numbers above are placeholders. Start the real model and drop the flag to get a
grounded answer.
"""

# llm_complete drives two best-effort features: calculator selection (wants JSON,
# gets none here, so it is skipped) and follow-up chips (these lines).
FAKE_COMPLETION = (
    "What does the PIA 2021 change for gas flaring penalties?\n"
    "How is Bonny Light priced against Brent?\n"
    "Which NUPRC rules govern field development plans?\n"
)

_TOKEN_RE = re.compile(r"\S+\s*")


async def fake_llm_stream(messages: list[dict], sampling: dict | None = None) -> AsyncIterator[str]:
    """Stream a canned markdown answer token by token, like a slow model."""
    for token in _TOKEN_RE.findall(FAKE_ANSWER):
        yield token
        await asyncio.sleep(0.02)


async def fake_llm_complete(messages: list[dict], sampling: dict | None = None) -> str:
    return FAKE_COMPLETION


def _fake_vector(text: str, dim: int) -> list[float]:
    """Deterministic unit vector derived from the text, same shape as a real embedding."""
    seed = (text or "").encode("utf-8")
    raw = bytearray()
    counter = 0
    while len(raw) < dim * 4:
        raw += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    ints = struct.unpack(f">{dim}i", bytes(raw[: dim * 4]))
    vals = [i / 2_147_483_648.0 for i in ints]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]     # L2-normalised, as pgvector cosine expects


async def fake_embed(texts: list[str], mode: str = "document") -> list[list[float]]:
    if isinstance(texts, str):
        texts = [texts]
    return [_fake_vector(f"{mode}:{t}", settings.embed_dim) for t in texts]


# --------------------------------------------------------------------------- #
# Background enrichment: the local stand-in for Modal's fire-and-forget spawn
# --------------------------------------------------------------------------- #
_background: set[asyncio.Task] = set()


def _make_spawn_enrich(embed_fn: Callable) -> Callable[[str], None]:
    async def _run(query: str) -> None:
        from app.rag import ingest

        try:
            await ingest.ingest_query(query, embed_fn)
        except Exception as e:   # never surfaces to the request that triggered it
            print(f"[local] enrich failed: {e}")

    def spawn_enrich(query: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:     # called outside the server loop: nothing to schedule on
            return
        task = loop.create_task(_run(query))
        _background.add(task)    # hold a reference so the task is not garbage collected
        task.add_done_callback(_background.discard)

    return spawn_enrich


# --------------------------------------------------------------------------- #
# App factory + CLI
# --------------------------------------------------------------------------- #
def _disable_ingestion() -> None:
    """Stop anything being written into the knowledge base.

    Used only with faked embeddings. The local app shares the production
    Supabase, and retrieval ingests live sources on its own whenever local
    coverage looks weak, which hashed placeholder vectors guarantee. Without
    this, running the UI in fake mode would quietly poison the real store with
    documents whose embeddings mean nothing.
    """
    from app.rag import ingest

    async def _no_query(*a, **kw):
        return {"searched": 0, "ingested_docs": 0, "new_chunks": 0, "urls": []}

    async def _no_text(*a, **kw):
        return {"inserted": False, "chunk_count": 0, "reason": "ingestion disabled in fake mode"}

    async def _no_urls(*a, **kw):
        return {"ingested_docs": 0, "new_chunks": 0}

    ingest.ingest_query = _no_query
    ingest.ingest_text = _no_text
    ingest.ingest_urls = _no_urls


def create_local_app():
    """Build the FastAPI app with local deps. Used by uvicorn (factory=True).

    The fake-mode flags arrive as environment variables because --reload re-imports
    this module in a child process that never sees the parsed arguments.
    """
    from app.api.server import Deps, create_app

    fake_llm = _env_flag("LOCAL_FAKE_LLM")
    fake_embed_mode = _env_flag("LOCAL_FAKE_EMBED")
    embed_fn = fake_embed if fake_embed_mode else embed
    if fake_embed_mode:
        _disable_ingestion()

    deps = Deps(
        embed=embed_fn,
        llm_stream=fake_llm_stream if fake_llm else llm_stream,
        llm_complete=fake_llm_complete if fake_llm else llm_complete,
        # Fake embeddings mean "download nothing", so the cross-encoder stays off too.
        rerank=rerank if (settings.enable_rerank and not fake_embed_mode) else None,
        # The store is shared with production, so never let hashed placeholder
        # vectors be written into it: no enrichment while embeddings are faked.
        spawn_enrich=None if fake_embed_mode else _make_spawn_enrich(embed_fn),
    )
    return create_app(deps)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="local_app.py",
        description="Run Naija-Petro locally: no Modal, no cloud GPU.",
    )
    p.add_argument("--fake-llm", action="store_true",
                   help="stream a canned markdown answer instead of calling a model")
    p.add_argument("--fake-embed", action="store_true",
                   help="deterministic hashed vectors instead of the embedding model")
    p.add_argument("--fake", action="store_true", help="shorthand for --fake-llm --fake-embed")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="restart the server on code changes")
    return p.parse_args(argv)


def _print_summary(args: argparse.Namespace, fake_llm: bool, fake_embed_mode: bool) -> None:
    """Show which backend each dep uses. Never prints a key, only whether one is set."""
    from app.api.server import FRONTEND_DIR

    llm = "canned answer (--fake-llm)" if fake_llm else f"{LLM_MODEL} via {LLM_BASE_URL}"
    if fake_embed_mode:
        emb = f"hashed {settings.embed_dim}-dim vectors (--fake-embed)"
    elif EMBED_MODEL_API:
        emb = f"{EMBED_MODEL_API} via {EMBED_BASE_URL} (no local download)"
    else:
        emb = f"{settings.embed_model} on CPU, loaded on first use"
    rerank_on = settings.enable_rerank and not fake_embed_mode
    rr = f"{settings.rerank_model} on CPU" if rerank_on else "off"
    tavily = "set" if settings.tavily_api_key else "MISSING"
    service_key = "set" if settings.supabase_service_key else "MISSING"

    # flush: stdout is block-buffered when redirected to a file, and uvicorn logs to
    # stderr, so without this the summary shows up after the server output.
    print("Naija-Petro, local mode", flush=True)
    print(f"  llm      : {llm}", flush=True)
    print(f"  embed    : {emb}", flush=True)
    print(f"  rerank   : {rr}", flush=True)
    print(f"  enrich   : background asyncio task, Tavily key {tavily}", flush=True)
    print(f"  supabase : {settings.supabase_url or '(not set)'} "
          f"schema={settings.supabase_db_schema}, service key {service_key}", flush=True)
    print(f"  frontend : {FRONTEND_DIR}{'' if FRONTEND_DIR.exists() else '  (missing: the UI will 404)'}",
          flush=True)
    print(f"  context  : {settings.context_char_budget} chars of sources, so the server needs "
          f"at least an 8k context window", flush=True)
    print(f"  open     : http://{args.host}:{args.port}", flush=True)
    if not settings.supabase_url:
        print("  ! SUPABASE_URL is not set. The UI still loads, but retrieval, history, "
              "and the daily limit check will fail. Set it in .env.", flush=True)
    if not fake_llm:
        print(f"  ! Expecting an OpenAI-compatible server at {LLM_BASE_URL}. "
              "Start LM Studio's server (`lms server start`) or run `ollama serve`, "
              "or use --fake-llm.", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    fake_llm = args.fake or args.fake_llm
    fake_embed_mode = args.fake or args.fake_embed
    os.environ["LOCAL_FAKE_LLM"] = "1" if fake_llm else "0"
    os.environ["LOCAL_FAKE_EMBED"] = "1" if fake_embed_mode else "0"

    _print_summary(args, fake_llm, fake_embed_mode)

    import uvicorn

    reload_opts = {"reload": True, "reload_dirs": [str(REPO_ROOT)]} if args.reload else {}
    uvicorn.run(
        "local_app:create_local_app", factory=True,
        host=args.host, port=args.port,
        app_dir=str(REPO_ROOT),     # so the reloader's child process can import this file
        **reload_opts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
