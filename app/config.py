"""Central configuration, read from environment variables (see .env.example).

Values are read at import time. In Modal, these arrive via attached Secrets; in
local dev they come from a .env loaded by the entrypoint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, str(int(default))).strip().lower() in ("1", "true", "yes", "on")


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- Model serving ---
    model_repo: str = os.environ.get("MODEL_REPO", "Shinzmann/naija-petro-8b")
    llm_gpu: str = os.environ.get("LLM_GPU", "L4")    # L4 (24GB) fits the 8B fp16 and is cheaper than A10G
    llm_scaledown_window: int = _i("LLM_SCALEDOWN_WINDOW", 120)
    max_model_len: int = _i("MAX_MODEL_LEN", 8192)

    # --- Embeddings / reranker ---
    embed_model: str = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    embed_dim: int = _i("EMBED_DIM", 768)
    enable_rerank: bool = _b("ENABLE_RERANK", False)   # off by default (CPU rerank is slow); RRF hybrid stays
    rerank_model: str = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    # --- Data stores / services ---
    # The RAG store lives in a self-hosted Supabase and is reached over the REST
    # (PostgREST) API, which is publicly reachable (works from Modal), see
    # app/rag/db.py. `supabase_db_schema` is the Postgres schema the tables and
    # functions live in (each app gets its own on a shared instance).
    supabase_url: str = os.environ.get("SUPABASE_URL", "").rstrip("/")
    supabase_service_key: str = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        or os.environ.get("SUPABASE_SERVICE_KEY", "")
    )
    supabase_anon_key: str = os.environ.get("SUPABASE_ANON_KEY", "")
    supabase_db_schema: str = os.environ.get("SUPABASE_DB_SCHEMA", "naija_petro")
    # Supabase serves PostgREST behind its gateway under /rest/v1. A PostgREST
    # running on its own serves the same API at the root, so a local stack sets
    # this to an empty string.
    supabase_rest_path: str = os.environ.get("SUPABASE_REST_PATH", "/rest/v1")
    # Optional direct Postgres DSN (used only by the offline admin scripts in
    # scripts/, not by the running app). Left blank in normal REST operation.
    supabase_db_url: str = os.environ.get("SUPABASE_DB_URL", "")
    tavily_api_key: str = os.environ.get("TAVILY_API_KEY", "")

    # --- Langfuse ---
    langfuse_public_key: str = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.environ.get("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # --- Dynamic-RAG tuning ---
    coverage_threshold: float = _f("RAG_COVERAGE_THRESHOLD", 0.55)
    min_chunks: int = _i("RAG_MIN_CHUNKS", 3)
    top_k: int = _i("RAG_TOP_K", 35)             # candidates pulled from the store
    final_k: int = _i("RAG_FINAL_K", 12)         # passages kept for the answer
    tavily_max_results: int = _i("TAVILY_MAX_RESULTS", 10)
    # Hard cap on context characters fed to the model (protects the window + cost
    # even when many sources are retrieved).
    context_char_budget: int = _i("RAG_CONTEXT_CHARS", 20000)
    always_enrich: bool = _b("RAG_ALWAYS_ENRICH", True)
    doc_ttl_days: int = _i("RAG_DOC_TTL_DAYS", 30)

    # --- Chunking ---
    chunk_chars: int = _i("RAG_CHUNK_CHARS", 1500)
    chunk_overlap: int = _i("RAG_CHUNK_OVERLAP", 200)

    # --- Generation defaults ---
    max_new_tokens: int = _i("RAG_MAX_NEW_TOKENS", 1024)
    temperature: float = _f("RAG_TEMPERATURE", 0.4)
    top_p: float = _f("RAG_TOP_P", 0.9)

    # --- Security / limits ---
    # The interface is OPEN to test. Anonymous visitors get `free_daily_limit`
    # queries per calendar day (UTC); holders of an active access token bypass
    # that limit. `admin_token` guards the /admin panel where tokens are managed.
    access_key: str = os.environ.get("ACCESS_KEY", "")        # legacy hard gate; empty = open
    admin_token: str = os.environ.get("ADMIN_TOKEN", "")      # unlocks the admin panel
    free_daily_limit: int = _i("FREE_DAILY_LIMIT", 10)        # anonymous queries per day
    max_query_chars: int = _i("MAX_QUERY_CHARS", 2000)
    rate_limit_max: int = _i("RATE_LIMIT_MAX", 20)            # burst requests per window (anti-abuse)
    rate_limit_window_s: int = _i("RATE_LIMIT_WINDOW_S", 60)
    rate_limit_max_hour: int = _i("RATE_LIMIT_MAX_HOUR", 200)
    ip_salt: str = os.environ.get("IP_SALT", "naija-petro-salt")
    enable_followups: bool = _b("ENABLE_FOLLOWUPS", True)
    # Allow /chat and the UI to pre-boot the GPU. Costs an L4 that then idles for
    # llm_scaledown_window seconds, but removes the cold start from the wait.
    # Inside /chat this is free (that turn boots the GPU anyway); the /warm
    # endpoint is the one that can spend without a question being asked, so it is
    # capped both per IP and globally. The global cap bounds the worst case even
    # if someone rotates IPs: at most this many boots per hour.
    enable_warm: bool = _b("ENABLE_WARM", True)
    warm_max_per_hour: int = _i("WARM_MAX_PER_HOUR", 12)
    warm_per_ip_window_s: int = _i("WARM_PER_IP_WINDOW_S", 600)

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()

APP_NAME = "naija-petro"

# Sentinel a streaming backend yields as its final item when generation stopped
# because it hit the token limit. Lets the API flag the answer as truncated so the
# UI can offer a "Continue" action.
STREAM_TRUNCATED = "\x00\x00NP_TRUNCATED\x00\x00"

# Injected as the user turn when continuing a previously truncated answer.
CONTINUE_INSTRUCTION = (
    "Continue your previous answer exactly from where it stopped. Do not repeat "
    "anything you already wrote, do not restate the question, and do not add a new "
    "introduction or heading. Pick up mid-sentence if needed and finish the response."
)

SYSTEM_PROMPT = (
    "You are Naija-Petro, an expert petroleum-engineering AI assistant focused on the "
    "Nigerian oil & gas sector, covering drilling, reservoir engineering, production, "
    "completions, EOR, well testing, petroleum geoscience, and Nigerian regulation "
    "(PIA 2021, NUPRC, NMDPRA, NNPC). Answer with engineering rigour: state assumptions "
    "explicitly, show the governing equations and a worked step-by-step solution for any "
    "calculation, carry units throughout, and give numeric results with units and sensible "
    "significant figures. Name the relevant correlations, standards, or methods when they "
    "apply (for example Darcy, Vogel, Buckley-Leverett, material balance, SPE or API "
    "references). Note limitations and flag where field data or a qualified engineer is "
    "needed. Structure answers with markdown headings and tables; write mathematics in "
    "LaTeX using \\( \\) for inline and \\[ \\] for display. Ground claims in the provided "
    "sources and cite them. Never use em-dashes or en-dashes; use commas, colons, or hyphens."
)
