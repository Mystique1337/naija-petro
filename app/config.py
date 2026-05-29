"""Central configuration, read from environment variables (see .env.example).

Values are read at import time. In Modal, these arrive via attached Secrets; in
local dev they come from a .env loaded by the entrypoint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    llm_gpu: str = os.environ.get("LLM_GPU", "A10G")
    llm_scaledown_window: int = _i("LLM_SCALEDOWN_WINDOW", 120)
    max_model_len: int = _i("MAX_MODEL_LEN", 8192)

    # --- Embeddings / reranker ---
    embed_model: str = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    embed_dim: int = _i("EMBED_DIM", 768)
    enable_rerank: bool = _b("ENABLE_RERANK", False)   # off by default (CPU rerank is slow); RRF hybrid stays
    rerank_model: str = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    # --- Data stores / services ---
    supabase_db_url: str = os.environ.get("SUPABASE_DB_URL", "")
    tavily_api_key: str = os.environ.get("TAVILY_API_KEY", "")

    # --- Langfuse ---
    langfuse_public_key: str = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.environ.get("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # --- Dynamic-RAG tuning ---
    coverage_threshold: float = _f("RAG_COVERAGE_THRESHOLD", 0.55)
    min_chunks: int = _i("RAG_MIN_CHUNKS", 3)
    top_k: int = _i("RAG_TOP_K", 20)
    final_k: int = _i("RAG_FINAL_K", 6)
    always_enrich: bool = _b("RAG_ALWAYS_ENRICH", True)
    doc_ttl_days: int = _i("RAG_DOC_TTL_DAYS", 30)

    # --- Chunking ---
    chunk_chars: int = _i("RAG_CHUNK_CHARS", 1200)
    chunk_overlap: int = _i("RAG_CHUNK_OVERLAP", 200)

    # --- Generation defaults ---
    max_new_tokens: int = _i("RAG_MAX_NEW_TOKENS", 1024)
    temperature: float = _f("RAG_TEMPERATURE", 0.4)
    top_p: float = _f("RAG_TOP_P", 0.9)

    # --- Security / limits ---
    access_key: str = os.environ.get("ACCESS_KEY", "")        # empty = open access
    max_query_chars: int = _i("MAX_QUERY_CHARS", 2000)
    rate_limit_max: int = _i("RATE_LIMIT_MAX", 20)            # requests per window
    rate_limit_window_s: int = _i("RATE_LIMIT_WINDOW_S", 60)
    rate_limit_max_hour: int = _i("RATE_LIMIT_MAX_HOUR", 200)
    ip_salt: str = os.environ.get("IP_SALT", "naija-petro-salt")
    enable_followups: bool = _b("ENABLE_FOLLOWUPS", True)

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()

APP_NAME = "naija-petro"

SYSTEM_PROMPT = (
    "You are Naija-Petro, an expert petroleum-engineering AI assistant with a focus "
    "on the Nigerian oil & gas sector. You provide precise, technically accurate answers "
    "covering drilling, reservoir engineering, production, completions, EOR, well testing, "
    "petroleum geoscience, and Nigerian regulation (PIA 2021, NUPRC, NMDPRA, NNPC). "
    "Include equations, units, and practical considerations where relevant. "
    "Format with markdown; write mathematics in LaTeX using \\( \\) for inline and \\[ \\] for display. "
    "Never use em-dashes or en-dashes; use commas, colons, or hyphens instead."
)
