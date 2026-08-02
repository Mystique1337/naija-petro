"""Shared test setup.

The app reads its configuration from environment variables at import time
(app/config.py), so this module scrubs every app-owned variable BEFORE any
`app.*` import happens. That keeps the suite hermetic: no Supabase URL, no
Tavily key and no Langfuse credentials can leak in from a developer shell, so
nothing can reach the network, a database or a GPU, and `Settings()` always
falls back to its declared defaults.
"""
from __future__ import annotations

import os

_ENV_PREFIXES = (
    "SUPABASE_", "TAVILY_", "LANGFUSE_", "RAG_", "MODEL_", "EMBED_", "RERANK_",
    "LLM_", "RATE_LIMIT_", "ENABLE_",
)
_ENV_NAMES = (
    "ACCESS_KEY", "ADMIN_TOKEN", "FREE_DAILY_LIMIT", "MAX_QUERY_CHARS",
    "MAX_MODEL_LEN", "IP_SALT", "FRONTEND_DIR",
)

for _name in [k for k in os.environ if k.startswith(_ENV_PREFIXES)] + list(_ENV_NAMES):
    os.environ.pop(_name, None)

import pytest  # noqa: E402  (import after the environment is scrubbed)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """Belt and braces: no tracing client, and a clean rate-limiter per test.

    The burst limiter in app/api/server.py is an in-memory dict shared for the
    life of the process, so it is reset here; API tests additionally use a
    distinct client IP each.
    """
    from app import observability as obs

    monkeypatch.setattr(obs, "_init_done", True, raising=False)
    monkeypatch.setattr(obs, "_client", None, raising=False)
    monkeypatch.setattr(obs, "get_client", lambda: None)

    from app.api import server

    server._hits.clear()
    yield
    server._hits.clear()
