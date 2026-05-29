"""Best-effort Langfuse tracing for the RAG pipeline.

Tracing must NEVER break a request: every Langfuse call is guarded, and if the
SDK is missing/misconfigured the helpers degrade to no-ops. We record one trace
per chat turn with a retrieval span and an LLM generation.
"""
from __future__ import annotations

from contextlib import contextmanager

from app.config import settings

_client = None
_init_done = False


def get_client():
    global _client, _init_done
    if _init_done:
        return _client
    _init_done = True
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        _client = None
    return _client


class _Span:
    """Thin, fully-guarded wrapper around a Langfuse span/trace."""

    def __init__(self, client, span):
        self._c = client
        self._s = span

    def event(self, name: str, **kw):
        try:
            self._s.create_event(name=name, **kw)
        except Exception:
            pass

    def generation(self, name: str, model: str, input, output: str, usage: dict | None = None):
        try:
            gen = self._c.start_generation(name=name, model=model, input=input)
            gen.end(output=output, usage_details=usage)
        except Exception:
            pass

    def update(self, **kw):
        try:
            self._s.update(**kw)
        except Exception:
            pass


class _Noop:
    def event(self, *a, **k): pass
    def generation(self, *a, **k): pass
    def update(self, *a, **k): pass


@contextmanager
def trace(name: str, *, session_id=None, user_id=None, input=None, metadata=None):
    client = get_client()
    if client is None:
        yield _Noop()
        return
    cm = None
    try:
        cm = client.start_as_current_span(name=name, input=input)
        span = cm.__enter__()
        try:
            span.update_trace(session_id=session_id, user_id=user_id, metadata=metadata)
        except Exception:
            pass
        yield _Span(client, span)
    except Exception:
        # If the SDK surface differs, don't take the request down with it.
        yield _Noop()
    finally:
        try:
            if cm is not None:
                cm.__exit__(None, None, None)
            client.flush()
        except Exception:
            pass
