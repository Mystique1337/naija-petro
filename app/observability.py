"""Best-effort Langfuse (v4) tracing for the RAG pipeline.

Tracing must NEVER break a request: every Langfuse call is guarded, and if the
SDK is missing/misconfigured the helpers degrade to no-ops. We record one trace
per chat turn (a "chat" span carrying retrieval metadata) with a nested LLM
"generation". Trace-level attributes (session_id/user_id) use propagate_attributes.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager

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

        kwargs = dict(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
        )
        if settings.langfuse_host:
            kwargs["host"] = settings.langfuse_host
        _client = Langfuse(**kwargs)
    except Exception:
        _client = None
    return _client


class _Span:
    """Fully-guarded wrapper around a Langfuse v4 span."""

    def __init__(self, span):
        self._s = span

    def event(self, name: str, **kw):
        try:
            self._s.create_event(name=name, **kw)
        except Exception:
            pass

    def generation(self, name: str, model: str, input, output: str, usage: dict | None = None):
        try:
            gen = self._s.start_observation(name=name, as_type="generation", model=model, input=input)
            gen.update(output=output, usage_details=usage)
            gen.end()
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

    stack = ExitStack()
    handle = _Noop()
    try:
        from langfuse import propagate_attributes

        stack.enter_context(propagate_attributes(session_id=session_id, user_id=user_id))
        span = stack.enter_context(
            client.start_as_current_observation(name=name, as_type="span", input=input)
        )
        if metadata:
            try:
                span.update(metadata=metadata)
            except Exception:
                pass
        handle = _Span(span)
    except Exception:
        # Setup failed — degrade to no-op but still yield exactly once.
        try:
            stack.close()
        except Exception:
            pass
        stack = None

    try:
        yield handle
    finally:
        if stack is not None:
            try:
                stack.close()
            except Exception:
                pass
        try:
            client.flush()
        except Exception:
            pass
