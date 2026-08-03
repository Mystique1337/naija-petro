"""Best-effort Langfuse (v4) tracing for the RAG pipeline.

Tracing must NEVER break a request: every Langfuse call is guarded, and if the
SDK is missing/misconfigured the helpers degrade to no-ops. We record one trace
per chat turn (a "chat" span carrying retrieval metadata) with a nested LLM
"generation". Trace-level attributes (session_id/user_id) use propagate_attributes.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import ExitStack, contextmanager

from app.config import settings

log = logging.getLogger("naija_petro")

_client = None
_init_done = False

# Strong refs to in-flight flushes: a bare future can be collected mid-run.
_flush_futures: set = set()


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
    except Exception as e:
        # Cached: this runs once, and if it fails every trace, span, generation
        # and feedback score is a no-op for the life of the container. Silently
        # is exactly how a whole subsystem stays dead for months.
        log.warning("langfuse disabled, client init failed: %s: %s", type(e).__name__, e)
        _client = None
    return _client


def _flush(client) -> None:
    """Push buffered observations out without parking the event loop.

    Langfuse's flush() blocks until its queue has drained, which is a network
    round trip, and this runs at the end of every chat turn inside the web
    container. Done inline it stalled the loop, and with it every other request
    the container was serving concurrently.
    """
    def _run() -> None:
        try:
            client.flush()
        except Exception as e:
            log.warning("langfuse flush failed: %s: %s", type(e).__name__, e)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:        # sync caller (scripts, tests): no loop to protect
        _run()
        return
    try:
        fut = loop.run_in_executor(None, _run)
        _flush_futures.add(fut)
        fut.add_done_callback(_flush_futures.discard)
    except Exception as e:
        log.warning("langfuse flush could not be scheduled: %s: %s", type(e).__name__, e)


def current_trace_id():
    """The active Langfuse trace id (for linking user feedback), or None."""
    try:
        client = get_client()
        return client.get_current_trace_id() if client else None
    except Exception as e:
        # Without an id the UI cannot attach a rating to the turn, so a thumbs
        # up/down quietly stops reaching Langfuse.
        log.warning("langfuse trace id unavailable: %s: %s", type(e).__name__, e)
        return None


class _Span:
    """Fully-guarded wrapper around a Langfuse v4 span."""

    def __init__(self, span):
        self._s = span

    def event(self, name: str, **kw):
        try:
            self._s.create_event(name=name, **kw)
        except Exception as e:
            log.warning("langfuse event %r dropped: %s: %s", name, type(e).__name__, e)

    def generation(self, name: str, model: str, input, output: str, usage: dict | None = None):
        try:
            gen = self._s.start_observation(name=name, as_type="generation", model=model, input=input)
            gen.update(output=output, usage_details=usage)
            gen.end()
        except Exception as e:
            # This is the prompt/answer pair, the whole point of the trace. A
            # swallow here leaves traces that look fine but carry no generation.
            log.warning("langfuse generation %r dropped: %s: %s", name, type(e).__name__, e)

    def update(self, **kw):
        try:
            self._s.update(**kw)
        except Exception as e:
            log.warning("langfuse span update dropped: %s: %s", type(e).__name__, e)


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
            except Exception as e:
                log.warning("langfuse trace metadata dropped: %s: %s", type(e).__name__, e)
        handle = _Span(span)
    except Exception as e:
        # Setup failed, degrade to no-op but still yield exactly once. The whole
        # turn goes untraced, so say so rather than losing it silently.
        log.warning("langfuse trace %r not started: %s: %s", name, type(e).__name__, e)
        try:
            stack.close()
        except Exception as close_err:
            log.warning("langfuse trace teardown failed: %s: %s",
                        type(close_err).__name__, close_err)
        stack = None

    try:
        yield handle
    finally:
        if stack is not None:
            try:
                stack.close()
            except Exception as e:
                log.warning("langfuse trace close failed: %s: %s", type(e).__name__, e)
        _flush(client)
