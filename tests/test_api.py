"""FastAPI surface, driven with stub GPU callables and a stubbed store.

`create_app(Deps(...))` takes every model-backed callable as an argument, so the
whole HTTP layer can be exercised without Modal, a GPU or a network. The store
functions are monkeypatched on `app.rag.db` (the server looks them up on the
module at call time) and retrieval is replaced at `app.api.server.retrieve`,
which is the name the route actually calls.
"""
from __future__ import annotations

import itertools
import json

import pytest
from fastapi.testclient import TestClient

from app.api import server
from app.api.server import Deps, create_app
from app.config import STREAM_TRUNCATED, settings
from app.rag import db
from app.rag.prompts import RetrievedChunk
from app.rag.retriever import RetrieveResult

_ips = itertools.count(1)


@pytest.fixture
def ip():
    """A distinct client IP per test: the burst limiter is process-wide."""
    return {"x-forwarded-for": f"203.0.113.{next(_ips) % 250 + 1}"}


@pytest.fixture(autouse=True)
def stub_store(monkeypatch):
    """Nothing may reach Supabase."""
    async def _token_active(token):
        return False

    async def _daily_ip_count(ip_hash):
        return 0

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(db, "token_active", _token_active)
    monkeypatch.setattr(db, "daily_ip_count", _daily_ip_count)
    monkeypatch.setattr(db, "log_usage", _noop)
    monkeypatch.setattr(db, "save_turns", _noop)
    monkeypatch.setattr(db, "log_feedback", _noop)


CHUNKS = [
    RetrievedChunk(
        content="Nigeria produced roughly 1.5 million barrels per day.",
        source_url="https://www.nuprc.gov.ng/report",
        title="NUPRC monthly report",
        domain="nuprc.gov.ng",
        source_tier=1,
        similarity=0.82,
    ),
    RetrievedChunk(
        content="OPEC lists the Nigerian quota for the period.",
        source_url="https://opec.org/nigeria",
        title="OPEC country page",
        domain="opec.org",
        source_tier=2,
        similarity=0.71,
    ),
]


def stub_retrieve(monkeypatch, chunks=CHUNKS, **kwargs):
    """Replace retrieval; returns the list of queries it was called with."""
    calls = []

    async def _retrieve(query, embed_fn, rerank_fn=None):
        calls.append(query)
        return RetrieveResult(chunks=list(chunks), coverage=kwargs.get("coverage", 0.82),
                              candidates=kwargs.get("candidates", 7),
                              enriched=kwargs.get("enriched", False),
                              enrichment=kwargs.get("enrichment", {"new_chunks": 0}),
                              reranked=kwargs.get("reranked", False))

    monkeypatch.setattr(server, "retrieve", _retrieve)
    return calls


async def fake_embed(texts, mode):
    return [[0.0, 1.0] for _ in texts]


def stream_of(tokens, sink=None):
    async def _stream(messages, sampling):
        if sink is not None:
            sink.append({"messages": messages, "sampling": sampling})
        for tok in tokens:
            yield tok
    return _stream


def build_client(**deps_kwargs) -> TestClient:
    deps = Deps(embed=fake_embed, llm_stream=stream_of(["Hello ", "world."]), **deps_kwargs)
    return TestClient(create_app(deps))


def sse_events(response) -> list[dict]:
    return [json.loads(line[len("data: "):])
            for line in response.text.splitlines()
            if line.startswith("data: ")]


def kinds(events) -> list[str]:
    return [e["type"] for e in events]


def only(events, kind) -> dict:
    matches = [e for e in events if e["type"] == kind]
    assert len(matches) == 1, f"expected exactly one {kind} event, got {kinds(events)}"
    return matches[0]


# --------------------------------------------------------------------------- #
# Plain endpoints
# --------------------------------------------------------------------------- #
def test_healthz_shape():
    from app import __version__

    r = build_client().get("/healthz")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "version": __version__,
        "open": True,
        "daily_limit": settings.free_daily_limit,
        "admin": bool(settings.admin_token),
    }


def test_tools_lists_every_calculator_with_arg_specs():
    from app.tools.calculators import CALCULATORS, TOOL_SPECS

    body = build_client().get("/tools").json()
    tools = body["calculators"]
    assert {t["name"] for t in tools} == set(TOOL_SPECS) == set(CALCULATORS)
    for t in tools:
        assert t["label"] and t["description"]
        assert t["args"], f"{t['name']} exposes no arguments"
        for arg in t["args"]:
            assert set(arg) <= {"name", "label", "opt"}
            assert arg["name"] and arg["label"]


def test_tools_run_computes():
    r = build_client().post("/tools/run", json={
        "name": "hydrostatic_pressure",
        "args": {"mud_weight_ppg": 10, "tvd_ft": 10000},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["pressure_psi"] == pytest.approx(5200.0, rel=1e-3)
    assert body["gradient_psi_per_ft"] == pytest.approx(0.52, rel=1e-3)


def test_tools_run_unknown_name_returns_an_error():
    r = build_client().post("/tools/run", json={"name": "not_a_tool", "args": {}})
    assert r.status_code == 200
    assert "error" in r.json()


def test_tools_run_bad_args_returns_an_error_not_a_500():
    r = build_client().post("/tools/run", json={"name": "hydrostatic_pressure", "args": {}})
    assert r.status_code == 200
    assert "error" in r.json()


# --------------------------------------------------------------------------- #
# /chat gates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("message", ["", "   \n  "])
def test_chat_rejects_an_empty_message(message, ip):
    r = build_client().post("/chat", json={"message": message}, headers=ip)
    assert r.status_code == 400
    assert r.json()["error"] == "Empty message"


def test_chat_rejects_an_over_long_message(ip):
    r = build_client().post("/chat", json={"message": "x" * (settings.max_query_chars + 1)}, headers=ip)
    assert r.status_code == 400
    assert "too long" in r.json()["error"].lower()


def test_chat_accepts_a_message_at_the_limit(monkeypatch, ip):
    stub_retrieve(monkeypatch)
    r = build_client().post("/chat", json={"message": "x" * settings.max_query_chars}, headers=ip)
    assert r.status_code == 200


def test_chat_burst_limit_returns_429(monkeypatch, ip):
    stub_retrieve(monkeypatch)
    client = build_client()
    for _ in range(settings.rate_limit_max):
        assert client.post("/chat", json={"message": "hello"}, headers=ip).status_code == 200
    blocked = client.post("/chat", json={"message": "hello"}, headers=ip)
    assert blocked.status_code == 429
    assert "too quickly" in blocked.json()["error"]


def test_chat_daily_limit_returns_429(monkeypatch, ip):
    stub_retrieve(monkeypatch)

    async def _used(ip_hash):
        return settings.free_daily_limit

    monkeypatch.setattr(db, "daily_ip_count", _used)
    r = build_client().post("/chat", json={"message": "hello"}, headers=ip)
    assert r.status_code == 429
    assert r.json()["limit"] == "daily"


# --------------------------------------------------------------------------- #
# /chat streaming
# --------------------------------------------------------------------------- #
def test_chat_happy_path_streams_meta_tokens_and_done(monkeypatch, ip):
    queries = stub_retrieve(monkeypatch)
    r = build_client().post("/chat", json={"message": "How much oil does Nigeria produce?"}, headers=ip)

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r)
    seen = kinds(events)
    assert "meta" in seen and "token" in seen and "done" in seen
    assert "error" not in seen
    assert queries == ["How much oil does Nigeria produce?"]

    meta = only(events, "meta")
    assert meta["coverage"] == pytest.approx(0.82)
    assert meta["enriched"] is False
    assert [s["url"] for s in meta["sources"]] == [
        "https://www.nuprc.gov.ng/report", "https://opec.org/nigeria",
    ]
    assert meta["sources"][0] == {
        "n": 1, "title": "NUPRC monthly report", "url": "https://www.nuprc.gov.ng/report",
        "domain": "nuprc.gov.ng", "tier": 1,
    }

    assert "".join(e["t"] for e in events if e["type"] == "token") == "Hello world."
    done = only(events, "done")
    assert done == {"type": "done", "chars": len("Hello world."), "truncated": False}
    # meta must arrive before the first token so the UI can render sources early.
    assert seen.index("meta") < seen.index("token")


def test_chat_reports_truncation_from_the_sentinel(monkeypatch, ip):
    stub_retrieve(monkeypatch)
    deps = Deps(embed=fake_embed, llm_stream=stream_of(["partial answer", STREAM_TRUNCATED]))
    r = TestClient(create_app(deps)).post("/chat", json={"message": "long question"}, headers=ip)

    events = sse_events(r)
    tokens = [e["t"] for e in events if e["type"] == "token"]
    assert tokens == ["partial answer"], "the sentinel must not leak into the answer"
    assert only(events, "done")["truncated"] is True


def test_chat_no_sources_still_streams(monkeypatch, ip):
    stub_retrieve(monkeypatch, chunks=[], coverage=0.0)
    r = build_client().post("/chat", json={"message": "something unheard of"}, headers=ip)
    events = sse_events(r)
    assert only(events, "meta")["sources"] == []
    assert only(events, "done")["chars"] > 0


def test_chat_si_units_prefix_reaches_the_model(monkeypatch, ip):
    stub_retrieve(monkeypatch)
    sink = []
    deps = Deps(embed=fake_embed, llm_stream=stream_of(["ok"], sink=sink))
    TestClient(create_app(deps)).post("/chat", json={"message": "pressure?", "units": "si"}, headers=ip)

    assert sink, "the model was never called"
    assert sink[0]["messages"][-1]["content"].startswith("Present all numeric results in SI units")


def test_chat_continuation_skips_retrieval(monkeypatch, ip):
    queries = stub_retrieve(monkeypatch)
    sink = []
    deps = Deps(embed=fake_embed, llm_stream=stream_of([" and finally the rest."], sink=sink))
    r = TestClient(create_app(deps)).post("/chat", json={
        "message": "continue",
        "continuation": True,
        "history": [
            {"role": "user", "content": "explain the PIA"},
            {"role": "assistant", "content": "The PIA 2021 is"},
            {"role": "system", "content": "should be dropped"},
        ],
    }, headers=ip)

    events = sse_events(r)
    assert queries == [], "a continuation must not run retrieval again"
    assert "meta" not in kinds(events)
    assert [e["t"] for e in events if e["type"] == "token"] == [" and finally the rest."]
    assert only(events, "done")["chars"] == len(" and finally the rest.")

    sent = sink[0]["messages"]
    assert sent[0]["role"] == "system"
    assert [m["role"] for m in sent[1:]] == ["user", "assistant", "user"]
    assert sent[-1]["content"].startswith("Continue your previous answer")
    assert sink[0]["sampling"]["reasoning"] is False


def test_chat_tool_prepass_emits_a_verified_result(monkeypatch, ip):
    stub_retrieve(monkeypatch)
    sink = []

    async def llm_complete(messages, sampling):
        if messages[0]["content"].startswith("You pick a calculator"):
            return ('Sure: {"tool": "hydrostatic_pressure", '
                    '"args": {"mud_weight_ppg": 10, "tvd_ft": 10000}}')
        return "What is the fracture gradient?\n2. How does mud weight affect ECD?\nWhy does TVD matter?"

    deps = Deps(embed=fake_embed, llm_stream=stream_of(["5200 psi"], sink=sink),
                llm_complete=llm_complete)
    r = TestClient(create_app(deps)).post("/chat", json={
        "message": "Calculate the hydrostatic pressure at 10000 ft with 10 ppg mud",
    }, headers=ip)

    events = sse_events(r)
    tool = only(events, "tool")
    assert tool["name"] == "hydrostatic_pressure"
    assert tool["result"]["pressure_psi"] == pytest.approx(5200.0, rel=1e-3)
    # The verified figure is injected into the prompt the model actually sees.
    assert "verified engineering calculator" in sink[0]["messages"][-1]["content"]

    followups = only(events, "followups")
    assert followups["items"] == [
        "What is the fracture gradient?",
        "How does mud weight affect ECD?",
        "Why does TVD matter?",
    ]


def test_chat_reports_a_model_failure_as_an_error_event(monkeypatch, ip):
    stub_retrieve(monkeypatch)

    async def _boom(messages, sampling):
        raise RuntimeError("GPU unavailable")
        yield ""      # pragma: no cover - makes this an async generator

    r = TestClient(create_app(Deps(embed=fake_embed, llm_stream=_boom))).post(
        "/chat", json={"message": "hello"}, headers=ip)

    assert r.status_code == 200
    error = only(sse_events(r), "error")
    assert "GPU unavailable" in error["message"]
