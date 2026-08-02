"""Evaluation harness for the Naija-Petro models and the deployed RAG app.

Why this file exists
--------------------
The project's previous evaluation was degenerate: an LLM judge returned a flat
3.00 out of 5.0 for both the base model and the fine-tuned model across 30
questions, a delta of exactly zero. A number that cannot move is not a
measurement, so no score could be published on the model cards.

This harness is built so that failure mode is detectable and, where possible,
prevented:

1. Scores are per dimension, never a single blended number, and every dimension
   carries an anchored rubric that says what a 1, a 3 and a 5 look like.
2. The judge must return strict JSON, one justification per dimension, and a
   verbatim quoted span for factual accuracy. The quote is checked against the
   answer in code, so a judge that invents evidence is visible.
3. Key-fact coverage is computed twice: once by the judge against the question's
   key_facts checklist, once lexically in code. If those two disagree wildly, or
   if the judge's value never moves, the run is suspect.
4. Numeric questions are graded deterministically against `numeric_answer` with a
   relative tolerance. The judge never decides arithmetic.
5. A control probe judges a deliberately poor canned answer alongside the real
   ones. A judge that cannot score the control below the real answers is broken.
6. The run scores itself: per-dimension variance and distinct-value counts, plus
   a loud warning when a dimension flatlines or when two compared systems differ
   by roughly zero, naming it as the previously observed degenerate-judge failure.

Targets: the deployed app (`/chat`, Server-Sent Events) or any OpenAI-compatible
chat endpoint (Ollama, vLLM, a hosted API), so base, fine-tuned and
fine-tuned-plus-RAG can be compared on identical questions.

Dependencies are deliberately light: httpx and python-dotenv, plus the standard
library. Nothing here imports the app package.

Usage examples are in eval/README.md. Nothing runs without an explicit target;
`--dry-run` prints the exact prompts and makes no network call.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = HERE / "questions.jsonl"

DEFAULT_JUDGE_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_JUDGE_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
DEFAULT_JUDGE_KEY_ENV = "NVIDIA_API_KEY"
DEFAULT_JUDGE_URL_ENV = "NVIDIA_BASE_URL"

# Mirrors the intent of app/config.py SYSTEM_PROMPT. Kept local so this script
# stays standalone. Use the SAME prompt for every system in a comparison, or the
# comparison measures prompting rather than the model. Override with
# --system-prompt-file, or drop it entirely with --no-system-prompt.
EVAL_SYSTEM_PROMPT = (
    "You are an expert petroleum-engineering assistant focused on the Nigerian oil and gas "
    "sector, covering drilling, reservoir engineering, production, completions, EOR, well "
    "testing, petroleum geoscience, and Nigerian regulation (PIA 2021, NUPRC, NMDPRA, NNPC). "
    "Answer with engineering rigour: state assumptions explicitly, show the governing "
    "equations and a worked step-by-step solution for any calculation, carry units "
    "throughout, and give numeric results with units and sensible significant figures. Name "
    "the relevant correlations, standards or methods when they apply. Say plainly when you "
    "do not know something or when a premise of the question is wrong, and never invent a "
    "citation, a statute, a section number or a figure."
)

# The judge scores this canned answer alongside real ones when --control-probe is
# used. It is fluent, on topic and almost content free, so a working judge scores
# it low. A judge that scores it near the real answers is not discriminating.
CONTROL_ANSWER = (
    "This is an important question in petroleum engineering and the Nigerian oil and gas "
    "industry. There are several factors to consider, and the answer depends on the specific "
    "circumstances of the field and the applicable regulations. Generally speaking, operators "
    "should follow best practice and consult the relevant authorities, since requirements can "
    "change over time. In summary, a careful assessment by qualified engineers is recommended "
    "before any decision is taken."
)

DEGENERATE_NOTE = (
    "This is the previously observed degenerate-judge failure: a judge that returns the same "
    "score regardless of the answer, which produced a flat 3.00 for both the base and the "
    "fine-tuned model and a delta of exactly zero. Treat the scores as unusable until fixed."
)


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    what: str
    anchor_1: str
    anchor_3: str
    anchor_5: str


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        key="factual_accuracy",
        label="Factual accuracy",
        what=(
            "Are the technical and regulatory claims correct? Judge the claims actually made, "
            "not the topic. You must quote the single span you consider most decisive."
        ),
        anchor_1=(
            "Contains at least one confidently stated claim that is plainly wrong (a wrong "
            "regulator, a wrong statute, an invented figure, a wrong governing equation), or "
            "the answer is wrong in its central assertion."
        ),
        anchor_3=(
            "Broadly correct in direction but with real errors or vagueness: a right idea with "
            "a wrong number or wrong attribution, or hedged text that avoids committing to any "
            "checkable claim."
        ),
        anchor_5=(
            "Every checkable claim is correct, including names of bodies, statutes, "
            "correlations and figures, and uncertainty is stated where it genuinely exists."
        ),
    ),
    Dimension(
        key="key_fact_coverage",
        label="Key-fact coverage",
        what=(
            "How much of the supplied key-facts checklist does the answer actually contain? "
            "Score only against the checklist, not against what you would have written."
        ),
        anchor_1="Fewer than about one fifth of the checklist items appear, in any wording.",
        anchor_3="Roughly half the checklist items appear, or many appear only in weakened form.",
        anchor_5="Nearly all checklist items appear, clearly stated, in any reasonable wording.",
    ),
    Dimension(
        key="nigerian_specificity",
        label="Nigerian specificity",
        what=(
            "Does the answer engage with the Nigerian context concretely: named regulators, "
            "the PIA 2021 and its mechanisms, named fields, terminals, basins, operators and "
            "local practice? For a question with no Nigerian content, score how well the answer "
            "avoids fake localisation while staying useful."
        ),
        anchor_1=(
            "Generic global petroleum answer with no Nigerian content where Nigerian content "
            "was required, or Nigerian names dropped in decoratively with no substance."
        ),
        anchor_3=(
            "Some correct Nigerian references but shallow: names the right body or Act without "
            "the mechanism, or gives Nigerian context that is generic enough to fit any country."
        ),
        anchor_5=(
            "Specific, correct and load-bearing Nigerian detail: the right regulator with its "
            "actual remit, the right statutory mechanism, named fields, terminals or basins used "
            "to make the point."
        ),
    ),
    Dimension(
        key="engineering_rigour",
        label="Engineering rigour",
        what=(
            "Units carried through, assumptions stated, governing equations given and used "
            "correctly, sensible significant figures, and limitations flagged."
        ),
        anchor_1="No equations or units where they were needed, or arithmetic and unit handling that is wrong.",
        anchor_3=(
            "Gives an equation or a number but skips steps, drops units somewhere, leaves "
            "assumptions implicit, or reports false precision."
        ),
        anchor_5=(
            "States assumptions, gives the governing equation, carries units through every step, "
            "reports a result with sensible significant figures, and notes the validity limits."
        ),
    ),
    Dimension(
        key="citation_quality",
        label="Citation quality",
        what=(
            "Are sources named and are they the right kind? Official and regulatory sources "
            "(NUPRC, NMDPRA, NNPC, NEITI, NCDMB, the PIA text) outrank news. An ungrounded "
            "answer that says plainly it is ungrounded is better than one with invented "
            "citations. Score 1 for any fabricated or unverifiable citation."
        ),
        anchor_1=(
            "Fabricated citations, invented section numbers, invented report titles, or "
            "confident sourcing that cannot be what it claims to be."
        ),
        anchor_3=(
            "Vague attribution such as 'industry sources' or 'the regulations', or correct "
            "sourcing for only part of what needed it, or no sourcing at all without saying so."
        ),
        anchor_5=(
            "Specific, checkable, appropriate sources attached to the claims that need them, "
            "with official sources preferred, or an explicit statement that the answer is "
            "ungrounded general knowledge where no source was available."
        ),
    ),
    Dimension(
        key="hallucination",
        label="Freedom from hallucination (inverted)",
        what=(
            "Higher is better: 5 means nothing invented. Count invented statutes, sections, "
            "bodies, projects, field names, figures presented as fact, and accepting a false "
            "premise in the question instead of correcting it."
        ),
        anchor_1=(
            "Substantial invention: fabricated Acts, sections, bodies, projects or figures, or "
            "the answer plays along with a false premise and elaborates on something that does "
            "not exist."
        ),
        anchor_3=(
            "One clear invented or unverifiable specific, or the false premise is neither "
            "accepted nor corrected, or plausible-sounding numbers with no basis."
        ),
        anchor_5=(
            "Nothing invented. Unknowns are named as unknowns and a false premise in the "
            "question is explicitly corrected."
        ),
    ),
)

DIM_KEYS: tuple[str, ...] = tuple(d.key for d in DIMENSIONS)

# Sanity thresholds. Deliberately conservative: they are meant to fire.
MIN_STDEV = 0.35            # per-dimension standard deviation below this is suspect
MIN_DISTINCT = 3            # distinct score values expected per dimension
MIN_N_FOR_VARIANCE = 10     # do not judge variance on a handful of questions
MAX_JUDGE_FAIL_RATE = 0.10  # unparseable judge replies
MIN_QUOTE_VERIFY_RATE = 0.50
MIN_CONTROL_GAP = 0.75      # real answers must beat the canned control by this much
FLAT_DELTA = 0.05           # per-dimension mean delta below this is "no signal"
MAX_TIE_RATE = 0.60
POSITION_BIAS_BAND = (0.35, 0.65)


# --------------------------------------------------------------------------- #
# Question loading
# --------------------------------------------------------------------------- #

REQUIRED_FIELDS = ("id", "category", "difficulty", "question", "key_facts", "requires_retrieval")


def load_questions(
    path: Path,
    *,
    limit: int | None = None,
    categories: Sequence[str] | None = None,
    difficulties: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Read questions.jsonl, validate required fields, apply filters."""
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            missing = [k for k in REQUIRED_FIELDS if k not in obj]
            if missing:
                raise SystemExit(f"{path}:{lineno}: missing field(s): {', '.join(missing)}")
            out.append(obj)

    if ids:
        wanted = {i.strip() for i in ids}
        out = [q for q in out if q["id"] in wanted]
    if categories:
        cats = {c.strip() for c in categories}
        out = [q for q in out if q["category"] in cats]
    if difficulties:
        diffs = {d.strip() for d in difficulties}
        out = [q for q in out if q["difficulty"] in diffs]
    if limit is not None and limit > 0:
        out = out[:limit]
    if not out:
        raise SystemExit("No questions selected. Check --category, --difficulty, --id and --limit.")
    return out


# --------------------------------------------------------------------------- #
# Targets: the deployed app, or any OpenAI-compatible endpoint
# --------------------------------------------------------------------------- #

@dataclass
class TargetSpec:
    """One system under test."""
    name: str
    kind: str                     # "app" or "openai"
    url: str
    model: str = ""
    key_env: str = ""             # env var NAME holding the API key (never the key itself)
    token_env: str = ""           # env var NAME holding the app access token
    reasoning: bool = False       # app only: ask for a <think> trace
    units: str = "field"          # app only: "field" or "si"
    temperature: float = 0.2
    max_tokens: int = 1400
    system_prompt: str = EVAL_SYSTEM_PROMPT

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "url": self.url,
            "model": self.model,
            "key_env": self.key_env or None,
            "reasoning": self.reasoning,
            "units": self.units,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt_sha_prefix": _short_hash(self.system_prompt),
        }


@dataclass
class Answer:
    text: str
    raw_text: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    latency_s: float = 0.0
    error: str = ""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove Qwen3-style <think> blocks so the judge scores the final answer."""
    cleaned = _THINK_RE.sub("", text or "")
    cleaned = _OPEN_THINK_RE.sub("", cleaned)
    return cleaned.strip()


def _short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


async def _sleep_backoff(attempt: int, base: float = 1.5, cap: float = 30.0) -> None:
    delay = min(cap, base * (2 ** attempt)) * (0.6 + 0.8 * random.random())
    await asyncio.sleep(delay)


class RetryableError(RuntimeError):
    pass


def _classify_status(status: int) -> None:
    if status == 429 or status >= 500:
        raise RetryableError(f"HTTP {status}")
    if status >= 400:
        raise RuntimeError(f"HTTP {status}")


async def answer_from_app(
    client: httpx.AsyncClient, spec: TargetSpec, question: str, *, retries: int, timeout: float
) -> Answer:
    """POST to the app's /chat SSE endpoint and reassemble the streamed answer."""
    url = spec.url.rstrip("/")
    if not url.endswith("/chat"):
        url = url + "/chat"
    payload = {
        "message": question,
        "session_id": f"eval-{uuid.uuid4().hex[:12]}",
        "user_id": "eval-harness",
        "history": [],
        "reasoning": spec.reasoning,
        "units": spec.units,
        "continuation": False,
    }
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    token = os.environ.get(spec.token_env, "") if spec.token_env else ""
    if token:
        headers["X-Access-Token"] = token

    last_error = ""
    for attempt in range(retries + 1):
        parts: list[str] = []
        sources: list[dict[str, Any]] = []
        tools: list[dict[str, Any]] = []
        stream_error = ""
        started = time.time()
        try:
            async with client.stream(
                "POST", url, json=payload, headers=headers, timeout=timeout
            ) as resp:
                _classify_status(resp.status_code)
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if not chunk:
                        continue
                    try:
                        evt = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    kind = evt.get("type")
                    if kind == "token":
                        parts.append(evt.get("t", ""))
                    elif kind == "meta":
                        sources = list(evt.get("sources") or [])
                    elif kind == "tool":
                        tools.append({"name": evt.get("name"), "args": evt.get("args"),
                                      "result": evt.get("result")})
                    elif kind == "error":
                        stream_error = str(evt.get("message", "stream error"))
            raw = "".join(parts)
            if stream_error and not raw:
                raise RetryableError(stream_error)
            return Answer(
                text=strip_reasoning(raw),
                raw_text=raw,
                sources=sources,
                tools=tools,
                latency_s=round(time.time() - started, 2),
                error=stream_error,
            )
        except (RetryableError, httpx.TransportError, httpx.ReadTimeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await _sleep_backoff(attempt)
                continue
        except Exception as exc:  # non-retryable
            return Answer(text="", latency_s=round(time.time() - started, 2),
                          error=f"{type(exc).__name__}: {exc}")
    return Answer(text="", error=f"gave up after {retries + 1} attempts: {last_error}")


async def answer_from_openai(
    client: httpx.AsyncClient, spec: TargetSpec, question: str, *, retries: int, timeout: float
) -> Answer:
    """Call any OpenAI-compatible /chat/completions endpoint (Ollama, vLLM, hosted)."""
    url = spec.url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    messages: list[dict[str, str]] = []
    if spec.system_prompt:
        messages.append({"role": "system", "content": spec.system_prompt})
    messages.append({"role": "user", "content": question})
    payload: dict[str, Any] = {
        "model": spec.model,
        "messages": messages,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(spec.key_env, "") if spec.key_env else ""
    if key:
        headers["Authorization"] = f"Bearer {key}"

    last_error = ""
    for attempt in range(retries + 1):
        started = time.time()
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
            _classify_status(resp.status_code)
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if not content and message.get("reasoning_content"):
                content = message["reasoning_content"]
            return Answer(
                text=strip_reasoning(content),
                raw_text=content,
                latency_s=round(time.time() - started, 2),
            )
        except (RetryableError, httpx.TransportError, httpx.ReadTimeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await _sleep_backoff(attempt)
                continue
        except Exception as exc:
            return Answer(text="", latency_s=round(time.time() - started, 2),
                          error=f"{type(exc).__name__}: {exc}")
    return Answer(text="", error=f"gave up after {retries + 1} attempts: {last_error}")


async def get_answer(
    client: httpx.AsyncClient, spec: TargetSpec, question: str, *, retries: int, timeout: float
) -> Answer:
    if spec.kind == "app":
        return await answer_from_app(client, spec, question, retries=retries, timeout=timeout)
    return await answer_from_openai(client, spec, question, retries=retries, timeout=timeout)


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #

@dataclass
class JudgeConfig:
    base_url: str
    model: str
    key_env: str
    temperature: float = 0.0
    max_tokens: int = 1600
    retries: int = 3
    timeout: float = 180.0
    thinking: str = "off"   # nemotron models accept a "detailed thinking on/off" system line

    def describe(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "key_env": self.key_env,
            "temperature": self.temperature,
            "thinking": self.thinking,
        }


def rubric_block() -> str:
    lines: list[str] = []
    for dim in DIMENSIONS:
        lines.append(f"### {dim.key} ({dim.label})")
        lines.append(dim.what)
        lines.append(f"  1 = {dim.anchor_1}")
        lines.append(f"  3 = {dim.anchor_3}")
        lines.append(f"  5 = {dim.anchor_5}")
        lines.append("  2 and 4 are available for cases that sit between the anchors.")
        lines.append("")
    return "\n".join(lines)


JUDGE_SYSTEM = (
    "You are a strict examiner grading answers from a petroleum-engineering assistant that "
    "covers the Nigerian oil and gas sector. You are grading, not helping and not rewriting.\n\n"
    "Rules you must follow:\n"
    "1. Use the full 1 to 5 range. A score of 3 is a real judgement about a middling answer, "
    "not a safe default. If you find yourself giving 3 to everything, you are not grading.\n"
    "2. Score each dimension independently. A fluent answer with wrong facts scores high on "
    "nothing except possibly rigour of presentation.\n"
    "3. Judge only what the answer says. Do not credit what it implies, gestures at, or could "
    "have meant.\n"
    "4. Check the answer against the key-facts checklist supplied with the question. That "
    "checklist is the ground truth for this task.\n"
    "5. Where the question contains a false premise, the checklist will say so. An answer that "
    "corrects the premise scores high on hallucination; one that plays along scores 1.\n"
    "6. Return one JSON object and nothing else. No preamble, no markdown fence, no commentary."
)


def build_judge_prompt(question: dict[str, Any], answer: str) -> str:
    facts = question.get("key_facts") or []
    checklist = "\n".join(f"  [{i}] {f}" for i, f in enumerate(facts))
    numeric_note = ""
    if "numeric_answer" in question:
        numeric_note = (
            "\nNote: the numeric result for this question is graded separately and "
            "deterministically in code. Judge the method, the units, the assumptions and the "
            "reasoning, and do not spend your score on re-checking the arithmetic.\n"
        )
    trap_note = ""
    if question.get("trap"):
        trap_note = (
            "\nNote: this question contains a deliberately false or confused premise. The "
            "checklist states what a correct response does about it.\n"
        )
    schema = {
        "factual_accuracy": {"score": 1, "justification": "one or two sentences",
                             "quote": "verbatim span copied from the answer"},
        "key_fact_coverage": {"score": 1, "justification": "one or two sentences"},
        "nigerian_specificity": {"score": 1, "justification": "one or two sentences"},
        "engineering_rigour": {"score": 1, "justification": "one or two sentences"},
        "citation_quality": {"score": 1, "justification": "one or two sentences"},
        "hallucination": {"score": 1, "justification": "one or two sentences"},
        "key_facts_found": [0, 2],
        "invented_specifics": ["any statute, section, figure or name that appears fabricated"],
        "one_line_verdict": "a single sentence",
    }
    return (
        f"# Rubric\n{rubric_block()}\n"
        f"# Question ({question.get('category')}, {question.get('difficulty')})\n"
        f"{question.get('question')}\n"
        f"{trap_note}{numeric_note}\n"
        f"# Key-facts checklist (ground truth)\n{checklist}\n\n"
        f"# Answer under evaluation\n<<<ANSWER_START>>>\n{answer}\n<<<ANSWER_END>>>\n\n"
        "# Your task\n"
        "Score every dimension from 1 to 5 with a short justification. For factual_accuracy you "
        "must also copy, verbatim, the span of the answer that most drove your score; copy it "
        "exactly, do not paraphrase, and keep it under 30 words. In key_facts_found list the "
        "indices of the checklist items that the answer actually contains. In "
        "invented_specifics list anything that looks fabricated, or an empty list.\n\n"
        "Reply with exactly this JSON shape and nothing else:\n"
        f"{json.dumps(schema, indent=2)}"
    )


PAIRWISE_SYSTEM = (
    "You are a strict examiner comparing two answers to the same petroleum-engineering "
    "question about the Nigerian oil and gas sector. Decide which answer is better against the "
    "key-facts checklist supplied with the question.\n\n"
    "Rules:\n"
    "1. Judge accuracy and checklist coverage first, then Nigerian specificity, then "
    "engineering rigour, then sourcing. Length, confidence and polish are not quality.\n"
    "2. An answer that invents a statute, a section, a figure or a project loses, even if it "
    "reads better.\n"
    "3. Return 'tie' only when the two answers are genuinely equal in quality after that check. "
    "A tie is a failure to discriminate and should be rare.\n"
    "4. The order of the two answers is randomised and carries no information.\n"
    "5. Return one JSON object and nothing else."
)


def build_pairwise_prompt(question: dict[str, Any], first: str, second: str) -> str:
    facts = question.get("key_facts") or []
    checklist = "\n".join(f"  [{i}] {f}" for i, f in enumerate(facts))
    trap_note = ""
    if question.get("trap"):
        trap_note = (
            "\nNote: this question contains a deliberately false or confused premise. An answer "
            "that corrects it beats one that plays along, however well written.\n"
        )
    schema = {
        "winner": "1 or 2 or tie",
        "margin": "slight or clear or decisive",
        "reason": "two sentences at most, citing the decisive difference",
        "dimension_winners": {k: "1 or 2 or tie" for k in DIM_KEYS},
    }
    return (
        f"# Question ({question.get('category')}, {question.get('difficulty')})\n"
        f"{question.get('question')}\n{trap_note}\n"
        f"# Key-facts checklist (ground truth)\n{checklist}\n\n"
        f"# Response 1\n<<<R1_START>>>\n{first}\n<<<R1_END>>>\n\n"
        f"# Response 2\n<<<R2_START>>>\n{second}\n<<<R2_END>>>\n\n"
        "# Your task\nPick the better response. Reply with exactly this JSON shape and nothing "
        f"else:\n{json.dumps(schema, indent=2)}"
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a judge reply, tolerating fences and prose."""
    body = strip_reasoning(text or "")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.DOTALL)
    candidates: list[str] = []
    if fenced:
        candidates.append(fenced.group(1))
    start, end = body.find("{"), body.rfind("}")
    if start != -1 and end > start:
        candidates.append(body[start:end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


async def call_judge(
    client: httpx.AsyncClient, cfg: JudgeConfig, system: str, user: str
) -> tuple[dict[str, Any] | None, str]:
    """Call the judge and return (parsed_json, raw_text). Retries on transport and on
    unparseable replies, with an explicit corrective nudge on the second try."""
    url = cfg.base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    key = os.environ.get(cfg.key_env, "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    system_text = system
    if cfg.thinking in ("on", "off") and "nemotron" in cfg.model.lower():
        system_text = f"detailed thinking {cfg.thinking}\n\n{system}"

    raw = ""
    for attempt in range(cfg.retries + 1):
        prompt = user if attempt == 0 else (
            user + "\n\nYour previous reply was not valid JSON. Reply with the JSON object only, "
            "starting with { and ending with }, with no other text."
        )
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "stream": False,
        }
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=cfg.timeout)
            _classify_status(resp.status_code)
            data = resp.json()
            raw = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            parsed = _extract_json_object(raw)
            if parsed is not None:
                return parsed, raw
        except (RetryableError, httpx.TransportError, httpx.ReadTimeout) as exc:
            raw = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        if attempt < cfg.retries:
            await _sleep_backoff(attempt)
    return None, raw


def _coerce_score(value: Any) -> int | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not (1.0 <= num <= 5.0):
        return None
    return int(round(num))


def _normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def validate_judgement(parsed: dict[str, Any], answer: str, n_facts: int) -> dict[str, Any]:
    """Turn a raw judge reply into a validated record. Never substitutes a default score:
    a missing or malformed dimension is recorded as missing, because silently writing a 3
    is exactly how the previous evaluation produced its flat result."""
    scores: dict[str, int] = {}
    justifications: dict[str, str] = {}
    missing: list[str] = []
    for key in DIM_KEYS:
        block = parsed.get(key)
        if isinstance(block, dict):
            score = _coerce_score(block.get("score"))
            justifications[key] = str(block.get("justification", ""))[:600]
        else:
            score = _coerce_score(block)
            justifications[key] = ""
        if score is None:
            missing.append(key)
        else:
            scores[key] = score

    quote = ""
    fa = parsed.get("factual_accuracy")
    if isinstance(fa, dict):
        quote = str(fa.get("quote", ""))[:400]
    quote_verified = bool(quote) and _normalise_ws(quote) in _normalise_ws(answer)

    found_raw = parsed.get("key_facts_found")
    found: list[int] = []
    if isinstance(found_raw, list):
        for item in found_raw:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n_facts and idx not in found:
                found.append(idx)
    judge_coverage = (len(found) / n_facts) if n_facts else 0.0

    invented = parsed.get("invented_specifics")
    invented_list = [str(x)[:200] for x in invented] if isinstance(invented, list) else []

    return {
        "ok": not missing,
        "missing_dimensions": missing,
        "scores": scores,
        "justifications": justifications,
        "factual_quote": quote,
        "quote_verified": quote_verified,
        "judge_key_facts_found": found,
        "judge_key_fact_coverage": round(judge_coverage, 4),
        "invented_specifics": invented_list,
        "one_line_verdict": str(parsed.get("one_line_verdict", ""))[:400],
    }


# --------------------------------------------------------------------------- #
# Programmatic graders: key-fact coverage and numeric answers
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset("""
about above after against also always among another answer because been before being below
between both cannot come complete could does doing done down during each either else even
ever every from further generally give given good greater have having here high higher hers
himself into itself just like made make many more most much must note noted notes only other
others ought over own rather same should since some stated states such than that the their
them themselves then there these they this those through thus together typically under until
upon used using very well were what when where which while whom will with within without
would your yours
""".split())

_NUM_TOKEN_RE = re.compile(r"^\d[\d.,]*$")


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9.\-]*", (text or "").lower())


def _fact_tokens(fact: str) -> tuple[set[str], set[str]]:
    """Return (content tokens, numeric tokens) that a covering answer should contain."""
    acronyms = {a.lower() for a in re.findall(r"\b[A-Z]{2,}\b", fact or "")}
    content: set[str] = set()
    numbers: set[str] = set()
    for tok in _tokenise(fact):
        tok = tok.strip(".-")
        if not tok:
            continue
        if _NUM_TOKEN_RE.match(tok):
            numbers.add(tok.replace(",", ""))
        elif len(tok) >= 4 and tok not in _STOPWORDS:
            content.add(tok)
    content |= acronyms
    return content, numbers


def fact_coverage(answer: str, key_facts: Sequence[str], threshold: float = 0.6) -> dict[str, Any]:
    """Lexical proxy for key-fact coverage, computed in code.

    This is deliberately crude and it is not ground truth. Its job is to be an
    independent signal: if the judge's coverage score never moves while this does
    (or the reverse), the judge is not reading the checklist.
    """
    ans_tokens = set()
    for tok in _tokenise(answer):
        tok = tok.strip(".-")
        if tok:
            ans_tokens.add(tok.replace(",", "") if _NUM_TOKEN_RE.match(tok) else tok)
    per_fact: list[float] = []
    for fact in key_facts:
        content, numbers = _fact_tokens(fact)
        pool = list(content) + list(numbers) * 2   # numbers count double
        if not pool:
            per_fact.append(0.0)
            continue
        hits = sum(1 for tok in pool if tok in ans_tokens)
        per_fact.append(hits / len(pool))
    covered = [1 if s >= threshold else 0 for s in per_fact]
    n = len(per_fact) or 1
    return {
        "programmatic_key_fact_coverage": round(sum(covered) / n, 4),
        "programmatic_fact_scores": [round(s, 3) for s in per_fact],
        "programmatic_facts_covered": sum(covered),
        "programmatic_facts_total": len(per_fact),
    }


_NUMBER_RE = re.compile(
    r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"      # 1,200,000
    r"|[-+]?\d+\.\d+(?:[eE][-+]?\d+)?"        # 12.5, 1.2e6
    r"|[-+]?\d+(?:[eE][-+]?\d+)?"             # 6370
)
_SCI_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*[xX*]\s*10\s*\^?\s*([-+]?\d+)")


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for mantissa, exponent in _SCI_RE.findall(text or ""):
        try:
            values.append(float(mantissa) * (10.0 ** int(exponent)))
        except ValueError:
            continue
    for raw in _NUMBER_RE.findall(text or ""):
        try:
            values.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


def grade_numeric(answer: str, target: float, rel_tolerance: float) -> dict[str, Any]:
    """Deterministic numeric grading. The judge is never asked to check arithmetic.

    Accepts the target expressed in a shifted magnitude (thousands, millions,
    billions) or as a fraction rather than a percentage, since a correct answer may
    legitimately write 223 MMSTB or 0.218 instead of the raw figure. The matched
    scale is recorded so a reviewer can see what was accepted.
    """
    scales = {
        "as_given": 1.0, "thousands": 1e-3, "millions": 1e-6, "billions": 1e-9,
        "fraction_of_percent": 1e-2, "percent_of_fraction": 1e2,
    }
    found = extract_numbers(answer)
    best: dict[str, Any] = {
        "numeric_correct": False, "numeric_target": target, "numeric_matched": None,
        "numeric_matched_scale": None, "numeric_rel_error": None,
        "numeric_candidates_examined": len(found),
    }
    best_err = math.inf
    for label, factor in scales.items():
        scaled = target * factor
        if scaled == 0:
            continue
        for value in found:
            err = abs(value - scaled) / abs(scaled)
            if err < best_err:
                best_err = err
                best["numeric_matched"] = value
                best["numeric_matched_scale"] = label
                best["numeric_rel_error"] = round(err, 6)
            if err <= rel_tolerance:
                best.update({
                    "numeric_correct": True, "numeric_matched": value,
                    "numeric_matched_scale": label, "numeric_rel_error": round(err, 6),
                })
                return best
    return best


# --------------------------------------------------------------------------- #
# Record sink (resumable output)
# --------------------------------------------------------------------------- #

class RecordSink:
    """Append-only JSONL sink so a crashed run keeps everything already scored."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def load(self) -> dict[str, dict[str, Any]]:
        done: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return done
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = rec.get("_key")
                if key:
                    done[key] = rec
        return done


# --------------------------------------------------------------------------- #
# Single-system evaluation
# --------------------------------------------------------------------------- #

async def evaluate_one(
    client: httpx.AsyncClient,
    spec: TargetSpec,
    judge_cfg: JudgeConfig,
    question: dict[str, Any],
    *,
    answer_retries: int,
    answer_timeout: float,
    fact_threshold: float,
    control_probe: bool,
) -> dict[str, Any]:
    key_facts = question.get("key_facts") or []
    answer = await get_answer(
        client, spec, question["question"], retries=answer_retries, timeout=answer_timeout
    )

    record: dict[str, Any] = {
        "_key": f"single|{spec.name}|{question['id']}",
        "mode": "single",
        "system": spec.name,
        "question_id": question["id"],
        "category": question["category"],
        "difficulty": question["difficulty"],
        "requires_retrieval": bool(question.get("requires_retrieval")),
        "trap": bool(question.get("trap")),
        "question": question["question"],
        "answer": answer.text,
        "answer_chars": len(answer.text),
        "answer_error": answer.error,
        "latency_s": answer.latency_s,
        "n_sources": len(answer.sources),
        "sources": answer.sources[:20],
        "tools": answer.tools,
        "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    record.update(fact_coverage(answer.text, key_facts, threshold=fact_threshold))
    if "numeric_answer" in question:
        record.update(grade_numeric(
            answer.text, float(question["numeric_answer"]),
            float(question.get("rel_tolerance", 0.02)),
        ))
        record["numeric_unit"] = question.get("unit", "")

    if not answer.text.strip():
        record.update({
            "judge_ok": False, "judge_error": answer.error or "empty answer",
            "scores": {}, "missing_dimensions": list(DIM_KEYS),
        })
        return record

    parsed, raw = await call_judge(
        client, judge_cfg, JUDGE_SYSTEM, build_judge_prompt(question, answer.text)
    )
    if parsed is None:
        record.update({
            "judge_ok": False, "judge_error": "judge reply was not parseable JSON",
            "judge_raw_tail": (raw or "")[-400:], "scores": {},
            "missing_dimensions": list(DIM_KEYS),
        })
        return record

    verdict = validate_judgement(parsed, answer.text, len(key_facts))
    record["judge_ok"] = verdict["ok"]
    record.update(verdict)
    if verdict["scores"]:
        record["mean_score"] = round(
            statistics.fmean(verdict["scores"].values()), 4
        )

    if control_probe:
        ctrl_parsed, _ = await call_judge(
            client, judge_cfg, JUDGE_SYSTEM, build_judge_prompt(question, CONTROL_ANSWER)
        )
        if ctrl_parsed is not None:
            ctrl = validate_judgement(ctrl_parsed, CONTROL_ANSWER, len(key_facts))
            record["control"] = {
                "scores": ctrl["scores"],
                "mean_score": (round(statistics.fmean(ctrl["scores"].values()), 4)
                               if ctrl["scores"] else None),
            }
        else:
            record["control"] = {"scores": {}, "mean_score": None, "error": "unparseable"}
    return record


# --------------------------------------------------------------------------- #
# Pairwise evaluation
# --------------------------------------------------------------------------- #

async def evaluate_pair(
    client: httpx.AsyncClient,
    spec_a: TargetSpec,
    spec_b: TargetSpec,
    judge_cfg: JudgeConfig,
    question: dict[str, Any],
    *,
    answer_retries: int,
    answer_timeout: float,
    seed: int,
) -> dict[str, Any]:
    ans_a, ans_b = await asyncio.gather(
        get_answer(client, spec_a, question["question"],
                   retries=answer_retries, timeout=answer_timeout),
        get_answer(client, spec_b, question["question"],
                   retries=answer_retries, timeout=answer_timeout),
    )

    # Deterministic per-question randomisation so a rerun reproduces the layout and
    # position bias can be measured rather than guessed at.
    rng = random.Random(f"{seed}:{question['id']}")
    a_first = rng.random() < 0.5
    first_text, second_text = (ans_a.text, ans_b.text) if a_first else (ans_b.text, ans_a.text)
    position_map = {"1": spec_a.name if a_first else spec_b.name,
                    "2": spec_b.name if a_first else spec_a.name}

    record: dict[str, Any] = {
        "_key": f"pairwise|{spec_a.name}|{spec_b.name}|{question['id']}",
        "mode": "pairwise",
        "system_a": spec_a.name,
        "system_b": spec_b.name,
        "question_id": question["id"],
        "category": question["category"],
        "difficulty": question["difficulty"],
        "requires_retrieval": bool(question.get("requires_retrieval")),
        "trap": bool(question.get("trap")),
        "a_in_position_1": a_first,
        "position_map": position_map,
        "answer_a_chars": len(ans_a.text),
        "answer_b_chars": len(ans_b.text),
        "answer_a": ans_a.text,
        "answer_b": ans_b.text,
        "answer_a_error": ans_a.error,
        "answer_b_error": ans_b.error,
        "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for label, ans in (("a", ans_a), ("b", ans_b)):
        cov = fact_coverage(ans.text, question.get("key_facts") or [])
        record[f"programmatic_key_fact_coverage_{label}"] = cov["programmatic_key_fact_coverage"]
        if "numeric_answer" in question:
            graded = grade_numeric(ans.text, float(question["numeric_answer"]),
                                   float(question.get("rel_tolerance", 0.02)))
            record[f"numeric_correct_{label}"] = graded["numeric_correct"]

    if not first_text.strip() or not second_text.strip():
        record.update({"judge_ok": False, "judge_error": "one or both answers were empty",
                       "winner": None})
        return record

    parsed, raw = await call_judge(
        client, judge_cfg, PAIRWISE_SYSTEM,
        build_pairwise_prompt(question, first_text, second_text),
    )
    if parsed is None:
        record.update({"judge_ok": False, "judge_error": "judge reply was not parseable JSON",
                       "judge_raw_tail": (raw or "")[-400:], "winner": None})
        return record

    raw_winner = str(parsed.get("winner", "")).strip().lower()
    if raw_winner in ("1", "response 1", "first"):
        winner = position_map["1"]
    elif raw_winner in ("2", "response 2", "second"):
        winner = position_map["2"]
    elif raw_winner in ("tie", "draw", "equal"):
        winner = "tie"
    else:
        winner = None

    dim_winners: dict[str, str] = {}
    raw_dims = parsed.get("dimension_winners")
    if isinstance(raw_dims, dict):
        for key in DIM_KEYS:
            val = str(raw_dims.get(key, "")).strip().lower()
            if val in ("1", "2"):
                dim_winners[key] = position_map[val]
            elif val in ("tie", "draw", "equal"):
                dim_winners[key] = "tie"

    record.update({
        "judge_ok": winner is not None,
        "winner": winner,
        "winner_position": raw_winner if raw_winner in ("1", "2") else None,
        "margin": str(parsed.get("margin", ""))[:40],
        "reason": str(parsed.get("reason", ""))[:800],
        "dimension_winners": dim_winners,
    })
    return record


# --------------------------------------------------------------------------- #
# Aggregation and the sanity check
# --------------------------------------------------------------------------- #

def _stats(values: Sequence[float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "distinct": 0, "min": None, "max": None,
                "histogram": {}}
    hist: dict[str, int] = {}
    for v in vals:
        hist[str(int(v)) if float(v).is_integer() else f"{v:.2f}"] = \
            hist.get(str(int(v)) if float(v).is_integer() else f"{v:.2f}", 0) + 1
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 4),
        "sd": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "distinct": len(set(vals)),
        "min": min(vals),
        "max": max(vals),
        "histogram": dict(sorted(hist.items())),
    }


def aggregate_single(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    judged = [r for r in records if r.get("judge_ok") and r.get("scores")]
    by_dim = {k: _stats([r["scores"][k] for r in judged if k in r["scores"]]) for k in DIM_KEYS}

    def _group(field_name: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for rec in judged:
            groups.setdefault(str(rec.get(field_name)), []).append(rec)
        out: dict[str, Any] = {}
        for name, recs in sorted(groups.items()):
            out[name] = {
                "n": len(recs),
                "mean_overall": round(statistics.fmean(
                    [r["mean_score"] for r in recs if r.get("mean_score") is not None]
                ), 4) if any(r.get("mean_score") is not None for r in recs) else None,
                "by_dimension": {
                    k: (round(statistics.fmean([r["scores"][k] for r in recs if k in r["scores"]]), 3)
                        if any(k in r["scores"] for r in recs) else None)
                    for k in DIM_KEYS
                },
            }
        return out

    numeric = [r for r in records if "numeric_correct" in r]
    traps = [r for r in judged if r.get("trap")]
    controls = [r["control"]["mean_score"] for r in records
                if isinstance(r.get("control"), dict) and r["control"].get("mean_score") is not None]

    overall_means = [r["mean_score"] for r in judged if r.get("mean_score") is not None]
    prog_cov = [r["programmatic_key_fact_coverage"] for r in records
                if r.get("programmatic_key_fact_coverage") is not None]
    judge_cov = [r["judge_key_fact_coverage"] for r in judged
                 if r.get("judge_key_fact_coverage") is not None]

    return {
        "n_questions": len(records),
        "n_judged": len(judged),
        "n_judge_failures": len(records) - len(judged),
        "judge_failure_rate": round((len(records) - len(judged)) / max(1, len(records)), 4),
        "n_answer_errors": sum(1 for r in records if r.get("answer_error")),
        "overall_mean": round(statistics.fmean(overall_means), 4) if overall_means else None,
        "overall_stats": _stats(overall_means),
        "by_dimension": by_dim,
        "by_category": _group("category"),
        "by_difficulty": _group("difficulty"),
        "by_requires_retrieval": _group("requires_retrieval"),
        "key_fact_coverage": {
            "programmatic_mean": round(statistics.fmean(prog_cov), 4) if prog_cov else None,
            "programmatic_stats": _stats(prog_cov),
            "judge_mean_normalised": round(statistics.fmean(judge_cov), 4) if judge_cov else None,
            "judge_dimension_mean": by_dim["key_fact_coverage"]["mean"],
        },
        "numeric": {
            "n": len(numeric),
            "n_correct": sum(1 for r in numeric if r.get("numeric_correct")),
            "accuracy": round(sum(1 for r in numeric if r.get("numeric_correct")) / len(numeric), 4)
            if numeric else None,
        },
        "traps": {
            "n": len(traps),
            "mean_hallucination": round(statistics.fmean(
                [r["scores"]["hallucination"] for r in traps if "hallucination" in r["scores"]]
            ), 4) if traps else None,
            "mean_overall": round(statistics.fmean(
                [r["mean_score"] for r in traps if r.get("mean_score") is not None]
            ), 4) if traps else None,
        },
        "quote_verification_rate": round(
            sum(1 for r in judged if r.get("quote_verified")) / len(judged), 4
        ) if judged else None,
        "control_probe": {
            "n": len(controls),
            "mean": round(statistics.fmean(controls), 4) if controls else None,
            "gap_to_real": (round(statistics.fmean(overall_means) - statistics.fmean(controls), 4)
                            if controls and overall_means else None),
        },
        "mean_latency_s": round(statistics.fmean(
            [r["latency_s"] for r in records if r.get("latency_s")]
        ), 2) if any(r.get("latency_s") for r in records) else None,
    }


def sanity_check_single(agg: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    notes: list[str] = []
    n = agg["n_judged"]

    if agg["judge_failure_rate"] > MAX_JUDGE_FAIL_RATE:
        warnings.append(
            f"Judge failed to return usable JSON on {agg['judge_failure_rate'] * 100:.0f} percent "
            f"of questions (threshold {MAX_JUDGE_FAIL_RATE * 100:.0f} percent). The aggregate is "
            "built on a biased subset."
        )
    if n < MIN_N_FOR_VARIANCE:
        notes.append(
            f"Only {n} judged questions: variance checks are not meaningful below "
            f"{MIN_N_FOR_VARIANCE}. Treat this as a smoke test, not an evaluation."
        )
    else:
        for key, st in agg["by_dimension"].items():
            if st["n"] == 0:
                warnings.append(f"Dimension '{key}' has no scores at all.")
                continue
            if st["sd"] is not None and st["sd"] < MIN_STDEV:
                warnings.append(
                    f"Dimension '{key}' has near-zero variance (sd = {st['sd']:.3f}, mean = "
                    f"{st['mean']:.2f}, {st['distinct']} distinct values). {DEGENERATE_NOTE}"
                )
            elif st["distinct"] < MIN_DISTINCT:
                warnings.append(
                    f"Dimension '{key}' used only {st['distinct']} distinct score value(s) "
                    f"across {st['n']} questions. {DEGENERATE_NOTE}"
                )
        means = [st["mean"] for st in agg["by_dimension"].values() if st["mean"] is not None]
        if len(means) >= 2 and (max(means) - min(means)) < 0.10:
            warnings.append(
                f"Every dimension landed within {max(means) - min(means):.3f} of every other "
                "dimension. Dimensions that never disagree are not being scored separately. "
                f"{DEGENERATE_NOTE}"
            )

    qvr = agg.get("quote_verification_rate")
    if qvr is not None and n >= MIN_N_FOR_VARIANCE and qvr < MIN_QUOTE_VERIFY_RATE:
        warnings.append(
            f"Only {qvr * 100:.0f} percent of factual_accuracy quotes were found verbatim in the "
            "answer. The judge is paraphrasing or inventing its evidence, so its justifications "
            "cannot be trusted."
        )

    ctrl = agg.get("control_probe") or {}
    if ctrl.get("n"):
        gap = ctrl.get("gap_to_real")
        if gap is None:
            notes.append(
                "A control probe ran but no real answer was judged successfully, so the two "
                "cannot be compared."
            )
        elif gap < MIN_CONTROL_GAP:
            warnings.append(
                f"The canned control answer scored {ctrl['mean']:.2f} against {agg['overall_mean']:.2f} "
                f"for real answers, a gap of only {gap:.2f} (expected at least {MIN_CONTROL_GAP}). "
                f"The judge cannot tell a real answer from filler. {DEGENERATE_NOTE}"
            )
        else:
            notes.append(
                f"Control probe passed: filler answer scored {ctrl['mean']:.2f} against "
                f"{agg['overall_mean']:.2f} for real answers (gap {gap:.2f})."
            )
    else:
        notes.append("No control probe was run. Use --control-probe to test the judge itself.")

    cov = agg.get("key_fact_coverage") or {}
    prog, judged_cov = cov.get("programmatic_mean"), cov.get("judge_mean_normalised")
    if prog is not None and judged_cov is not None:
        gap = abs(prog - judged_cov)
        notes.append(
            f"Key-fact coverage: programmatic {prog:.2f} against judge {judged_cov:.2f} "
            f"(difference {gap:.2f}). The lexical measure is a proxy, so some gap is normal; a "
            "judge value that is flat while the programmatic one moves is not."
        )
        prog_sd = (cov.get("programmatic_stats") or {}).get("sd")
        judge_sd = agg["by_dimension"]["key_fact_coverage"]["sd"]
        if (n >= MIN_N_FOR_VARIANCE and prog_sd is not None and judge_sd is not None
                and prog_sd > 0.10 and judge_sd < MIN_STDEV):
            warnings.append(
                "Programmatic key-fact coverage varies across questions but the judge's coverage "
                f"score does not (sd {judge_sd:.3f}). The judge is not reading the checklist. "
                f"{DEGENERATE_NOTE}"
            )

    return {"passed": not warnings, "warnings": warnings, "notes": notes}


def aggregate_pairwise(records: Sequence[dict[str, Any]], a_name: str, b_name: str) -> dict[str, Any]:
    judged = [r for r in records if r.get("judge_ok") and r.get("winner")]
    wins_a = sum(1 for r in judged if r["winner"] == a_name)
    wins_b = sum(1 for r in judged if r["winner"] == b_name)
    ties = sum(1 for r in judged if r["winner"] == "tie")
    n = len(judged) or 1
    pos1_wins = sum(1 for r in judged if r.get("winner_position") == "1")
    decided = sum(1 for r in judged if r.get("winner_position") in ("1", "2"))

    by_category: dict[str, dict[str, int]] = {}
    for rec in judged:
        bucket = by_category.setdefault(rec["category"], {"a": 0, "b": 0, "tie": 0})
        if rec["winner"] == a_name:
            bucket["a"] += 1
        elif rec["winner"] == b_name:
            bucket["b"] += 1
        else:
            bucket["tie"] += 1

    dim_tally: dict[str, dict[str, int]] = {k: {"a": 0, "b": 0, "tie": 0} for k in DIM_KEYS}
    for rec in judged:
        for key, winner in (rec.get("dimension_winners") or {}).items():
            if key not in dim_tally:
                continue
            if winner == a_name:
                dim_tally[key]["a"] += 1
            elif winner == b_name:
                dim_tally[key]["b"] += 1
            else:
                dim_tally[key]["tie"] += 1

    return {
        "system_a": a_name,
        "system_b": b_name,
        "n_questions": len(records),
        "n_judged": len(judged),
        "n_judge_failures": len(records) - len(judged),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": round(wins_a / n, 4),
        "win_rate_b": round(wins_b / n, 4),
        "tie_rate": round(ties / n, 4),
        "position_1_win_rate": round(pos1_wins / decided, 4) if decided else None,
        "n_decided": decided,
        "by_category": by_category,
        "by_dimension": dim_tally,
    }


def sanity_check_pairwise(agg: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    notes: list[str] = []
    n = agg["n_judged"]
    if n < MIN_N_FOR_VARIANCE:
        notes.append(f"Only {n} judged comparisons: rates below {MIN_N_FOR_VARIANCE} are noise.")
        return {"passed": True, "warnings": warnings, "notes": notes}

    if agg["tie_rate"] > MAX_TIE_RATE:
        warnings.append(
            f"Tie rate is {agg['tie_rate'] * 100:.0f} percent, above the "
            f"{MAX_TIE_RATE * 100:.0f} percent threshold. An all-ties result is the degenerate "
            f"signature this harness exists to catch. {DEGENERATE_NOTE}"
        )
    if agg["tie_rate"] >= 0.95:
        warnings.append("Almost every comparison was a tie. The judge is not discriminating at all.")

    low, high = POSITION_BIAS_BAND
    pos_rate = agg["position_1_win_rate"]
    if pos_rate is None or agg["n_decided"] < MIN_N_FOR_VARIANCE:
        notes.append(
            f"Only {agg['n_decided']} comparisons were decided either way, so position bias "
            "cannot be measured. That is itself a discrimination problem, not a clean result."
        )
    elif not (low <= pos_rate <= high):
        warnings.append(
            f"Position 1 won {pos_rate * 100:.0f} percent of decided comparisons, outside the "
            f"{low * 100:.0f} to {high * 100:.0f} percent band. The judge is scoring position, "
            "not content. Rerun with a different --seed to confirm."
        )
    else:
        notes.append(
            f"Position bias looks acceptable: position 1 won {pos_rate * 100:.0f} percent of "
            f"{agg['n_decided']} decided comparisons."
        )

    delta = abs(agg["win_rate_a"] - agg["win_rate_b"])
    if delta < 0.05 and agg["tie_rate"] < MAX_TIE_RATE:
        notes.append(
            f"The two systems are separated by {delta * 100:.1f} percentage points, which is "
            "within noise at this sample size. That is a real finding only if the tie rate and "
            "position bias checks passed."
        )
    return {"passed": not warnings, "warnings": warnings, "notes": notes}


def compare_runs(run_a: dict[str, Any], run_b: dict[str, Any]) -> dict[str, Any]:
    """Paired per-dimension delta between two completed single runs."""
    recs_a = {r["question_id"]: r for r in run_a["records"] if r.get("judge_ok")}
    recs_b = {r["question_id"]: r for r in run_b["records"] if r.get("judge_ok")}
    shared = sorted(set(recs_a) & set(recs_b))
    name_a = run_a["meta"]["target"]["name"]
    name_b = run_b["meta"]["target"]["name"]

    per_dim: dict[str, Any] = {}
    for key in DIM_KEYS:
        diffs = [recs_b[q]["scores"][key] - recs_a[q]["scores"][key]
                 for q in shared
                 if key in recs_a[q].get("scores", {}) and key in recs_b[q].get("scores", {})]
        if not diffs:
            per_dim[key] = {"n": 0}
            continue
        mean_diff = statistics.fmean(diffs)
        sd_diff = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
        t_stat = (mean_diff / (sd_diff / math.sqrt(len(diffs)))) if sd_diff > 0 else None
        per_dim[key] = {
            "n": len(diffs),
            "mean_delta_b_minus_a": round(mean_diff, 4),
            "sd_of_paired_delta": round(sd_diff, 4),
            "t_like": round(t_stat, 3) if t_stat is not None else None,
            "b_better": sum(1 for d in diffs if d > 0),
            "a_better": sum(1 for d in diffs if d < 0),
            "equal": sum(1 for d in diffs if d == 0),
        }

    overall_a = [recs_a[q]["mean_score"] for q in shared if recs_a[q].get("mean_score") is not None]
    overall_b = [recs_b[q]["mean_score"] for q in shared if recs_b[q].get("mean_score") is not None]
    overall_delta = (round(statistics.fmean(overall_b) - statistics.fmean(overall_a), 4)
                     if overall_a and overall_b else None)

    warnings: list[str] = []
    notes: list[str] = []
    if not shared:
        warnings.append("The two runs share no judged questions, so no comparison is possible.")
    else:
        flat = [k for k, v in per_dim.items()
                if v.get("n") and abs(v["mean_delta_b_minus_a"]) < FLAT_DELTA]
        if len(flat) == len([k for k in per_dim if per_dim[k].get("n")]):
            warnings.append(
                f"Every dimension moved by less than {FLAT_DELTA} between {name_a} and {name_b}. "
                f"{DEGENERATE_NOTE} Before publishing this as 'no difference', confirm both runs "
                "passed their own sanity checks and that the judge distinguished the control probe."
            )
        elif overall_delta is not None and abs(overall_delta) < 0.02:
            warnings.append(
                f"Overall mean delta is {overall_delta:+.4f}, which is indistinguishable from "
                f"zero. {DEGENERATE_NOTE}"
            )
        identical = sum(1 for q in shared
                        if recs_a[q].get("scores") == recs_b[q].get("scores"))
        if identical / len(shared) > 0.5:
            warnings.append(
                f"{identical} of {len(shared)} questions received byte-identical score vectors "
                "from both systems. That is what a judge ignoring the answer looks like."
            )
        notes.append(f"Compared {len(shared)} questions judged successfully in both runs.")

    return {
        "system_a": name_a,
        "system_b": name_b,
        "n_shared": len(shared),
        "overall_mean_a": round(statistics.fmean(overall_a), 4) if overall_a else None,
        "overall_mean_b": round(statistics.fmean(overall_b), 4) if overall_b else None,
        "overall_delta_b_minus_a": overall_delta,
        "by_dimension": per_dim,
        "sanity": {"passed": not warnings, "warnings": warnings, "notes": notes},
    }


# --------------------------------------------------------------------------- #
# Markdown reporting
# --------------------------------------------------------------------------- #

def _fmt(value: Any, spec: str = ".2f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


def _sanity_block(sanity: dict[str, Any]) -> list[str]:
    lines = ["## Sanity check", ""]
    if sanity["passed"]:
        lines.append("**VERDICT: PASS.** No degeneracy signature detected. Scores are usable, "
                     "subject to the notes below.")
    else:
        lines.append("**VERDICT: FAIL. Do not publish these scores.**")
    lines.append("")
    for warn in sanity["warnings"]:
        lines.append(f"- WARNING: {warn}")
    for note in sanity["notes"]:
        lines.append(f"- Note: {note}")
    lines.append("")
    return lines


def render_single_markdown(payload: dict[str, Any]) -> str:
    meta, agg, sanity = payload["meta"], payload["aggregate"], payload["sanity"]
    records = payload["records"]
    lines: list[str] = [
        f"# Naija-Petro evaluation: {meta['target']['name']}",
        "",
        f"- Run at: {meta['started_at']}",
        f"- Target: `{meta['target']['kind']}` at `{meta['target']['url']}` "
        f"model `{meta['target']['model'] or 'app default'}`",
        f"- Judge: `{meta['judge']['model']}` at `{meta['judge']['base_url']}` "
        f"(temperature {meta['judge']['temperature']})",
        f"- Questions: {agg['n_questions']} attempted, {agg['n_judged']} judged, "
        f"{agg['n_judge_failures']} judge failures, {agg['n_answer_errors']} answer errors",
        f"- Overall mean across dimensions: **{_fmt(agg['overall_mean'])}** / 5.0",
        f"- Mean answer latency: {_fmt(agg['mean_latency_s'])} s",
        "",
    ]
    lines += _sanity_block(sanity)

    lines += ["## Scores by dimension", "",
              "| Dimension | n | Mean | SD | Distinct | Min | Max | Histogram |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for dim in DIMENSIONS:
        st = agg["by_dimension"][dim.key]
        hist = ", ".join(f"{k}:{v}" for k, v in (st.get("histogram") or {}).items()) or "n/a"
        lines.append(
            f"| {dim.key} | {st['n']} | {_fmt(st['mean'])} | {_fmt(st['sd'], '.3f')} | "
            f"{st['distinct']} | {_fmt(st['min'], '.0f')} | {_fmt(st['max'], '.0f')} | {hist} |"
        )
    lines += ["",
              "SD and the distinct-value count are the anti-degeneracy columns. A dimension with "
              f"SD below {MIN_STDEV} or fewer than {MIN_DISTINCT} distinct values is not measuring "
              "anything.", ""]

    lines += ["## Scores by category", "",
              "| Category | n | Overall | " + " | ".join(DIM_KEYS) + " |",
              "|---|---:|---:|" + "---:|" * len(DIM_KEYS)]
    for cat, st in agg["by_category"].items():
        cells = " | ".join(_fmt(st["by_dimension"][k]) for k in DIM_KEYS)
        lines.append(f"| {cat} | {st['n']} | {_fmt(st['mean_overall'])} | {cells} |")
    lines.append("")

    lines += ["## Scores by difficulty", "", "| Difficulty | n | Overall |", "|---|---:|---:|"]
    for diff, st in agg["by_difficulty"].items():
        lines.append(f"| {diff} | {st['n']} | {_fmt(st['mean_overall'])} |")
    lines.append("")

    lines += ["## Retrieval dependence", "",
              "Questions marked `requires_retrieval` need Nigeria-specific facts a base model "
              "would not reliably hold. This split is where a RAG layer has to earn its place.",
              "", "| requires_retrieval | n | Overall | nigerian_specificity | citation_quality |",
              "|---|---:|---:|---:|---:|"]
    for flag, st in agg["by_requires_retrieval"].items():
        lines.append(
            f"| {flag} | {st['n']} | {_fmt(st['mean_overall'])} | "
            f"{_fmt(st['by_dimension']['nigerian_specificity'])} | "
            f"{_fmt(st['by_dimension']['citation_quality'])} |"
        )
    lines.append("")

    cov = agg["key_fact_coverage"]
    lines += [
        "## Key-fact coverage: two independent measures", "",
        f"- Programmatic (lexical, computed in code): **{_fmt(cov['programmatic_mean'])}** "
        "of facts covered",
        f"- Judge, as a fraction of the checklist it marked found: "
        f"**{_fmt(cov['judge_mean_normalised'])}**",
        f"- Judge, as a 1 to 5 dimension score: **{_fmt(cov['judge_dimension_mean'])}**",
        "",
        "These are computed independently on purpose. The lexical measure is a proxy and will "
        "differ in absolute value; what matters is that both move across questions.",
        "",
    ]

    num = agg["numeric"]
    if num["n"]:
        lines += ["## Numeric questions (graded in code, not by the judge)", "",
                  f"- {num['n_correct']} of {num['n']} correct within tolerance "
                  f"({_fmt((num['accuracy'] or 0) * 100, '.0f')} percent)", "",
                  "| Question | Target | Matched | Scale | Rel. error | Correct |",
                  "|---|---:|---:|---|---:|---|"]
        for rec in records:
            if "numeric_correct" not in rec:
                continue
            lines.append(
                f"| {rec['question_id']} | {rec['numeric_target']:g} "
                f"| {_fmt(rec.get('numeric_matched'), '.6g')} "
                f"| {rec.get('numeric_matched_scale') or 'n/a'} "
                f"| {_fmt(rec.get('numeric_rel_error'), '.4f')} "
                f"| {'yes' if rec['numeric_correct'] else 'NO'} |"
            )
        lines.append("")

    traps = agg["traps"]
    if traps["n"]:
        lines += ["## Trap questions (false premises)", "",
                  f"- {traps['n']} trap questions. Mean hallucination score (higher is better): "
                  f"**{_fmt(traps['mean_hallucination'])}**, mean overall {_fmt(traps['mean_overall'])}",
                  "", "| Question | Hallucination | Factual | Verdict |", "|---|---:|---:|---|"]
        for rec in records:
            if not rec.get("trap") or not rec.get("scores"):
                continue
            lines.append(
                f"| {rec['question_id']} | {rec['scores'].get('hallucination', 'n/a')} "
                f"| {rec['scores'].get('factual_accuracy', 'n/a')} "
                f"| {(rec.get('one_line_verdict') or '').replace('|', '/')[:120]} |"
            )
        lines.append("")

    judged = [r for r in records if r.get("mean_score") is not None]
    worst = sorted(judged, key=lambda r: r["mean_score"])[:10]
    if worst:
        lines += ["## Ten weakest answers", "", "| Question | Category | Mean | Verdict |",
                  "|---|---|---:|---|"]
        for rec in worst:
            lines.append(
                f"| {rec['question_id']} | {rec['category']} | {_fmt(rec['mean_score'])} "
                f"| {(rec.get('one_line_verdict') or '').replace('|', '/')[:120]} |"
            )
        lines.append("")

    failures = [r for r in records if not r.get("judge_ok")]
    if failures:
        lines += ["## Unjudged questions", "", "| Question | Reason |", "|---|---|"]
        for rec in failures:
            reason = rec.get("judge_error") or rec.get("answer_error") or "unknown"
            lines.append(f"| {rec['question_id']} | {str(reason).replace('|', '/')[:160]} |")
        lines.append("")

    lines += ["---", "",
              "Scores in this report are only meaningful if the sanity check above passed. "
              "The previous evaluation of this project reported a flat 3.00 for both the base and "
              "the fine-tuned model with a delta of exactly zero, which is why every table here "
              "carries a spread column.", ""]
    return "\n".join(lines)


def render_pairwise_markdown(payload: dict[str, Any]) -> str:
    meta, agg, sanity = payload["meta"], payload["aggregate"], payload["sanity"]
    a, b = agg["system_a"], agg["system_b"]
    lines = [
        f"# Pairwise comparison: {a} against {b}",
        "",
        f"- Run at: {meta['started_at']}",
        f"- Judge: `{meta['judge']['model']}` at `{meta['judge']['base_url']}`",
        f"- Order randomised per question with seed {meta['seed']}; the mapping is recorded on "
        "every record so position bias is measurable.",
        f"- {agg['n_judged']} of {agg['n_questions']} comparisons judged "
        f"({agg['n_judge_failures']} failures)",
        "",
        "| Outcome | Count | Rate |",
        "|---|---:|---:|",
        f"| {a} wins | {agg['wins_a']} | {agg['win_rate_a'] * 100:.1f} percent |",
        f"| {b} wins | {agg['wins_b']} | {agg['win_rate_b'] * 100:.1f} percent |",
        f"| Ties | {agg['ties']} | **{agg['tie_rate'] * 100:.1f} percent** |",
        "",
        (f"Position 1 won {agg['position_1_win_rate'] * 100:.1f} percent of "
         f"{agg['n_decided']} decided comparisons (50 percent is unbiased)."
         if agg["position_1_win_rate"] is not None
         else "No comparison was decided either way, so position bias could not be measured."),
        "",
    ]
    lines += _sanity_block(sanity)

    lines += ["## By dimension", "", f"| Dimension | {a} | {b} | Tie |", "|---|---:|---:|---:|"]
    for key in DIM_KEYS:
        tally = agg["by_dimension"][key]
        lines.append(f"| {key} | {tally['a']} | {tally['b']} | {tally['tie']} |")
    lines.append("")

    lines += ["## By category", "", f"| Category | {a} | {b} | Tie |", "|---|---:|---:|---:|"]
    for cat, tally in sorted(agg["by_category"].items()):
        lines.append(f"| {cat} | {tally['a']} | {tally['b']} | {tally['tie']} |")
    lines += ["", "---", "",
              "The tie rate is broken out deliberately. An all-ties result is the degenerate "
              "signature this harness exists to catch, and it is not evidence that two systems "
              "are equivalent.", ""]
    return "\n".join(lines)


def render_compare_markdown(cmp: dict[str, Any], meta: dict[str, Any]) -> str:
    a, b = cmp["system_a"], cmp["system_b"]
    lines = [
        f"# Delta report: {b} against {a}",
        "",
        f"- Generated at: {meta['generated_at']}",
        f"- Source runs: `{meta['run_a']}` and `{meta['run_b']}`",
        f"- Questions judged in both runs: {cmp['n_shared']}",
        f"- Overall mean: {a} {_fmt(cmp['overall_mean_a'])}, {b} {_fmt(cmp['overall_mean_b'])}, "
        f"delta **{_fmt(cmp['overall_delta_b_minus_a'], '+.4f')}**",
        "",
    ]
    lines += _sanity_block(cmp["sanity"])
    lines += ["## Per-dimension paired delta", "",
              f"| Dimension | n | Mean delta ({b} minus {a}) | SD of delta | t-like | "
              f"{b} better | {a} better | Equal |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in DIM_KEYS:
        st = cmp["by_dimension"][key]
        if not st.get("n"):
            lines.append(f"| {key} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {key} | {st['n']} | {st['mean_delta_b_minus_a']:+.4f} | "
            f"{st['sd_of_paired_delta']:.3f} | {_fmt(st['t_like'], '.2f')} | "
            f"{st['b_better']} | {st['a_better']} | {st['equal']} |"
        )
    lines += ["",
              "`t-like` is mean delta divided by the standard error of the paired differences. It "
              "is a rough guide, not a significance test: above about 2 in absolute value the "
              "difference is unlikely to be sampling noise at this sample size.",
              "",
              "A delta of exactly or nearly zero on every dimension is the failure this harness "
              "was built to detect. Check both source runs' own sanity verdicts before reading "
              "any conclusion into it.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #

def add_target_args(parser: argparse.ArgumentParser, prefix: str, label: str) -> None:
    p = f"--{prefix}" if prefix else "--"
    pre = prefix + "-" if prefix else ""
    parser.add_argument(f"--{pre}name", default=None, help=f"Label for {label} in the output")
    parser.add_argument(f"--{pre}kind", choices=("app", "openai"), default="openai",
                        help=f"{label}: 'app' for the deployed /chat SSE endpoint, 'openai' for "
                             "any OpenAI-compatible chat endpoint")
    parser.add_argument(f"--{pre}url", default=None,
                        help=f"{label} base URL, for example https://naija-petro.shinzii.tech "
                             "or http://localhost:11434/v1")
    parser.add_argument(f"--{pre}model", default="",
                        help=f"{label} model name (openai kind only)")
    parser.add_argument(f"--{pre}key-env", default="",
                        help=f"Name of the env var holding {label}'s API key (never the key)")
    parser.add_argument(f"--{pre}token-env", default="",
                        help=f"Name of the env var holding the app access token for {label}")
    parser.add_argument(f"--{pre}reasoning", action="store_true",
                        help=f"{label}: ask the app for a reasoning trace (stripped before judging)")
    parser.add_argument(f"--{pre}units", choices=("field", "si"), default="field",
                        help=f"{label}: unit system requested from the app")
    parser.add_argument(f"--{pre}temperature", type=float, default=0.2)
    parser.add_argument(f"--{pre}max-tokens", type=int, default=1400)
    del p


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--category", action="append", default=None, help="Repeatable filter")
    parser.add_argument("--difficulty", action="append", default=None, help="Repeatable filter")
    parser.add_argument("--id", action="append", default=None, help="Repeatable question id filter")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Questions in flight at once. Keep low against a single cold GPU.")
    parser.add_argument("--answer-retries", type=int, default=2)
    parser.add_argument("--answer-timeout", type=float, default=300.0,
                        help="Seconds. The app can cold-start for one to two minutes.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Results JSON path. A .jsonl (resumable) and a .md are written "
                             "alongside it.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip questions already present in the sibling .jsonl")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the exact prompts and exit without any network call")
    parser.add_argument("--system-prompt-file", type=Path, default=None,
                        help="Override the system prompt sent to openai-kind targets")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="Send no system prompt at all (bare base-model comparison)")
    # Judge
    parser.add_argument("--judge-url", default=None,
                        help=f"Judge base URL. Defaults to ${DEFAULT_JUDGE_URL_ENV} then "
                             f"{DEFAULT_JUDGE_BASE_URL}")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-key-env", default=DEFAULT_JUDGE_KEY_ENV,
                        help="Name of the env var holding the judge API key")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--judge-timeout", type=float, default=180.0)
    parser.add_argument("--judge-thinking", choices=("on", "off"), default="off",
                        help="Nemotron models take a 'detailed thinking on/off' system line")


def resolve_system_prompt(args: argparse.Namespace) -> str:
    if getattr(args, "no_system_prompt", False):
        return ""
    path = getattr(args, "system_prompt_file", None)
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    return EVAL_SYSTEM_PROMPT


def target_from_args(args: argparse.Namespace, prefix: str, default_name: str) -> TargetSpec:
    pre = (prefix + "_") if prefix else ""

    def get(field_name: str, default: Any = None) -> Any:
        return getattr(args, pre + field_name, default)

    url = get("url")
    if not url:
        raise SystemExit(f"--{prefix + '-' if prefix else ''}url is required")
    kind = get("kind", "openai")
    model = get("model", "") or ""
    if kind == "openai" and not model:
        raise SystemExit(f"--{prefix + '-' if prefix else ''}model is required for an openai target")
    return TargetSpec(
        name=get("name") or default_name or (model or url),
        kind=kind,
        url=url,
        model=model,
        key_env=get("key_env", "") or "",
        token_env=get("token_env", "") or "",
        reasoning=bool(get("reasoning", False)),
        units=get("units", "field") or "field",
        temperature=float(get("temperature", 0.2)),
        max_tokens=int(get("max_tokens", 1400)),
        system_prompt=resolve_system_prompt(args),
    )


def judge_from_args(args: argparse.Namespace) -> JudgeConfig:
    base = args.judge_url or os.environ.get(DEFAULT_JUDGE_URL_ENV) or DEFAULT_JUDGE_BASE_URL
    return JudgeConfig(
        base_url=base,
        model=args.judge_model,
        key_env=args.judge_key_env,
        temperature=args.judge_temperature,
        retries=args.judge_retries,
        timeout=args.judge_timeout,
        thinking=args.judge_thinking,
    )


def sibling_paths(out: Path) -> tuple[Path, Path]:
    return out.with_suffix(".jsonl"), out.with_suffix(".md")


def warn_missing_key(env_name: str, what: str) -> None:
    if env_name and not os.environ.get(env_name):
        print(f"[warn] ${env_name} is not set in the environment or .env; {what} will be called "
              "without an Authorization header.", file=sys.stderr)


def print_dry_run(questions: Sequence[dict[str, Any]], specs: Sequence[TargetSpec],
                  judge_cfg: JudgeConfig, pairwise: bool, seed: int) -> None:
    print("=" * 78)
    print("DRY RUN. No network call is made.")
    print("=" * 78)
    print(f"\nQuestions selected: {len(questions)}")
    for spec in specs:
        print(f"\nTarget: {json.dumps(spec.describe(), indent=2)}")
        if spec.kind == "openai" and spec.system_prompt:
            print(f"\n--- system prompt sent to {spec.name} ---\n{spec.system_prompt}")
        elif spec.kind == "openai":
            print(f"\n--- no system prompt sent to {spec.name} ---")
        else:
            print(f"\n--- {spec.name} is the deployed app; it applies its own system prompt "
                  "and retrieval ---")
    print(f"\nJudge: {json.dumps(judge_cfg.describe(), indent=2)}")

    sample = questions[:2]
    for q in sample:
        print("\n" + "=" * 78)
        print(f"QUESTION {q['id']} ({q['category']}, {q['difficulty']})")
        print("=" * 78)
        print(f"\n--- user turn sent to the target ---\n{q['question']}")
        if pairwise:
            rng = random.Random(f"{seed}:{q['id']}")
            a_first = rng.random() < 0.5
            first = f"<answer from {specs[0].name if a_first else specs[1].name}>"
            second = f"<answer from {specs[1].name if a_first else specs[0].name}>"
            print(f"\n--- pairwise judge system ---\n{PAIRWISE_SYSTEM}")
            print(f"\n--- pairwise judge user ---\n{build_pairwise_prompt(q, first, second)}")
        else:
            print(f"\n--- judge system ---\n{judge_cfg_system_preview(judge_cfg)}")
            print("\n--- judge user ---")
            print(build_judge_prompt(q, "<the target's answer is inserted here verbatim>"))
        if "numeric_answer" in q:
            print(f"\n--- deterministic numeric grading ---\ntarget {q['numeric_answer']} "
                  f"{q.get('unit', '')} within relative tolerance "
                  f"{q.get('rel_tolerance', 0.02)} (graded in code, never by the judge)")
    if len(questions) > len(sample):
        print(f"\n... {len(questions) - len(sample)} further questions not shown.")
    print("\nDry run complete. Remove --dry-run to execute.")


def judge_cfg_system_preview(cfg: JudgeConfig) -> str:
    if cfg.thinking in ("on", "off") and "nemotron" in cfg.model.lower():
        return f"detailed thinking {cfg.thinking}\n\n{JUDGE_SYSTEM}"
    return JUDGE_SYSTEM


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

async def cmd_single(args: argparse.Namespace) -> int:
    questions = load_questions(args.questions, limit=args.limit, categories=args.category,
                               difficulties=args.difficulty, ids=args.id)
    spec = target_from_args(args, "", "system-under-test")
    judge_cfg = judge_from_args(args)

    if args.dry_run:
        print_dry_run(questions, [spec], judge_cfg, pairwise=False, seed=args.seed)
        return 0

    warn_missing_key(judge_cfg.key_env, "the judge")
    if spec.kind == "openai":
        warn_missing_key(spec.key_env, f"target {spec.name}")

    jsonl_path, md_path = sibling_paths(args.out)
    sink = RecordSink(jsonl_path)
    done = sink.load() if args.resume else {}
    if done:
        print(f"[resume] {len(done)} records already in {jsonl_path}")

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sem = asyncio.Semaphore(max(1, args.concurrency))
    pending = [q for q in questions if f"single|{spec.name}|{q['id']}" not in done]
    control_ids = {q["id"] for q in pending[:max(0, args.control_probe)]}
    completed = 0

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def worker(question: dict[str, Any]) -> dict[str, Any]:
            nonlocal completed
            async with sem:
                record = await evaluate_one(
                    client, spec, judge_cfg, question,
                    answer_retries=args.answer_retries,
                    answer_timeout=args.answer_timeout,
                    fact_threshold=args.fact_threshold,
                    control_probe=question["id"] in control_ids,
                )
                await sink.write(record)
                completed += 1
                mean = record.get("mean_score")
                status = f"{mean:.2f}" if mean is not None else "JUDGE FAIL"
                print(f"[{completed}/{len(pending)}] {question['id']:6s} "
                      f"{question['category']:24s} {status}", flush=True)
                return record

        fresh = await asyncio.gather(*(worker(q) for q in pending), return_exceptions=True)

    records = list(done.values())
    for item in fresh:
        if isinstance(item, BaseException):
            print(f"[error] worker raised: {item}", file=sys.stderr)
            continue
        records.append(item)
    records.sort(key=lambda r: r.get("question_id", ""))

    agg = aggregate_single(records)
    sanity = sanity_check_single(agg)
    payload = {
        "meta": {
            "mode": "single",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "questions_file": str(args.questions),
            "n_selected": len(questions),
            "target": spec.describe(),
            "judge": judge_cfg.describe(),
            "fact_threshold": args.fact_threshold,
            "control_probe": args.control_probe,
        },
        "aggregate": agg,
        "sanity": sanity,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_single_markdown(payload), encoding="utf-8")
    print(f"\nWrote {args.out}\nWrote {md_path}\nRaw records: {jsonl_path}")
    print_verdict(sanity)
    return 0 if sanity["passed"] else 2


async def cmd_pairwise(args: argparse.Namespace) -> int:
    questions = load_questions(args.questions, limit=args.limit, categories=args.category,
                               difficulties=args.difficulty, ids=args.id)
    spec_a = target_from_args(args, "a", "system-a")
    spec_b = target_from_args(args, "b", "system-b")
    if spec_a.name == spec_b.name:
        spec_b.name = spec_b.name + "-b"
    judge_cfg = judge_from_args(args)

    if args.dry_run:
        print_dry_run(questions, [spec_a, spec_b], judge_cfg, pairwise=True, seed=args.seed)
        return 0

    warn_missing_key(judge_cfg.key_env, "the judge")
    for spec in (spec_a, spec_b):
        if spec.kind == "openai":
            warn_missing_key(spec.key_env, f"target {spec.name}")

    jsonl_path, md_path = sibling_paths(args.out)
    sink = RecordSink(jsonl_path)
    done = sink.load() if args.resume else {}
    if done:
        print(f"[resume] {len(done)} records already in {jsonl_path}")

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sem = asyncio.Semaphore(max(1, args.concurrency))
    key_of = lambda q: f"pairwise|{spec_a.name}|{spec_b.name}|{q['id']}"  # noqa: E731
    pending = [q for q in questions if key_of(q) not in done]
    completed = 0

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def worker(question: dict[str, Any]) -> dict[str, Any]:
            nonlocal completed
            async with sem:
                record = await evaluate_pair(
                    client, spec_a, spec_b, judge_cfg, question,
                    answer_retries=args.answer_retries,
                    answer_timeout=args.answer_timeout,
                    seed=args.seed,
                )
                await sink.write(record)
                completed += 1
                print(f"[{completed}/{len(pending)}] {question['id']:6s} "
                      f"winner: {record.get('winner') or 'JUDGE FAIL'}", flush=True)
                return record

        fresh = await asyncio.gather(*(worker(q) for q in pending), return_exceptions=True)

    records = list(done.values())
    for item in fresh:
        if isinstance(item, BaseException):
            print(f"[error] worker raised: {item}", file=sys.stderr)
            continue
        records.append(item)
    records.sort(key=lambda r: r.get("question_id", ""))

    agg = aggregate_pairwise(records, spec_a.name, spec_b.name)
    sanity = sanity_check_pairwise(agg)
    payload = {
        "meta": {
            "mode": "pairwise",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "questions_file": str(args.questions),
            "seed": args.seed,
            "target_a": spec_a.describe(),
            "target_b": spec_b.describe(),
            "judge": judge_cfg.describe(),
        },
        "aggregate": agg,
        "sanity": sanity,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_pairwise_markdown(payload), encoding="utf-8")
    print(f"\nWrote {args.out}\nWrote {md_path}\nRaw records: {jsonl_path}")
    print_verdict(sanity)
    return 0 if sanity["passed"] else 2


def cmd_compare(args: argparse.Namespace) -> int:
    """Offline delta report between two completed single runs. Makes no network call."""
    run_a = json.loads(args.run_a.read_text(encoding="utf-8"))
    run_b = json.loads(args.run_b.read_text(encoding="utf-8"))
    for run, path in ((run_a, args.run_a), (run_b, args.run_b)):
        if run.get("meta", {}).get("mode") != "single":
            raise SystemExit(f"{path} is not a 'single' mode results file")
    cmp = compare_runs(run_a, run_b)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_a": str(args.run_a),
        "run_b": str(args.run_b),
        "run_a_sanity_passed": run_a.get("sanity", {}).get("passed"),
        "run_b_sanity_passed": run_b.get("sanity", {}).get("passed"),
    }
    if not meta["run_a_sanity_passed"] or not meta["run_b_sanity_passed"]:
        cmp["sanity"]["warnings"].insert(
            0,
            "At least one source run failed its own sanity check, so this delta inherits that "
            "failure. Fix the run before reading anything into the comparison.",
        )
        cmp["sanity"]["passed"] = False
    payload = {"meta": meta, "comparison": cmp}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = args.out.with_suffix(".md")
    md_path.write_text(render_compare_markdown(cmp, meta), encoding="utf-8")
    print(f"Wrote {args.out}\nWrote {md_path}")
    print_verdict(cmp["sanity"])
    return 0 if cmp["sanity"]["passed"] else 2


def cmd_validate(args: argparse.Namespace) -> int:
    """Structural check of the question set. Makes no network call."""
    questions = load_questions(args.questions)
    by_cat: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    numeric, traps, retrieval = 0, 0, 0
    problems: list[str] = []
    seen: set[str] = set()
    for q in questions:
        if q["id"] in seen:
            problems.append(f"duplicate id: {q['id']}")
        seen.add(q["id"])
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1
        by_diff[q["difficulty"]] = by_diff.get(q["difficulty"], 0) + 1
        if q["difficulty"] not in ("easy", "medium", "hard"):
            problems.append(f"{q['id']}: unexpected difficulty {q['difficulty']}")
        if not isinstance(q["key_facts"], list) or len(q["key_facts"]) < 3:
            problems.append(f"{q['id']}: key_facts must be a list of at least three claims")
        if not isinstance(q["requires_retrieval"], bool):
            problems.append(f"{q['id']}: requires_retrieval must be a boolean")
        if "numeric_answer" in q:
            numeric += 1
            if "unit" not in q:
                problems.append(f"{q['id']}: numeric_answer without a unit")
        if q.get("trap"):
            traps += 1
        if q["requires_retrieval"]:
            retrieval += 1
    print(f"questions: {len(questions)}")
    print("by category:")
    for cat, count in sorted(by_cat.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {cat:26s} {count}")
    print(f"by difficulty: {dict(sorted(by_diff.items()))}")
    print(f"numeric: {numeric}   traps: {traps}   requires_retrieval: {retrieval}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\nNo structural problems found.")
    return 0


def print_verdict(sanity: dict[str, Any]) -> None:
    print()
    if sanity["passed"]:
        print("SANITY CHECK: PASS. Scores are usable, subject to the notes in the report.")
    else:
        print("!" * 78)
        print("SANITY CHECK: FAIL. DO NOT PUBLISH THESE SCORES.")
        for warn in sanity["warnings"]:
            print(f"  - {warn}")
        print("!" * 78)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description=(
            "Evaluation harness for Naija-Petro. Scores per dimension against an anchored "
            "rubric, grades numeric questions in code, and checks itself for the flat-judge "
            "failure that made the previous evaluation unusable."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # what would be sent, no network call\n"
            "  python eval/run_eval.py single --kind openai --url http://localhost:11434/v1 \\\n"
            "      --model qwen3:8b --name base-8b --out eval/results/base.json --dry-run\n\n"
            "  # score the deployed app\n"
            "  python eval/run_eval.py single --kind app --url https://naija-petro.shinzii.tech \\\n"
            "      --name app-rag --out eval/results/app.json --control-probe 5\n\n"
            "  # head to head with randomised order\n"
            "  python eval/run_eval.py pairwise --a-url http://localhost:11434/v1 "
            "--a-model qwen3:8b \\\n"
            "      --a-name base-8b --b-kind app --b-url https://naija-petro.shinzii.tech \\\n"
            "      --b-name app-rag --out eval/results/base_vs_app.json\n\n"
            "  # offline delta between two finished single runs\n"
            "  python eval/run_eval.py compare --run-a eval/results/base.json \\\n"
            "      --run-b eval/results/finetuned.json --out eval/results/delta.json\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    single = sub.add_parser("single", help="Score one system per dimension")
    add_common_args(single)
    add_target_args(single, "", "the system under test")
    single.add_argument("--control-probe", type=int, default=0,
                        help="Also judge a canned filler answer on the first N questions, to test "
                             "that the judge can tell filler from a real answer")
    single.add_argument("--fact-threshold", type=float, default=0.6,
                        help="Lexical overlap fraction at which a key fact counts as covered")
    single.add_argument("--seed", type=int, default=1337)
    single.set_defaults(func=lambda a: asyncio.run(cmd_single(a)))

    pair = sub.add_parser("pairwise", help="Compare two systems with randomised answer order")
    add_common_args(pair)
    add_target_args(pair, "a", "system A")
    add_target_args(pair, "b", "system B")
    pair.add_argument("--seed", type=int, default=1337,
                      help="Controls the per-question answer order; change it to test position bias")
    pair.set_defaults(func=lambda a: asyncio.run(cmd_pairwise(a)))

    comp = sub.add_parser("compare", help="Offline per-dimension delta between two single runs")
    comp.add_argument("--run-a", type=Path, required=True, help="Baseline results JSON")
    comp.add_argument("--run-b", type=Path, required=True, help="Challenger results JSON")
    comp.add_argument("--out", type=Path, required=True)
    comp.set_defaults(func=cmd_compare)

    val = sub.add_parser("validate", help="Structural check of questions.jsonl, no network call")
    val.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    val.set_defaults(func=cmd_validate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()   # reads .env for NVIDIA_API_KEY, NVIDIA_BASE_URL and any target key
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
