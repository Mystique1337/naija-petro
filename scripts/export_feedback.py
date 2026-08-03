#!/usr/bin/env python3
"""Export collected feedback from Supabase into training-ready JSONL.

Produces, in the output directory (the current directory by default):
  - feedback_raw.jsonl : every rating with query, answer, rating, comment, sources
  - feedback_sft.jsonl : positively-rated answers in Alpaca format for LoRA SFT
  - feedback_dpo.jsonl : (query, chosen, rejected) pairs where the same question has
                         both an up- and a down-rated answer (for DPO/preference tuning)

This is the data side of the improvement loop: run it periodically, then fine-tune
the 8B (LoRA/DPO) on the exported files. Reads the store over the Supabase REST API
(SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY + SUPABASE_DB_SCHEMA + SUPABASE_REST_PATH
in .env).

    python scripts/export_feedback.py
    python scripts/export_feedback.py --out-dir data/feedback
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import httpx

URL = (os.environ.get("SUPABASE_URL") or "").strip().strip('"').rstrip("/")
KEY = (
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("SUPABASE_SERVICE_KEY")
    or ""
).strip().strip('"')
SCHEMA = (os.environ.get("SUPABASE_DB_SCHEMA") or "naija_petro").strip().strip('"')
# Supabase serves PostgREST under /rest/v1; a bare PostgREST (scripts/local_stack.sh)
# serves it at the root, which is why this is configurable and may be empty.
REST_PATH = os.environ.get("SUPABASE_REST_PATH", "/rest/v1").strip().strip('"')

PAGE = 1000        # rows we ask for per request
MAX_ROWS = 500_000  # refuse to spin forever if the server ignores offset


def die(message: str) -> int:
    """One clear line on stderr, no traceback."""
    print(f"error: {message}", file=sys.stderr)
    return 1


def headers(*, count: bool = False) -> dict[str, str]:
    h = {
        "apikey": KEY,
        "Authorization": f"Bearer {KEY}",
        "Accept-Profile": SCHEMA,
    }
    if count:
        h["Prefer"] = "count=exact"
    return h


def explain_http_error(exc: httpx.HTTPStatusError) -> str:
    """Turn a PostgREST error into something actionable. Never echoes credentials."""
    status = exc.response.status_code
    try:
        body = exc.response.json()
        detail = body.get("message") or body.get("hint") or str(body)
    except Exception:
        detail = (exc.response.text or "").strip()[:300]
    if status in (401, 403):
        return (
            f"HTTP {status} from PostgREST: {detail}. SUPABASE_SERVICE_ROLE_KEY is missing, "
            "wrong, or not a service_role key for this instance."
        )
    if status == 404:
        return (
            f"HTTP 404 from PostgREST: {detail}. The 'feedback' table is not exposed in schema "
            f"'{SCHEMA}'. Check SUPABASE_DB_SCHEMA and SUPABASE_REST_PATH."
        )
    if status == 400 and "column" in detail.lower():
        return (
            f"HTTP 400 from PostgREST: {detail}. The feedback table is missing a column this "
            "export selects (answer and sources are written by the app but are not in "
            "supabase/schema.sql). Add them before exporting."
        )
    return f"HTTP {status} from PostgREST: {detail}"


def _total_from_content_range(value: str) -> int | None:
    """PostgREST returns 'Content-Range: 0-999/12345' when asked to count."""
    if "/" not in value:
        return None
    tail = value.rsplit("/", 1)[1].strip()
    return int(tail) if tail.isdigit() else None


def fetch_feedback(client: httpx.Client) -> tuple[list[dict], int | None]:
    """All rated rows with an answer, oldest first, paged over the REST API.

    Two things the previous version got wrong:

      * it advanced the offset by the page size it *asked* for and stopped as soon
        as a short page came back. PostgREST caps rows per response at
        PGRST_DB_MAX_ROWS, so on any instance where that cap is below PAGE the very
        first response is short and the export silently stopped after one page.
        The offset now advances by the number of rows actually returned, and the
        loop only stops on an empty page or on reaching the server's own count.
      * `order=created_at.asc` alone is not a total order. Rows sharing a timestamp
        can be returned in a different order on each request, so offset paging can
        skip and duplicate them. `id` breaks the tie.
    """
    rows: list[dict] = []
    offset = 0
    total: int | None = None
    params = {
        "select": "id,query,answer,rating,comment,sources,created_at",
        "answer": "not.is.null",
        "order": "created_at.asc,id.asc",
    }
    while True:
        r = client.get(
            "/feedback",
            params={**params, "limit": PAGE, "offset": offset},
            headers=headers(count=total is None),
        )
        r.raise_for_status()
        if total is None:
            total = _total_from_content_range(r.headers.get("content-range", ""))
        batch = r.json()
        if not isinstance(batch, list):
            raise RuntimeError(f"unexpected response shape from /feedback: {type(batch).__name__}")
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)          # what came back, not what we asked for
        if total is not None and offset >= total:
            break
        if len(rows) >= MAX_ROWS:
            print(f"warning: stopped at {MAX_ROWS} rows. The server may be ignoring 'offset'.",
                  file=sys.stderr)
            break
    return rows, total


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("."),
                        help="Directory for the three JSONL files (default: current directory)")
    args = parser.parse_args()

    missing = [name for name, value in (("SUPABASE_URL", URL),
                                        ("SUPABASE_SERVICE_ROLE_KEY", KEY)) if not value]
    if missing:
        return die(
            f"{' and '.join(missing)} not set. Add it to .env, or run "
            "`source .localdb/env` to use the local stack from scripts/local_stack.sh."
        )

    base = f"{URL}{REST_PATH}"
    try:
        with httpx.Client(base_url=base, timeout=60) as client:
            rows, total = fetch_feedback(client)
    except httpx.HTTPStatusError as exc:
        return die(explain_http_error(exc))
    except httpx.HTTPError as exc:
        return die(f"could not reach {base}: {type(exc).__name__}: {exc}")
    except RuntimeError as exc:
        return die(str(exc))

    if total is not None and len(rows) != total:
        print(f"warning: the server counted {total} matching rows but {len(rows)} were "
              "retrieved. The export is incomplete.", file=sys.stderr)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "feedback_raw.jsonl"
    sft_path = out_dir / "feedback_sft.jsonl"
    dpo_path = out_dir / "feedback_dpo.jsonl"

    by_query: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"chosen": [], "rejected": []})
    n_sft = 0
    n_skipped = 0
    with raw_path.open("w", encoding="utf-8") as raw, sft_path.open("w", encoding="utf-8") as sft:
        for row in rows:
            query = row.get("query")
            answer = row.get("answer")
            rating = row.get("rating")
            raw.write(json.dumps({"query": query, "answer": answer, "rating": rating,
                                  "comment": row.get("comment"), "sources": row.get("sources"),
                                  "ts": str(row.get("created_at"))}, ensure_ascii=False) + "\n")
            # A row with no question text cannot be a training example, and grouping
            # every such row under one null key would fabricate DPO pairs between
            # unrelated questions.
            if not _norm(str(query or "")) or not _norm(str(answer or "")):
                n_skipped += 1
                continue
            try:
                rating = int(rating)
            except (TypeError, ValueError):
                continue
            key = _norm(str(query))
            if rating >= 1:
                sft.write(json.dumps({"instruction": query, "input": "", "output": answer},
                                     ensure_ascii=False) + "\n")
                n_sft += 1
                by_query[key]["chosen"].append(answer)
            elif rating <= -1:
                by_query[key]["rejected"].append(answer)

    n_dpo = 0
    with dpo_path.open("w", encoding="utf-8") as dpo:
        for key, v in by_query.items():
            if v["chosen"] and v["rejected"]:
                dpo.write(json.dumps({"prompt": key, "chosen": v["chosen"][0],
                                      "rejected": v["rejected"][0]}, ensure_ascii=False) + "\n")
                n_dpo += 1

    print(f"feedback rows: {len(rows)}" + (f" of {total} counted by the server" if total is not None else ""))
    print(f"{raw_path} : {len(rows)} rows")
    print(f"{sft_path} : {n_sft} positive examples (Alpaca format for LoRA SFT)")
    print(f"{dpo_path} : {n_dpo} preference pairs (for DPO)")
    if n_skipped:
        print(f"skipped {n_skipped} row(s) with an empty query or answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
