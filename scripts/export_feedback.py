#!/usr/bin/env python3
"""Export collected feedback from Supabase into training-ready JSONL.

Produces, in the current directory:
  - feedback_raw.jsonl : every rating with query, answer, rating, comment, sources
  - feedback_sft.jsonl : positively-rated answers in Alpaca format for LoRA SFT
  - feedback_dpo.jsonl : (query, chosen, rejected) pairs where the same question has
                         both an up- and a down-rated answer (for DPO/preference tuning)

This is the data side of the improvement loop: run it periodically, then fine-tune
the 8B (LoRA/DPO) on the exported files. Reads the store over the Supabase REST API
(SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY + SUPABASE_DB_SCHEMA in .env).

    python scripts/export_feedback.py
"""
import json
import os
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
PAGE = 1000  # PostgREST caps rows per response (PGRST_DB_MAX_ROWS); paginate.


def fetch_feedback() -> list[dict]:
    """All rated rows with an answer, oldest first, paged over the REST API."""
    headers = {
        "apikey": KEY,
        "Authorization": f"Bearer {KEY}",
        "Accept-Profile": SCHEMA,
    }
    rows: list[dict] = []
    offset = 0
    with httpx.Client(base_url=f"{URL}/rest/v1", timeout=60) as c:
        while True:
            r = c.get(
                "/feedback",
                params={
                    "select": "query,answer,rating,comment,sources,created_at",
                    "answer": "not.is.null",
                    "order": "created_at.asc",
                    "limit": PAGE,
                    "offset": offset,
                },
                headers=headers,
            )
            r.raise_for_status()
            batch = r.json()
            rows.extend(batch)
            if len(batch) < PAGE:
                break
            offset += PAGE
    return rows


def main() -> int:
    if not URL or not KEY:
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set in .env")
        return 1
    rows = fetch_feedback()

    by_query = defaultdict(lambda: {"chosen": [], "rejected": []})
    n_sft = 0
    with open("feedback_raw.jsonl", "w") as raw, open("feedback_sft.jsonl", "w") as sft:
        for row in rows:
            query = row.get("query")
            answer = row.get("answer")
            rating = row.get("rating")
            raw.write(json.dumps({"query": query, "answer": answer, "rating": rating,
                                  "comment": row.get("comment"), "sources": row.get("sources"),
                                  "ts": str(row.get("created_at"))}) + "\n")
            if rating and rating >= 1:
                sft.write(json.dumps({"instruction": query, "input": "", "output": answer}) + "\n")
                n_sft += 1
                by_query[query]["chosen"].append(answer)
            elif rating and rating <= -1:
                by_query[query]["rejected"].append(answer)

    n_dpo = 0
    with open("feedback_dpo.jsonl", "w") as dpo:
        for query, v in by_query.items():
            if v["chosen"] and v["rejected"]:
                dpo.write(json.dumps({"prompt": query, "chosen": v["chosen"][0],
                                      "rejected": v["rejected"][0]}) + "\n")
                n_dpo += 1

    print(f"feedback rows: {len(rows)}")
    print(f"feedback_raw.jsonl : {len(rows)} rows")
    print(f"feedback_sft.jsonl : {n_sft} positive examples (Alpaca format for LoRA SFT)")
    print(f"feedback_dpo.jsonl : {n_dpo} preference pairs (for DPO)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
