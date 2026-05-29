#!/usr/bin/env python3
"""Export collected feedback from Supabase into training-ready JSONL.

Produces, in the current directory:
  - feedback_raw.jsonl : every rating with query, answer, rating, comment, sources
  - feedback_sft.jsonl : positively-rated answers in Alpaca format for LoRA SFT
  - feedback_dpo.jsonl : (query, chosen, rejected) pairs where the same question has
                         both an up- and a down-rated answer (for DPO/preference tuning)

This is the data side of the improvement loop: run it periodically, then fine-tune
the 8B (LoRA/DPO) on the exported files. Reads SUPABASE_DB_URL from .env.

    python scripts/export_feedback.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import psycopg

URL = (os.environ.get("SUPABASE_DB_URL") or "").strip().strip('"')


def main() -> int:
    if not URL:
        print("SUPABASE_DB_URL not set in .env")
        return 1
    with psycopg.connect(URL, connect_timeout=20) as c, c.cursor() as cur:
        cur.execute(
            "SELECT query, answer, rating, comment, sources, created_at "
            "FROM feedback WHERE answer IS NOT NULL ORDER BY created_at"
        )
        rows = cur.fetchall()

    by_query = defaultdict(lambda: {"chosen": [], "rejected": []})
    n_sft = 0
    with open("feedback_raw.jsonl", "w") as raw, open("feedback_sft.jsonl", "w") as sft:
        for query, answer, rating, comment, sources, ts in rows:
            raw.write(json.dumps({"query": query, "answer": answer, "rating": rating,
                                  "comment": comment, "sources": sources, "ts": str(ts)}) + "\n")
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
