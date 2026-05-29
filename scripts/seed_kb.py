#!/usr/bin/env python3
"""Seed the knowledge base with authoritative Nigerian petroleum sources.

Run AFTER `modal deploy modal_app.py`:

    python scripts/seed_kb.py                 # seed the built-in source list
    python scripts/seed_kb.py URL [URL ...]   # seed specific URLs

(Equivalent one-liner without this script: `modal run modal_app.py::seed`.)
"""
import sys

import modal

from app.config import APP_NAME


def main() -> int:
    urls = sys.argv[1:] or None
    seed = modal.Function.from_name(APP_NAME, "seed")
    print("Seeding knowledge base (this fetches + embeds several documents)…")
    result = seed.remote(urls)
    print(f"Done: ingested {result.get('ingested_docs', 0)} docs, "
          f"{result.get('new_chunks', 0)} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
