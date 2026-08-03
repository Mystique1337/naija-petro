#!/usr/bin/env python3
"""Seed the knowledge base with authoritative Nigerian petroleum sources.

Run AFTER `modal deploy modal_app.py`:

    python scripts/seed_kb.py                 # seed the built-in source list
    python scripts/seed_kb.py URL [URL ...]   # seed specific URLs

(Equivalent one-liner without this script: `modal run modal_app.py::seed`.)

This spends Modal GPU time: every fetched document is chunked and embedded on the
remote container. Pass explicit URLs to keep a re-run small.
"""
from __future__ import annotations

import sys
from pathlib import Path

# `python scripts/seed_kb.py` puts scripts/ on sys.path, not the repo root, so
# `from app.config import ...` fails with ModuleNotFoundError unless PYTHONPATH is
# already set. Put the repo root first so the script works from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def die(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 0

    try:
        import modal
    except ImportError:
        return die("the 'modal' package is not installed. Run: pip install modal")

    try:
        from app.config import APP_NAME
    except ImportError as exc:
        return die(f"cannot import app.config from {ROOT}: {exc}")

    urls = args or None
    try:
        seed = modal.Function.from_name(APP_NAME, "seed")
    except Exception as exc:
        return die(
            f"could not find the 'seed' function in Modal app '{APP_NAME}': "
            f"{type(exc).__name__}: {exc}. Deploy it first: modal deploy modal_app.py"
        )

    what = f"{len(urls)} URL(s)" if urls else "the built-in source list"
    print(f"Seeding the knowledge base from {what}. This fetches and embeds documents on a "
          "Modal container and costs GPU time.")
    try:
        result = seed.remote(urls)
    except Exception as exc:
        return die(f"the remote seed call failed: {type(exc).__name__}: {exc}")

    if not isinstance(result, dict):
        print(f"Done, but the remote returned {type(result).__name__} rather than a summary dict: "
              f"{result!r}")
        return 0
    print(f"Done: ingested {result.get('ingested_docs', 0)} docs, "
          f"{result.get('new_chunks', 0)} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
