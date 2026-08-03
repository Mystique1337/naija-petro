#!/usr/bin/env python3
"""Remove documents that should never have entered the knowledge base.

Ingestion now refuses video, homework and answer-mill domains, and non-Latin
script documents (app/rag/sources.py), but anything already stored predates those
filters and keeps being retrieved. This applies the same rules to what is already
there.

It found, in a live store: YouTube listings, a Chegg "Solved..." page, SEO
calculator pages, a World Bank household-survey interviewer manual, and a Chinese
edition of the SPE reserves guidelines that had become the single largest document
present and was being retrieved for ordinary English questions.

    python scripts/purge_junk.py                 # dry run, lists what it would remove
    python scripts/purge_junk.py --apply         # actually delete
    python scripts/purge_junk.py --apply --yes   # no confirmation prompt

Reads the store over the REST API (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
SUPABASE_DB_SCHEMA, SUPABASE_REST_PATH), so it works against the deployed Supabase
and against the local stack (source .localdb/env first).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:                       # dotenv is optional if the env is already set
    pass

import httpx  # noqa: E402

from app.rag.sources import is_blocked, is_english  # noqa: E402

# Titles that identify a page carrying no engineering content. Matched case
# insensitively against the title only, so a report that merely mentions cookies
# in its body is untouched.
JUNK_TITLE_MARKERS = (
    "privacy policy", "cookie policy", "terms of use", "terms and conditions",
    "interviewer's manual", "ai writing report", "sitemap", "page not found",
)


def _cfg() -> tuple[str, str, str]:
    url = (os.environ.get("SUPABASE_URL") or "").strip().strip('"').rstrip("/")
    key = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
           or os.environ.get("SUPABASE_SERVICE_KEY") or "").strip().strip('"')
    schema = (os.environ.get("SUPABASE_DB_SCHEMA") or "naija_petro").strip().strip('"')
    if not url or not key:
        sys.exit("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set. "
                 "For the local stack run: source .localdb/env")
    path = os.environ.get("SUPABASE_REST_PATH", "/rest/v1")
    return f"{url}{path}", key, schema


def reason_to_drop(doc: dict) -> str | None:
    """Why this document does not belong, or None to keep it."""
    title = (doc.get("title") or "").strip()
    url = doc.get("source_url") or ""
    if url and not url.startswith("upload://") and is_blocked(url):
        return "blocked domain"
    if title and not is_english(title):
        return "non-english title"
    content = doc.get("content") or ""
    if content and not is_english(content):
        return "non-english content"
    low = title.lower()
    for marker in JUNK_TITLE_MARKERS:
        if marker in low:
            return f"boilerplate page ({marker})"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Purge junk documents from the knowledge base.")
    ap.add_argument("--apply", action="store_true", help="delete, instead of only listing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    base, key, schema = _cfg()
    headers = {"apikey": key, "Authorization": f"Bearer {key}",
               "Accept-Profile": schema, "Content-Profile": schema}
    client = httpx.Client(base_url=base, headers=headers, timeout=120)

    docs: list[dict] = []
    offset = 0
    while True:
        r = client.get("/documents", params={
            "select": "id,title,source_url,domain,content", "limit": 500, "offset": offset,
            "order": "id.asc"})
        r.raise_for_status()
        batch = r.json()
        docs.extend(batch)
        if not batch:
            break
        offset += len(batch)

    flagged = [(d, why) for d in docs if (why := reason_to_drop(d))]
    print(f"{len(docs)} documents in {schema}, {len(flagged)} flagged for removal\n")
    if not flagged:
        print("Nothing to do.")
        return 0
    for d, why in flagged:
        print(f"  [{why:28}] {(d.get('domain') or '?'):26} {(d.get('title') or '')[:52]}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to delete these and their chunks.")
        return 0
    if not args.yes:
        if input(f"\nDelete {len(flagged)} documents and all their chunks? Type DELETE: ").strip() != "DELETE":
            print("Aborted, nothing was changed.")
            return 1

    removed = 0
    for d, _ in flagged:
        did = d["id"]
        client.delete("/document_chunks", params={"document_id": f"eq.{did}"}).raise_for_status()
        client.delete("/documents", params={"id": f"eq.{did}"}).raise_for_status()
        removed += 1
    print(f"\nRemoved {removed} documents and their chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
