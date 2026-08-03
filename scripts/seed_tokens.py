#!/usr/bin/env python3
"""Seed access tokens: 3 primary + 7 secondary. Idempotent (only seeds if empty).

Run once after deploying the schema:

    python scripts/seed_tokens.py --dry-run   # check connectivity, write nothing
    python scripts/seed_tokens.py             # seed if the table is empty
    python scripts/seed_tokens.py --show      # ...and print the new tokens once

Talks to the store over the Supabase REST (PostgREST) API, the same path the app
uses (SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY + SUPABASE_DB_SCHEMA in .env). It
used to open a direct Postgres connection on SUPABASE_DB_URL, which on this
deployment points at 127.0.0.1 inside the database host and is unreachable from a
laptop, so the script only ever produced a connection timeout.

Tokens are secrets. They are printed masked unless you pass --show, and they are
never written to a file. Manage them (activate/deactivate) in the admin panel.
"""
from __future__ import annotations

import argparse
import os
import secrets
import sys
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

N_PRIMARY = 3
N_SECONDARY = 7


def die(message: str) -> int:
    """One clear line on stderr, no traceback."""
    print(f"error: {message}", file=sys.stderr)
    return 1


def gen(prefix: str) -> str:
    return f"{prefix}-{secrets.token_urlsafe(12)}"


def mask(token: str) -> str:
    """Enough to identify a token in a list, not enough to use it."""
    if not token:
        return "(empty)"
    if len(token) <= 10:
        return token[:2] + "*" * (len(token) - 2)
    return f"{token[:7]}{'*' * 8}{token[-4:]}"


def headers() -> dict[str, str]:
    return {
        "apikey": KEY,
        "Authorization": f"Bearer {KEY}",
        "Accept-Profile": SCHEMA,
        "Content-Profile": SCHEMA,
        "Content-Type": "application/json",
    }


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
            f"HTTP 404 from PostgREST: {detail}. The 'access_tokens' table is not exposed in "
            f"schema '{SCHEMA}'. Apply supabase/schema.sql, and check SUPABASE_DB_SCHEMA and "
            "SUPABASE_REST_PATH."
        )
    return f"HTTP {status} from PostgREST: {detail}"


def existing_count(client: httpx.Client) -> int:
    r = client.get(
        "/access_tokens",
        params={"select": "id", "limit": 1},
        headers={**headers(), "Prefer": "count=exact"},
    )
    r.raise_for_status()
    content_range = r.headers.get("content-range", "")
    total = content_range.rsplit("/", 1)[-1] if "/" in content_range else ""
    if total.isdigit():
        return int(total)
    # No count header (an older PostgREST): fall back to reading the ids.
    r = client.get("/access_tokens", params={"select": "id"}, headers=headers())
    r.raise_for_status()
    return len(r.json())


def list_existing(client: httpx.Client) -> list[dict]:
    r = client.get(
        "/access_tokens",
        params={"select": "kind,label,active,token", "order": "kind.asc,id.asc"},
        headers=headers(),
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed the access_tokens table over the Supabase REST API. Idempotent.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Check the connection and report the current count, write nothing")
    parser.add_argument("--show", action="store_true",
                        help="Print the newly minted tokens in full (they are secrets). "
                             "Without this they are masked")
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
        with httpx.Client(base_url=base, timeout=30) as client:
            n = existing_count(client)

            if n > 0:
                print(f"{n} token(s) already exist in {SCHEMA}.access_tokens; not re-seeding.")
                for row in list_existing(client):
                    print(f"  [{str(row.get('kind')):9}] {str(row.get('label')):12} "
                          f"active={row.get('active')}  {mask(str(row.get('token') or ''))}")
                print("Tokens are masked. Read them in the admin panel if you need one in full.")
                return 0

            if args.dry_run:
                print(f"Connected to {base} (schema {SCHEMA}). access_tokens is empty.")
                print(f"Would insert {N_PRIMARY} primary and {N_SECONDARY} secondary tokens.")
                print("Re-run without --dry-run to seed.")
                return 0

            rows = [{"token": gen("np-pri"), "label": f"Primary {i}", "kind": "primary",
                     "active": True} for i in range(1, N_PRIMARY + 1)]
            rows += [{"token": gen("np-sec"), "label": f"Secondary {i}", "kind": "secondary",
                      "active": True} for i in range(1, N_SECONDARY + 1)]

            r = client.post(
                "/access_tokens",
                json=rows,
                headers={**headers(), "Prefer": "return=minimal"},
            )
            r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return die(explain_http_error(exc))
    except httpx.HTTPError as exc:
        return die(f"could not reach {base}: {type(exc).__name__}: {exc}")

    print(f"Seeded {len(rows)} access tokens into {SCHEMA}.access_tokens.")
    for row in rows:
        shown = row["token"] if args.show else mask(row["token"])
        print(f"  [{row['kind']:9}] {row['label']:12} {shown}")
    if not args.show:
        print("\nTokens are masked. Re-run with --show to print them once, or read them in the "
              "admin panel. They are not written to any file.")
    else:
        print("\nThese are secrets: share them over a private channel and do not paste them "
              "into a file in this repo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
