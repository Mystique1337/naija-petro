#!/usr/bin/env python3
"""Seed access tokens: 3 primary + 7 secondary. Idempotent (only seeds if empty).

Run once after deploying the schema:  python scripts/seed_tokens.py
Prints the tokens so you can share them. Manage (activate/deactivate) them in the
admin panel.
"""
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import psycopg

URL = (os.environ.get("SUPABASE_DB_URL") or "").strip().strip('"')
SCHEMA = (os.environ.get("SUPABASE_DB_SCHEMA") or "naija_petro").strip().strip('"')
# Prepend the app schema (e.g. naija_petro on a shared self-hosted Supabase).
DB_OPTS = f"-c search_path={SCHEMA},public" if SCHEMA and SCHEMA != "public" else "-c search_path=public"


def gen(prefix: str) -> str:
    return f"{prefix}-{secrets.token_urlsafe(12)}"


def main() -> int:
    if not URL:
        print("SUPABASE_DB_URL not set in .env")
        return 1
    with psycopg.connect(URL, connect_timeout=20, options=DB_OPTS) as c:
        c.autocommit = True
        with c.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS access_tokens (
                    id          BIGSERIAL PRIMARY KEY,
                    token       TEXT UNIQUE NOT NULL,
                    label       TEXT,
                    kind        TEXT DEFAULT 'secondary',
                    active      BOOLEAN DEFAULT TRUE,
                    created_at  TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            cur.execute("SELECT count(*) FROM access_tokens")
            n = cur.fetchone()[0]
            if n > 0:
                print(f"{n} tokens already exist; not re-seeding. Current tokens:")
                cur.execute("SELECT kind, label, active, token FROM access_tokens ORDER BY kind, id")
                for kind, label, active, token in cur.fetchall():
                    print(f"  [{kind:9}] {label:12} active={active}  {token}")
                return 0
            toks = [(gen("np-pri"), f"Primary {i}", "primary") for i in range(1, 4)]
            toks += [(gen("np-sec"), f"Secondary {i}", "secondary") for i in range(1, 8)]
            for token, label, kind in toks:
                cur.execute(
                    "INSERT INTO access_tokens (token, label, kind, active) VALUES (%s,%s,%s,TRUE)",
                    (token, label, kind),
                )
            print("Seeded 10 access tokens (share these; manage them in the admin panel):")
            for token, label, kind in toks:
                print(f"  [{kind:9}] {label:12} {token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
