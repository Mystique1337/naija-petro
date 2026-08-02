#!/usr/bin/env bash
# Bring up a complete Naija-Petro data layer on this machine: Postgres with
# pgvector, plus PostgREST in front of it.
#
# Why PostgREST and not a direct database client: the app talks to its store over
# PostgREST, so running one locally means local development exercises exactly the
# same code path as production instead of a parallel implementation that can drift.
#
#   bash scripts/local_stack.sh up       # install nothing, start everything, apply the schema
#   bash scripts/local_stack.sh status   # what is running, and the knowledge base size
#   bash scripts/local_stack.sh down     # stop both services, keep the data
#   bash scripts/local_stack.sh reset    # stop and delete the data directory
#
# Everything lives in .localdb/ inside the repo, which is gitignored. Nothing here
# touches the deployed app or its Supabase.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/.localdb"
PGDATA="$DATA/pgdata"
PGPORT="${LOCAL_PG_PORT:-55432}"
RESTPORT="${LOCAL_REST_PORT:-3111}"
DBNAME="naija_petro_local"
SCHEMA="naija_petro"
ENVFILE="$DATA/env"
PGLOG="$DATA/postgres.log"
RESTLOG="$DATA/postgrest.log"

BREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"

find_pg_bin() {
  # Prefer a versioned keg (they are keg-only and not on PATH by default).
  for d in "$BREW_PREFIX"/opt/postgresql@*/bin "$BREW_PREFIX"/bin; do
    if [[ -x "$d/initdb" && -x "$d/pg_ctl" ]]; then echo "$d"; return 0; fi
  done
  return 1
}

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing '$1'. Run: brew install $2" >&2; exit 1; }
}

port_taken() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

rest_running() {
  # Ours specifically: the pid we started, still alive.
  local pidfile="$DATA/postgrest.pid"
  [[ -f "$pidfile" ]] || return 1
  local pid; pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && curl -s -m 2 -o /dev/null "http://localhost:$RESTPORT/"
}

mint_jwt() {
  # PostgREST authorises by JWT role claim. Supabase's service key is exactly this,
  # so minting one locally keeps the app's auth headers unchanged. Stdlib only.
  python3 - "$1" <<'PY'
import base64, hashlib, hmac, json, sys

def b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

secret = sys.argv[1].encode()
header = b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
# No exp: this is a local development token for a database on loopback.
payload = b64(json.dumps({"role": "service_role", "iss": "naija-petro-local"},
                         separators=(",", ":")).encode())
signing_input = f"{header}.{payload}".encode()
sig = b64(hmac.new(secret, signing_input, hashlib.sha256).digest())
print(f"{header}.{payload}.{sig}")
PY
}

cmd_up() {
  require brew brew
  PGBIN="$(find_pg_bin)" || { echo "Postgres not found. Run: brew install pgvector" >&2; exit 1; }
  require postgrest postgrest

  mkdir -p "$DATA"

  if [[ ! -d "$PGDATA" ]]; then
    echo "Creating the database cluster in $PGDATA"
    "$PGBIN/initdb" -D "$PGDATA" -U postgres --auth=trust >/dev/null
  fi

  if "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
    echo "Postgres already running on port $PGPORT"
  else
    echo "Starting Postgres on port $PGPORT"
    "$PGBIN/pg_ctl" -D "$PGDATA" -l "$PGLOG" -o "-p $PGPORT -k $DATA" -w start >/dev/null
  fi

  export PGHOST="$DATA" PGPORT="$PGPORT" PGUSER=postgres
  PSQL="$PGBIN/psql -v ON_ERROR_STOP=1 -q"

  if ! $PSQL -d postgres -tAc "select 1 from pg_database where datname='$DBNAME'" | grep -q 1; then
    echo "Creating database $DBNAME"
    "$PGBIN/createdb" "$DBNAME"
  fi

  # Roles PostgREST needs, and the three the schema grants to. `authenticator` is
  # the login role PostgREST connects as; it holds no rights of its own and only
  # switches into the role named by the JWT.
  echo "Ensuring roles"
  $PSQL -d "$DBNAME" <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='anon') THEN CREATE ROLE anon NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='authenticated') THEN CREATE ROLE authenticated NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='service_role') THEN CREATE ROLE service_role NOLOGIN BYPASSRLS; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='authenticator') THEN CREATE ROLE authenticator LOGIN NOINHERIT; END IF;
END
\$\$;
GRANT anon, authenticated, service_role TO authenticator;
SQL

  echo "Applying supabase/schema.sql into schema '$SCHEMA'"
  $PSQL -d "$DBNAME" -f "$ROOT/supabase/schema.sql" >/dev/null

  # The schema's grant block only fires for roles that exist, and it ran above, but
  # re-assert defaults so tables created later are reachable too.
  $PSQL -d "$DBNAME" <<SQL >/dev/null
ALTER DEFAULT PRIVILEGES IN SCHEMA $SCHEMA GRANT ALL ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA $SCHEMA GRANT ALL ON SEQUENCES TO service_role;
SQL

  if [[ ! -f "$DATA/jwt_secret" ]]; then
    python3 -c "import secrets; print(secrets.token_urlsafe(48))" > "$DATA/jwt_secret"
  fi
  JWT_SECRET="$(cat "$DATA/jwt_secret")"
  SERVICE_KEY="$(mint_jwt "$JWT_SECRET")"

  cat > "$DATA/postgrest.conf" <<CONF
db-uri = "postgres://authenticator@localhost:$PGPORT/$DBNAME?host=$DATA"
db-schemas = "$SCHEMA"
db-anon-role = "anon"
jwt-secret = "$JWT_SECRET"
server-port = $RESTPORT
CONF

  # Track our own process rather than trusting "something answered on that port":
  # an unrelated service on the port would otherwise look like a healthy PostgREST.
  if rest_running; then
    echo "PostgREST already running on port $RESTPORT"
  else
    if port_taken "$RESTPORT"; then
      echo "Port $RESTPORT is already in use by something else." >&2
      echo "Re-run with a free port:  LOCAL_REST_PORT=3222 bash scripts/local_stack.sh up" >&2
      exit 1
    fi
    echo "Starting PostgREST on port $RESTPORT"
    postgrest "$DATA/postgrest.conf" >"$RESTLOG" 2>&1 &
    echo $! > "$DATA/postgrest.pid"
    for _ in $(seq 1 60); do
      rest_running && break
      sleep 0.5
    done
    if ! rest_running; then
      echo "PostgREST did not come up. Last lines of $RESTLOG:" >&2
      tail -5 "$RESTLOG" >&2 || true
      exit 1
    fi
  fi

  cat > "$ENVFILE" <<ENVEOF
# Source this to point the app at the local stack:  source .localdb/env
export SUPABASE_URL=http://localhost:$RESTPORT
export SUPABASE_REST_PATH=
export SUPABASE_SERVICE_ROLE_KEY=$SERVICE_KEY
export SUPABASE_DB_SCHEMA=$SCHEMA
export SUPABASE_DB_URL="postgres://postgres@localhost:$PGPORT/$DBNAME?host=$DATA"
ENVEOF

  echo
  echo "Local stack is up."
  echo "  postgres : port $PGPORT, data in $PGDATA"
  echo "  postgrest: http://localhost:$RESTPORT (schema $SCHEMA)"
  echo
  echo "Use it:"
  echo "  source .localdb/env"
  echo "  python3 local_app.py --write"
}

cmd_down() {
  PGBIN="$(find_pg_bin)" || true
  if [[ -f "$DATA/postgrest.pid" ]] && kill "$(cat "$DATA/postgrest.pid")" 2>/dev/null; then
    rm -f "$DATA/postgrest.pid"; echo "Stopped PostgREST"
  else
    rm -f "$DATA/postgrest.pid"; echo "PostgREST not running"
  fi
  if [[ -n "${PGBIN:-}" ]] && "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
    "$PGBIN/pg_ctl" -D "$PGDATA" -m fast -w stop >/dev/null && echo "Stopped Postgres"
  else
    echo "Postgres not running"
  fi
}

cmd_status() {
  PGBIN="$(find_pg_bin)" || true
  if [[ -n "${PGBIN:-}" ]] && "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
    echo "postgres : running (port $PGPORT)"
  else
    echo "postgres : stopped"
  fi
  if rest_running; then
    echo "postgrest: running (http://localhost:$RESTPORT)"
  else
    echo "postgrest: stopped"
  fi
  if [[ -f "$ENVFILE" ]] && [[ -n "${PGBIN:-}" ]] && "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
    "$PGBIN/psql" -h "$DATA" -p "$PGPORT" -U postgres -d "$DBNAME" -tAc \
      "select 'knowledge base: '||(select count(*) from $SCHEMA.documents)||' documents, '||(select count(*) from $SCHEMA.document_chunks)||' chunks'" 2>/dev/null || true
  fi
}

cmd_reset() {
  cmd_down || true
  rm -rf "$DATA"
  echo "Deleted $DATA"
}

case "${1:-up}" in
  up) cmd_up ;;
  down) cmd_down ;;
  status) cmd_status ;;
  reset) cmd_reset ;;
  *) echo "usage: bash scripts/local_stack.sh [up|down|status|reset]" >&2; exit 1 ;;
esac
