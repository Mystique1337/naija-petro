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
#   bash scripts/local_stack.sh down     # stop both services, keep ALL data
#   bash scripts/local_stack.sh reset    # DESTRUCTIVE: stop and delete the data directory
#
# Environment overrides:
#   LOCAL_PG_PORT    Postgres port          (default 55432)
#   LOCAL_REST_PORT  PostgREST port         (default 3111)
#   LOCAL_PG_BIN     Postgres bin directory (default: chosen deterministically, see below)
#   LOCAL_STACK_YES  set to 1 to skip the confirmation prompt on `reset`
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
CONFFILE="$DATA/postgrest.conf"
SECRETFILE="$DATA/jwt_secret"

BREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing '$1'. Run: brew install $2" >&2; exit 1; }
}

# --------------------------------------------------------------------------- #
# Choosing the Postgres binaries
#
# pgvector is compiled against ONE Postgres major version at a time, so picking a
# different keg than the one pgvector was built for fails later with the useless
# 'extension "vector" is not available'. Worse, starting an existing cluster with
# the wrong major fails with an incompatible-catalog error. So the choice is made
# explicitly rather than by whatever the shell glob happens to expand first:
#
#   1. $LOCAL_PG_BIN wins if set.
#   2. If a cluster already exists, its own major version wins (nothing else can
#      read those files).
#   3. Otherwise the highest installed major that actually has pgvector wins.
# --------------------------------------------------------------------------- #

# Emit "major<TAB>bindir" for every usable Postgres install, versioned kegs first.
pg_candidates() {
  local d ver
  for d in "$BREW_PREFIX"/opt/postgresql@*/bin; do
    [[ -x "$d/initdb" && -x "$d/pg_ctl" && -x "$d/psql" ]] || continue
    ver="${d#*postgresql@}"
    ver="${ver%%/*}"
    [[ "$ver" =~ ^[0-9]+$ ]] || continue
    printf '%s\t%s\n' "$ver" "$d"
  done
  d="$BREW_PREFIX/bin"
  if [[ -x "$d/initdb" && -x "$d/pg_ctl" && -x "$d/psql" ]]; then
    ver="$("$d/pg_ctl" --version 2>/dev/null | tr -dc '0-9. ' | tr ' ' '\n' | tail -1)"
    ver="${ver%%.*}"
    [[ "$ver" =~ ^[0-9]+$ ]] && printf '%s\t%s\n' "$ver" "$d"
  fi
  return 0
}

# True when pgvector is installed for the Postgres that owns this bin directory.
pg_has_vector() {
  local bin="$1" sharedir
  [[ -x "$bin/pg_config" ]] || return 1
  sharedir="$("$bin/pg_config" --sharedir 2>/dev/null || true)"
  [[ -n "$sharedir" && -f "$sharedir/extension/vector.control" ]]
}

pg_bin_version() {
  local bin="$1" ver
  ver="$("$bin/pg_ctl" --version 2>/dev/null | tr -dc '0-9. ' | tr ' ' '\n' | tail -1)"
  printf '%s\n' "${ver%%.*}"
}

no_postgres_found() {
  echo "No usable Postgres found under $BREW_PREFIX." >&2
  echo "Install one together with pgvector:" >&2
  echo "  brew install postgresql@17 pgvector" >&2
  echo "Or point the script at an existing install: LOCAL_PG_BIN=/path/to/bin $0 $*" >&2
  exit 1
}

find_pg_bin() {
  local want="" ver dir sorted

  # 1. Explicit override.
  if [[ -n "${LOCAL_PG_BIN:-}" ]]; then
    if [[ -x "$LOCAL_PG_BIN/initdb" && -x "$LOCAL_PG_BIN/pg_ctl" && -x "$LOCAL_PG_BIN/psql" ]]; then
      printf '%s\n' "$LOCAL_PG_BIN"
      return 0
    fi
    echo "LOCAL_PG_BIN=$LOCAL_PG_BIN does not contain initdb, pg_ctl and psql." >&2
    exit 1
  fi

  sorted="$(pg_candidates | sort -s -t"$(printf '\t')" -k1,1nr)"
  [[ -n "$sorted" ]] || no_postgres_found "$@"

  # 2. An existing cluster pins the major version: only that major can read it.
  if [[ -f "$PGDATA/PG_VERSION" ]]; then
    want="$(tr -d '[:space:]' < "$PGDATA/PG_VERSION")"
    want="${want%%.*}"
    while IFS="$(printf '\t')" read -r ver dir; do
      [[ -n "$dir" ]] || continue
      if [[ "$ver" == "$want" ]]; then printf '%s\n' "$dir"; return 0; fi
    done <<< "$sorted"
    echo "The cluster in $PGDATA was created by Postgres $want, which is not installed." >&2
    echo "Installed majors: $(printf '%s\n' "$sorted" | cut -f1 | tr '\n' ' ')" >&2
    echo "Either install it (brew install postgresql@$want) or, if you do not need the" >&2
    echo "existing data, delete it: bash scripts/local_stack.sh reset" >&2
    exit 1
  fi

  # 3. Fresh install: highest major that actually has pgvector.
  while IFS="$(printf '\t')" read -r ver dir; do
    [[ -n "$dir" ]] || continue
    if pg_has_vector "$dir"; then printf '%s\n' "$dir"; return 0; fi
  done <<< "$sorted"

  echo "Found Postgres but none of these installs has pgvector:" >&2
  printf '%s\n' "$sorted" | while IFS="$(printf '\t')" read -r ver dir; do
    [[ -n "$dir" ]] && echo "  postgres $ver  $dir" >&2
  done
  echo "pgvector is compiled per Postgres major version. Install it:" >&2
  echo "  brew install pgvector" >&2
  echo "then re-run: bash scripts/local_stack.sh up" >&2
  exit 1
}

port_taken() {
  command -v lsof >/dev/null 2>&1 || return 1
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

# The port a live postmaster in $PGDATA is actually serving (line 4 of postmaster.pid).
pg_actual_port() {
  [[ -f "$PGDATA/postmaster.pid" ]] || return 1
  awk 'NR==4 {gsub(/[^0-9]/, ""); print; exit}' "$PGDATA/postmaster.pid"
}

pg_running() {
  [[ -n "${PGBIN:-}" ]] && "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1
}

# The pid of OUR PostgREST, or non-zero. Checked against the process name too, so a
# recycled pid belonging to something else is never signalled.
rest_pid() {
  local pidfile="$DATA/postgrest.pid" pid comm
  [[ -f "$pidfile" ]] || return 1
  pid="$(tr -dc '0-9' < "$pidfile" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  comm="$(ps -o comm= -p "$pid" 2>/dev/null || true)"
  case "$comm" in
    *postgrest*) printf '%s\n' "$pid" ;;
    *) return 1 ;;
  esac
}

rest_running() {
  rest_pid >/dev/null || return 1
  curl -s -m 2 -o /dev/null "http://localhost:$RESTPORT/"
}

stop_rest() {
  local pid
  if ! pid="$(rest_pid)"; then
    rm -f "$DATA/postgrest.pid"
    return 1
  fi
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 40); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  rm -f "$DATA/postgrest.pid"
  return 0
}

mint_jwt() {
  # PostgREST authorises by JWT role claim. Supabase's service key is exactly this,
  # so minting one locally keeps the app's auth headers unchanged. Stdlib only.
  # The secret travels in the environment, never in argv, so it does not show up in
  # `ps` output for other users on this machine.
  NP_JWT_SECRET="$1" python3 - <<'PY'
import base64, hashlib, hmac, json, os

def b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

secret = os.environ["NP_JWT_SECRET"].encode()
header = b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
# No exp: this is a local development token for a database on loopback.
payload = b64(json.dumps({"role": "service_role", "iss": "naija-petro-local"},
                         separators=(",", ":")).encode())
signing_input = f"{header}.{payload}".encode()
sig = b64(hmac.new(secret, signing_input, hashlib.sha256).digest())
print(f"{header}.{payload}.{sig}")
PY
}

check_gitignored() {
  # .localdb/ holds a service-role JWT, the signing secret and the whole cluster.
  if ! grep -qE '^[[:space:]]*\.localdb/?[[:space:]]*$' "$ROOT/.gitignore" 2>/dev/null; then
    echo "WARNING: '.localdb/' is not listed in $ROOT/.gitignore." >&2
    echo "         It holds a service-role JWT and its signing secret. Add it before" >&2
    echo "         committing anything." >&2
  fi
}

cmd_up() {
  require brew brew
  require python3 python
  require curl curl
  PGBIN="$(find_pg_bin up)"
  require postgrest postgrest

  check_gitignored

  mkdir -p "$DATA"
  chmod 700 "$DATA" 2>/dev/null || true

  if [[ ! -d "$PGDATA" ]]; then
    echo "Creating the database cluster in $PGDATA (postgres $(pg_bin_version "$PGBIN"))"
    "$PGBIN/initdb" -D "$PGDATA" -U postgres --auth=trust >/dev/null
  fi

  if pg_running; then
    actual="$(pg_actual_port || true)"
    if [[ -n "$actual" && "$actual" != "$PGPORT" ]]; then
      echo "Postgres from $PGDATA is already running, but on port $actual, not $PGPORT." >&2
      echo "Either adopt it:   LOCAL_PG_PORT=$actual bash scripts/local_stack.sh up" >&2
      echo "or stop it first:  bash scripts/local_stack.sh down" >&2
      exit 1
    fi
    echo "Postgres already running on port $PGPORT"
  else
    echo "Starting Postgres on port $PGPORT"
    "$PGBIN/pg_ctl" -D "$PGDATA" -l "$PGLOG" -o "-p $PGPORT -k $DATA" -w start >/dev/null
  fi
  chmod 600 "$PGLOG" 2>/dev/null || true

  export PGHOST="$DATA" PGPORT="$PGPORT" PGUSER=postgres
  PSQL="$PGBIN/psql -v ON_ERROR_STOP=1 -q"

  # pgvector must be visible to THIS server before the schema is applied, otherwise
  # supabase/schema.sql fails halfway with a bare 'extension "vector" is not available'
  # and leaves a partly built schema behind.
  if ! $PSQL -d postgres -tAc \
        "select 1 from pg_available_extensions where name='vector'" | grep -q 1; then
    echo "The 'vector' extension is not available to the Postgres in $PGBIN." >&2
    echo "pgvector is compiled per Postgres major version, and this one (major" >&2
    echo "$(pg_bin_version "$PGBIN")) does not have it. Install it and re-run:" >&2
    echo "  brew install pgvector" >&2
    echo "  bash scripts/local_stack.sh up" >&2
    exit 1
  fi

  if ! $PSQL -d postgres -tAc "select 1 from pg_database where datname='$DBNAME'" | grep -q 1; then
    echo "Creating database $DBNAME"
    "$PGBIN/createdb" "$DBNAME"
  fi

  # Roles PostgREST needs, and the three the schema grants to. `authenticator` is
  # the login role PostgREST connects as; it holds no rights of its own and only
  # switches into the role named by the JWT. Every statement below is a no-op on a
  # second run: the CREATE ROLEs are guarded, and re-granting an existing role
  # membership does nothing.
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
  # re-assert defaults so tables created later are reachable too. ALTER DEFAULT
  # PRIVILEGES is declarative, so running it again changes nothing.
  $PSQL -d "$DBNAME" <<SQL >/dev/null
ALTER DEFAULT PRIVILEGES IN SCHEMA $SCHEMA GRANT ALL ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA $SCHEMA GRANT ALL ON SEQUENCES TO service_role;
SQL

  # Secrets: created under a restrictive umask so there is no window in which they
  # are world readable, then pinned to 0600.
  if [[ ! -f "$SECRETFILE" ]]; then
    ( umask 077; python3 -c "import secrets; print(secrets.token_urlsafe(48))" > "$SECRETFILE" )
  fi
  chmod 600 "$SECRETFILE"
  JWT_SECRET="$(cat "$SECRETFILE")"
  SERVICE_KEY="$(mint_jwt "$JWT_SECRET")"

  # postgrest.conf carries the signing secret, so it is 0600 too. If the config
  # changed (a different port, a regenerated secret) a PostgREST that is already
  # running is still serving the old one, so stop it and let it restart below.
  NEW_CONF="db-uri = \"postgres://authenticator@localhost:$PGPORT/$DBNAME?host=$DATA\"
db-schemas = \"$SCHEMA\"
db-anon-role = \"anon\"
jwt-secret = \"$JWT_SECRET\"
server-port = $RESTPORT"

  if [[ ! -f "$CONFFILE" ]] || [[ "$NEW_CONF" != "$(cat "$CONFFILE")" ]]; then
    if rest_pid >/dev/null; then
      echo "PostgREST config changed; restarting it"
      stop_rest >/dev/null || true
    fi
    ( umask 077; printf '%s\n' "$NEW_CONF" > "$CONFFILE" )
  fi
  chmod 600 "$CONFFILE"

  # Track our own process rather than trusting "something answered on that port":
  # an unrelated service on the port would otherwise look like a healthy PostgREST.
  if rest_running; then
    echo "PostgREST already running on port $RESTPORT"
  else
    # A recorded pid that is alive but not answering on $RESTPORT is a stale
    # PostgREST of ours (usually from a run with a different LOCAL_REST_PORT).
    # Stop it instead of orphaning it behind a new pid file.
    if rest_pid >/dev/null; then
      echo "Stopping a stale PostgREST that is not answering on port $RESTPORT"
      stop_rest >/dev/null || true
    fi
    if port_taken "$RESTPORT"; then
      echo "Port $RESTPORT is already in use by something else." >&2
      echo "Re-run with a free port:  LOCAL_REST_PORT=3222 bash scripts/local_stack.sh up" >&2
      exit 1
    fi
    echo "Starting PostgREST on port $RESTPORT"
    nohup postgrest "$CONFFILE" >"$RESTLOG" 2>&1 &
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

  # The env file carries a service-role JWT: same treatment as the secret above.
  ( umask 077; cat > "$ENVFILE" <<ENVEOF
# Source this to point the app at the local stack:  source .localdb/env
# Contains a service-role JWT. Keep it 0600 and inside .localdb/ (gitignored).
export SUPABASE_URL=http://localhost:$RESTPORT
export SUPABASE_REST_PATH=
export SUPABASE_SERVICE_ROLE_KEY=$SERVICE_KEY
export SUPABASE_DB_SCHEMA=$SCHEMA
export SUPABASE_DB_URL="postgres://postgres@localhost:$PGPORT/$DBNAME?host=$DATA"
ENVEOF
  )
  chmod 600 "$ENVFILE"

  echo
  echo "Local stack is up."
  echo "  postgres : port $PGPORT, data in $PGDATA (postgres $(pg_bin_version "$PGBIN"))"
  echo "  postgrest: http://localhost:$RESTPORT (schema $SCHEMA)"
  echo "  secrets  : $ENVFILE and $SECRETFILE, mode 0600, not printed here"
  echo
  echo "Use it:"
  echo "  source .localdb/env"
  echo "  python3 local_app.py --write"
}

# Stops both services. Never deletes anything: the cluster, the JWT secret and the
# generated env file all survive, so `up` afterwards resumes with the same data and
# the same service key.
cmd_down() {
  local pid from_reset="${1:-}"
  PGBIN="$(find_pg_bin down 2>/dev/null || true)"
  if pid="$(rest_pid 2>/dev/null)"; then
    stop_rest >/dev/null || true
    echo "Stopped PostgREST (pid $pid)"
  else
    rm -f "$DATA/postgrest.pid"
    echo "PostgREST not running"
  fi
  if pg_running; then
    "$PGBIN/pg_ctl" -D "$PGDATA" -m fast -w stop >/dev/null && echo "Stopped Postgres"
  else
    echo "Postgres not running"
  fi
  [[ "$from_reset" == "--from-reset" ]] || \
    echo "Data kept in $PGDATA. Use 'reset' to delete it."
}

cmd_status() {
  PGBIN="$(find_pg_bin status 2>/dev/null || true)"
  local running=0 actual=""
  if pg_running; then
    running=1
    actual="$(pg_actual_port || true)"
    if [[ -n "$actual" && "$actual" != "$PGPORT" ]]; then
      echo "postgres : running (port $actual, but LOCAL_PG_PORT says $PGPORT)"
    else
      echo "postgres : running (port ${actual:-$PGPORT})"
    fi
  else
    echo "postgres : stopped"
  fi
  if rest_running; then
    echo "postgrest: running (http://localhost:$RESTPORT)"
  else
    echo "postgrest: stopped"
  fi
  if [[ -f "$ENVFILE" && "$running" == 1 ]]; then
    "$PGBIN/psql" -h "$DATA" -p "${actual:-$PGPORT}" -U postgres -d "$DBNAME" -tAc \
      "select 'knowledge base: '||(select count(*) from $SCHEMA.documents)||' documents, '||(select count(*) from $SCHEMA.document_chunks)||' chunks'" 2>/dev/null || true
  fi
}

# DESTRUCTIVE. Deletes the cluster, the JWT signing secret and the generated env
# file. `down` is the non-destructive counterpart.
cmd_reset() {
  local confirm="${1:-}" reply=""
  echo "reset is DESTRUCTIVE. It permanently deletes:"
  echo "  - the Postgres cluster in $PGDATA (every document, chunk, feedback and token row)"
  echo "  - the JWT signing secret in $SECRETFILE, so every key minted from it stops working"
  echo "  - the generated $ENVFILE and $CONFFILE"
  echo "It does NOT touch the deployed app or its Supabase."
  echo "To stop the stack without losing data, use: bash scripts/local_stack.sh down"
  echo

  if [[ "${LOCAL_STACK_YES:-}" != "1" && "$confirm" != "--yes" ]]; then
    printf 'Type DELETE to confirm: '
    read -r reply || reply=""
    if [[ "$reply" != "DELETE" ]]; then
      echo "Aborted. Nothing was deleted."
      return 1
    fi
  fi

  # Belt and braces: only ever remove the repo's own .localdb directory.
  case "$DATA" in
    */.localdb) : ;;
    *) echo "Refusing to delete '$DATA': not a .localdb directory." >&2; exit 1 ;;
  esac

  cmd_down --from-reset || true
  rm -rf "$DATA"
  echo "Deleted $DATA"
}

case "${1:-up}" in
  up) cmd_up ;;
  down) cmd_down ;;
  status) cmd_status ;;
  reset) cmd_reset "${2:-}" ;;
  *) echo "usage: bash scripts/local_stack.sh [up|down|status|reset [--yes]]" >&2; exit 1 ;;
esac
