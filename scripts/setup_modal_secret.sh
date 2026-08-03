#!/usr/bin/env bash
# Create/update the Modal secret bundle "naija-petro-secrets" from your local .env.
# Uses Modal's own dotenv parser (--from-dotenv), which is robust to passwords /
# connection strings containing shell-special characters.
#
# The whole .env is uploaded, so it prints variable NAMES only and never a value.
#
# Run from the repo root after filling in .env:  bash scripts/setup_modal_secret.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/.env"
SECRET_NAME="naija-petro-secrets"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "No .env found at $ENV_FILE. Copy .env.example to .env and fill it in." >&2
  exit 1
fi

if ! command -v modal >/dev/null 2>&1; then
  echo "The 'modal' CLI is not on PATH. Install it and authenticate:" >&2
  echo "  pip install modal && modal token new" >&2
  exit 1
fi

# A .env readable by other users on this machine is a problem regardless of Modal.
PERMS="$(stat -f '%Lp' "$ENV_FILE" 2>/dev/null || stat -c '%a' "$ENV_FILE" 2>/dev/null || echo '')"
case "$PERMS" in
  600|400|"") : ;;
  *) echo "warning: $ENV_FILE is mode $PERMS. Tighten it:  chmod 600 $ENV_FILE" >&2 ;;
esac

# Names only. Values are never echoed: they go straight from the file to Modal.
echo "Uploading these variables from .env to Modal secret '$SECRET_NAME' (names only, no values):"
while IFS= read -r name; do
  [[ -n "$name" ]] && echo "  $name"
done < <(grep -oE '^[[:space:]]*(export[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=' "$ENV_FILE" \
         | sed -E 's/^[[:space:]]*(export[[:space:]]+)?//; s/[[:space:]]*=$//' | sort -u)

echo
echo "Creating/updating Modal secret '$SECRET_NAME'..."
modal secret create "$SECRET_NAME" --from-dotenv "$ENV_FILE" --force
echo "Done."
