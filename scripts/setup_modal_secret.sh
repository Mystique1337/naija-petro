#!/usr/bin/env bash
# Create/update the Modal secret bundle "naija-petro-secrets" from your local .env.
# Run from the repo root after filling in .env:  bash scripts/setup_modal_secret.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "No .env found at $ENV_FILE — copy .env.example to .env and fill it in." >&2
  exit 1
fi

# Pass only the keys the app needs at runtime (skip blanks).
KEYS=(SUPABASE_DB_URL SUPABASE_DB_SSL TAVILY_API_KEY HF_TOKEN \
      LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY LANGFUSE_HOST \
      MODEL_REPO EMBED_MODEL EMBED_DIM ENABLE_RERANK RERANK_MODEL \
      RAG_COVERAGE_THRESHOLD RAG_MIN_CHUNKS RAG_TOP_K RAG_FINAL_K \
      RAG_ALWAYS_ENRICH MAX_MODEL_LEN LLM_GPU EMBED_GPU)

ARGS=()
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a
for k in "${KEYS[@]}"; do
  v="${!k:-}"
  [[ -n "$v" ]] && ARGS+=("$k=$v")
done

echo "Creating Modal secret 'naija-petro-secrets' with ${#ARGS[@]} keys…"
modal secret create naija-petro-secrets "${ARGS[@]}" --force
echo "Done."
