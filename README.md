<div align="center">

# 🛢️ Naija-Petro

### A Nigeria-aware petroleum-engineering AI

Fine-tuned **Qwen3** models plus a dynamic, citation-grounded **RAG** assistant with exact engineering calculators.

<br>

[![Live demo](https://img.shields.io/badge/demo-live-16a34a?style=for-the-badge&logo=rocket&logoColor=white)](https://naija-petro.shinzii.tech)
[![Model 8B](https://img.shields.io/badge/🤗%20model-naija--petro--8b-yellow?style=for-the-badge)](https://huggingface.co/Shinzmann/naija-petro-8b)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)

<br>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Modal](https://img.shields.io/badge/Modal-serverless%20GPU-7B3FE4?logo=modal&logoColor=white)
![vLLM](https://img.shields.io/badge/vLLM-OpenAI%20API-FF6F00)
![Qwen3](https://img.shields.io/badge/Qwen3-8B%20%2F%2032B-615CED)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?logo=fastapi&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-pgvector-3ECF8E?logo=supabase&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗-models%20%2B%20spaces-yellow)
![Langfuse](https://img.shields.io/badge/Langfuse-tracing-0A0A0A)
![Tailwind](https://img.shields.io/badge/UI-Tailwind%20%2B%20Alpine-38BDF8?logo=tailwindcss&logoColor=white)
![Unsloth](https://img.shields.io/badge/Unsloth-QLoRA-FFB300)

[Live app](https://naija-petro.shinzii.tech) ·
[What it is](#what-this-is) ·
[Features](#features) ·
[Models](#models) ·
[Quick start](#quick-start) ·
[API](#api) ·
[Dataset](#dataset)

Built by **Ashinze Emmanuel** · [GitHub](https://github.com/Mystique1337/naija-petro) · [Hugging Face](https://huggingface.co/Shinzmann)

</div>

---

> [!NOTE]
> Naija-Petro produces **decision support** for research and education. Validate every figure with qualified
> engineers and primary sources before any operational decision.

## What this is

Naija-Petro is a pair of [Qwen3](https://huggingface.co/Qwen) models fine-tuned (QLoRA via [Unsloth](https://github.com/unslothai/unsloth)) on 20,000+ synthetic petroleum-engineering instruction-response pairs, plus the full data pipeline that produced them and a production retrieval-augmented assistant.

The base models learned **general, global** petroleum knowledge (drilling, reservoir, production, completions, EOR, well testing, geoscience). They are strong on fundamentals but weak on **Nigeria-specific** facts: regulation, the Petroleum Industry Act 2021, NUPRC, NMDPRA, NNPC, local fields, and fiscal terms. The RAG layer closes that gap. It fetches authoritative Nigerian sources on demand, converts them to clean markdown, cites them, and grows its own knowledge base as it is used. For numeric questions it calls deterministic calculators so the figures are exact, not estimated.

Live app: **https://naija-petro.shinzii.tech** (also at `https://peniel-tish--naija-petro.modal.run`)

## Features

- **Dynamic, self-updating RAG.** Hybrid retrieval (dense vectors plus Postgres full-text, fused with Reciprocal Rank Fusion). When local coverage is weak it fetches live from authoritative Nigerian sources, ingests them, and re-retrieves; a background job keeps growing the store after every query.
- **Verifiable citations.** Inline numbered citations and a sources panel with site favicons and trust tiers (official, reference, news).
- **Engineering calculators (tool calling).** For a computational question the model picks a calculator and the exact result is computed and shown: Arps decline, OOIP and OGIP volumetrics, Vogel IPR, Darcy radial inflow, hydrostatic pressure, Standing bubble point and Bo, gas FVF, productivity index, exponential EUR, recovery factor.
- **Reasoning trace** (optional toggle), **field/SI unit** toggle, **light and dark** themes, fully responsive.
- **Upload your own documents.** Drop in a PDF or text file and ask questions grounded in it.
- **Saved chat history** (anonymous, per browser), **copy and export**, and **streaming** answers with KaTeX math.
- **Feedback loop.** Thumbs and an optional comment store the full exchange in Supabase as training and preference data; `scripts/export_feedback.py` turns it into SFT and DPO files for fine-tuning.
- **Optional email capture** and a **feature-request board**.
- **Usage analytics** in Supabase and **Langfuse** tracing of every turn.
- **Cost controlled.** One GPU only (capped), CPU embeddings, fast scale-down, scale to zero when idle.

## Models

| Variant | Base | Use case | Links |
|---|---|---|---|
| **naija-petro-8b** | Qwen3-8B | Fast inference, low-cost deployment (served by the app) | [model](https://huggingface.co/Shinzmann/naija-petro-8b), [GGUF](https://huggingface.co/Shinzmann/naija-petro-8b-GGUF) |
| **naija-petro** (32B) | Qwen3-32B | Highest quality, needs a GPU | [model](https://huggingface.co/Shinzmann/naija-petro), [GGUF](https://huggingface.co/Shinzmann/naija-petro-GGUF) |

## How a query flows

1. Embed the query with `nomic-embed-text-v1.5` (on CPU) and run hybrid retrieval over Supabase pgvector (vectors plus full-text, fused with RRF, executed as a Postgres function over the REST API).
2. Score local coverage. If it is weak, fetch live: Tavily search biased to authoritative Nigerian domains, clean to markdown (`trafilatura` for HTML, `pymupdf4llm` for PDF), structure-aware chunking, embed, upsert with SHA-256 dedup, then re-retrieve. A background job also enriches after every query.
3. If the question is computational, a calculator is selected and the exact result is injected so the answer reports verified figures.
4. The 8B model, served with vLLM on Modal (OpenAI-compatible, streaming), answers with inline citations.
5. The turn is logged to Supabase analytics and traced to Langfuse.

### Stack

`Modal` (serverless GPU) · `vLLM` · `FastAPI` (SSE streaming) · self-hosted `Supabase` and `pgvector` over the REST API · `nomic-embed-text-v1.5` · `Tavily` · `trafilatura` / `pymupdf4llm` · `Langfuse` · `Tailwind` and `Alpine` frontend with `KaTeX`.

## Repository layout

```
naija-petro/
├── notebooks/        Data + training pipeline (Colab/Jupyter)
│   ├── 01_corpus_builder.ipynb     Scrape and consolidate the seed corpus
│   ├── 02_data_designer.ipynb      NVIDIA Data Designer synthetic generation
│   ├── 03_eda.ipynb                Exploratory analysis of the 20K dataset
│   ├── 04_finetune_8b.ipynb        Fine-tune, evaluate, deploy the 8B model
│   └── 05_finetune_32b.ipynb       Fine-tune, evaluate, deploy the 32B model
├── modal_app.py      Modal entrypoint (vLLM serving, encoders, ASGI app, jobs)
├── app/
│   ├── api/server.py               FastAPI routes (chat SSE, tools, upload, feedback, history)
│   ├── rag/                        embeddings, retrieval, ingestion, chunking, sources, prompts, db
│   ├── tools/calculators.py        deterministic engineering calculators
│   ├── frontend/index.html         streaming chat UI
│   ├── config.py                   env-driven settings + system prompt
│   └── observability.py            Langfuse tracing (best-effort)
├── supabase/
│   ├── schema.sql                  pgvector schema, hybrid_search RPC, analytics, feedback, history
│   └── migrations/                 incremental SQL applied to an existing database
├── hf_cards/             Hugging Face model and dataset cards
├── scripts/              scrub_notebooks, push_cards, seed_kb, setup_modal_secret, export_feedback
├── .env.example          all configuration and secrets (copy to .env)
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/Mystique1337/naija-petro.git
cd naija-petro
cp .env.example .env            # fill in your keys (see the file for each one)
pip install -r requirements.txt

# 1) Provision the vector store + analytics (run once)
#    Paste supabase/schema.sql into the Supabase Studio SQL editor, or:
psql "$SUPABASE_DB_URL" -f supabase/schema.sql

# 2) Authenticate Modal and push your .env into a Modal secret
modal token new
bash scripts/setup_modal_secret.sh     # .env -> Modal secret "naija-petro-secrets"

# 3) Run it
modal serve  modal_app.py              # dev URL with hot reload
modal deploy modal_app.py              # production URL

# 4) (optional) seed authoritative Nigerian docs, once deployed
python scripts/seed_kb.py
```

Notes:
- The running app reaches the store over the **Supabase REST (PostgREST) API**: `SUPABASE_URL` plus `SUPABASE_SERVICE_ROLE_KEY`, with every call pinned to `SUPABASE_DB_SCHEMA` through the `Accept-Profile` and `Content-Profile` headers. Vector search and the KB counts run as Postgres functions exposed as RPC. No database port needs to be reachable from Modal.
- `SUPABASE_DB_URL` is only used by the offline admin scripts (`scripts/seed_tokens.py`) and to apply SQL with `psql`. The app itself never opens a direct Postgres connection.
- One Supabase instance can host several apps: each gets its own schema (`SUPABASE_DB_SCHEMA`, default `naija_petro`). Functions in a non-public schema must pin their own `search_path`, which `schema.sql` does; otherwise PostgREST calls fail with `relation "documents" does not exist`.
- After changing `app/` code, a warm container can keep serving the old code. Force fresh containers with `modal app stop naija-petro --yes` then redeploy.
- `pgvector` must be enabled on the database; the schema runs `CREATE EXTENSION IF NOT EXISTS vector`.
- Modal deploys into the **active profile's workspace**. If you keep several profiles, run `modal profile list` and activate the right one before deploying, or you will create a second app under a different URL.

## Run locally

`local_app.py` runs the whole app on a laptop: no Modal, no cloud GPU. It builds the same FastAPI app as the deployment, but injects local implementations of the GPU work: an OpenAI-compatible server on `localhost` for the model, `sentence-transformers` on CPU for embeddings, and a background task for enrichment.

```bash
pip install -r requirements.txt -r requirements-local.txt

# 1) Serve the model. Ollama pulls the published GGUF straight from the Hub.
ollama pull hf.co/Shinzmann/naija-petro-8b-GGUF:Q4_K_M

# 2) Run the app
python local_app.py                    # http://127.0.0.1:8000
```

Working on the frontend? The fake modes need no model at all, so nothing is downloaded and nothing runs on the GPU:

```bash
python local_app.py --fake-llm         # canned markdown answer, real retrieval
python local_app.py --fake-embed       # deterministic hashed vectors, no encoder
python local_app.py --fake             # both
```

Other flags: `--host` (default `127.0.0.1`), `--port` (default `8000`), `--reload` to restart on code changes.

### With LM Studio

LM Studio works as well as Ollama and can serve the embedding model too, which skips the
550 MB `sentence-transformers` download entirely:

```bash
lms server start
lms load naija-petro-8b -c 8192          # context matters, see below
LOCAL_LLM_BASE_URL=http://localhost:1234/v1 \
LOCAL_LLM_MODEL=naija-petro-8b \
LOCAL_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5 \
python local_app.py
```

**Load the model with at least an 8k context.** `RAG_CONTEXT_CHARS` is 20000, so a grounded
prompt runs well past the 4096 tokens LM Studio defaults to, and the sources get truncated
before the model ever sees them.

| Variable | Purpose |
|---|---|
| `LOCAL_LLM_BASE_URL` | OpenAI-compatible endpoint (default `http://localhost:11434/v1`, which is Ollama; LM Studio is `http://localhost:1234/v1`) |
| `LOCAL_LLM_MODEL` | Model to request (default `hf.co/Shinzmann/naija-petro-8b-GGUF:Q4_K_M`) |
| `LOCAL_LLM_API_KEY` | Key for that endpoint (default `local`; most local servers ignore it) |
| `LOCAL_EMBED_MODEL` | Optional. Embedding model id on the same server. Set it to use the API instead of a local `sentence-transformers` copy |
| `LOCAL_EMBED_BASE_URL` | Optional. Defaults to `LOCAL_LLM_BASE_URL` |

The embedding model must be `nomic-embed-text-v1.5` at 768 dimensions to match the stored
vectors. A mismatch is reported loudly at first use rather than silently returning nonsense.

Notes:
- It loads the repo `.env` and reads and writes the **same Supabase over the REST API** as the deployed app, so there is no local database to run and no schema to apply. Documents you ingest locally land in the shared store. Without `SUPABASE_URL` the UI still loads, but retrieval and history fail; the startup summary says so.
- `--fake-embed` disables ingestion entirely, so placeholder vectors can never be written into the shared knowledge base.
- Any OpenAI-compatible server works: point `LOCAL_LLM_BASE_URL` at llama.cpp, LM Studio, or a local vLLM instead of Ollama. The 32B GGUF works the same way if you have the memory for it.
- The first real query downloads `nomic-embed-text-v1.5` (roughly 550 MB) and runs it on CPU, so it is slow once and fast afterwards. `--fake-embed` skips that entirely.

## Configuration

All settings live in `.env` (see [`.env.example`](.env.example)). The essentials:

| Variable | Purpose |
|---|---|
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | Supabase REST endpoint and service-role key. This is how the app reads and writes the vector store, analytics, and history. |
| `SUPABASE_DB_SCHEMA` | Postgres schema holding this app's tables and functions (default `naija_petro`). |
| `SUPABASE_DB_URL` | Direct Postgres URL. Only for applying SQL and the offline admin scripts, not for the running app. |
| `TAVILY_API_KEY` | Live web retrieval of Nigerian sources. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | Prompt and RAG tracing. |
| `HF_TOKEN` | Pushing the Hugging Face cards (`scripts/push_cards.py`). |
| `MODEL_REPO`, `LLM_GPU`, `LLM_SCALEDOWN_WINDOW` | Model, GPU type (default L4), and idle window (default short, for cost). |
| `ENABLE_RERANK` | Cross-encoder rerank, off by default (hybrid RRF is already strong). |
| `RAG_TOP_K`, `RAG_FINAL_K`, `RAG_CONTEXT_CHARS` | Retrieval breadth and the context budget cap. |
| `ACCESS_KEY`, `RATE_LIMIT_*` | Optional access gate and rate limiting. |

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Chat UI |
| `/chat` | POST | Streaming answer (Server-Sent Events) |
| `/tools`, `/tools/run` | GET, POST | List and run the engineering calculators |
| `/upload` | POST | Upload a PDF or text file into the knowledge base |
| `/feedback` | POST | Thumbs and comment (stored for training) |
| `/subscribe`, `/feature`, `/features` | POST/GET | Email capture and feature-request board |
| `/history`, `/history/{id}` | GET | Saved conversations (anonymous) |
| `/kb/stats`, `/healthz` | GET | Knowledge-base stats and health |

## Analytics

Query usage directly in Supabase:

```sql
SELECT * FROM usage_summary;   -- total queries, sessions, unique users, distinct IPs
SELECT * FROM usage_daily;     -- per-day queries, latency, coverage, docs added
SELECT * FROM feedback;        -- ratings + the full exchange for model improvement
```

## Cost

The app runs on one L4 GPU (capped to a single container), embeddings on CPU, a short idle window, and scale to zero when idle, so you pay only while it is actively used. The first request after idle cold-starts the model (roughly one to two minutes, weights cached), then it is fast. Set a hard spending limit in the Modal dashboard for peace of mind.

## Dataset

20,000+ synthetic instruction-response pairs generated with NVIDIA Data Designer from a scraped, de-duplicated petroleum corpus (arXiv, Semantic Scholar, OpenAlex, Crossref, DOE/OSTI, PetroWiki, the SLB glossary, EIA, and more). The pipeline is in `notebooks/01` and `notebooks/02`; the dataset card is in [`hf_cards/dataset_card.md`](hf_cards/dataset_card.md).

## License

Apache-2.0, following the Qwen3 base models. Outputs are decision support for research and education. Validate with qualified engineers and primary sources before any operational decision.
