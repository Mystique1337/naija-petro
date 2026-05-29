<div align="center">

# 🛢️ Naija-Petro

**Domain-specialised petroleum-engineering LLMs + a dynamic, citation-grounded RAG assistant for the Nigerian oil & gas context.**

[![Model (8B)](https://img.shields.io/badge/🤗%20Model-naija--petro--8b-yellow)](https://huggingface.co/Shinzmann/naija-petro-8b)
[![Model (32B)](https://img.shields.io/badge/🤗%20Model-naija--petro-yellow)](https://huggingface.co/Shinzmann/naija-petro)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

</div>

---

## What this is

Naija-Petro is a pair of [Qwen3](https://huggingface.co/Qwen) models fine-tuned (QLoRA, [Unsloth](https://github.com/unslothai/unsloth)) on **20,000+ synthetic petroleum-engineering instruction–response pairs**, plus the full pipeline that produced them and a production **retrieval-augmented assistant** that grounds answers in **verifiable Nigerian sources**.

The base models were trained on *general / global* petroleum knowledge — drilling, reservoir, production, completions, EOR, well testing, geoscience. They are strong on fundamentals but **weak on Nigeria-specific facts** (regulation, the Petroleum Industry Act 2021, NUPRC/NMDPRA, NNPC, local fields and fiscal terms). The RAG layer closes that gap: it fetches authoritative Nigerian resources on demand, converts them to clean markdown, cites them, and **continuously grows its own knowledge base** as it is used.

## Models

| Variant | Base | Use case | Links |
|---|---|---|---|
| **naija-petro-8b** | Qwen3-8B | Fast inference, free/low-cost deployment (used in the RAG app) | [model](https://huggingface.co/Shinzmann/naija-petro-8b) · [GGUF](https://huggingface.co/Shinzmann/naija-petro-8b-GGUF) |
| **naija-petro** (32B) | Qwen3-32B | Highest quality, GPU required | [model](https://huggingface.co/Shinzmann/naija-petro) · [GGUF](https://huggingface.co/Shinzmann/naija-petro-GGUF) |

## Repository layout

```
naija-petro/
├── notebooks/        Data + training pipeline (Colab/Jupyter)
│   ├── 01_corpus_builder.ipynb     Scrape & consolidate the seed corpus
│   ├── 02_data_designer.ipynb      NVIDIA Data Designer synthetic generation
│   ├── 03_eda.ipynb                Exploratory analysis of the 20K dataset
│   ├── 04_finetune_8b.ipynb        Fine-tune, evaluate & deploy the 8B model
│   └── 05_finetune_32b.ipynb       Fine-tune, evaluate & deploy the 32B model
├── modal_app.py      Modal entrypoint (vLLM serving, encoders, ASGI app, jobs)
├── app/              Dynamic-RAG assistant
│   ├── api/                        FastAPI routes (chat, SSE streaming, stats)
│   ├── rag/                        Embeddings, retrieval, ingestion, sources, prompts
│   ├── frontend/                   Streaming chat UI with a citations panel
│   ├── config.py                   Env-driven settings + system prompt
│   └── observability.py            Langfuse tracing (best-effort)
├── supabase/         pgvector schema + hybrid-search RPC
├── hf_cards/         Source-of-truth Hugging Face model & dataset cards
├── scripts/          Notebook scrubber, card pusher, KB seeder, Modal secret setup
├── .env.example      All configuration / secrets (copy to .env)
└── README.md
```

## The RAG assistant

A query flows through a **dynamic, self-updating** pipeline (full design in [`app/`](app/)):

1. Embed the query (`nomic-embed-text-v1.5`) and run **hybrid retrieval** over Supabase pgvector — dense vectors **+** Postgres full-text, fused with Reciprocal Rank Fusion, then reranked.
2. Score local **coverage**. If the knowledge base can't answer confidently, fetch live: **Tavily** search restricted to authoritative Nigerian domains → clean markdown (`trafilatura` / `pymupdf4llm`) → chunk → embed → **upsert** (SHA-256 dedup) → re-retrieve.
3. After **every** query, a non-blocking background job enriches the store, so the system keeps getting better with use.
4. The 8B model (served with **vLLM** on Modal, OpenAI-compatible) streams an answer with **inline citations**; sources are shown in a side panel.
5. Every step is traced to **Langfuse**.

### Stack

`Modal` (serverless GPU) · `vLLM` · `FastAPI` (SSE streaming) · self-hosted `Supabase` + `pgvector` · `nomic-embed-text-v1.5` · `Tavily` · `trafilatura` / `pymupdf4llm` · `Langfuse`

## Quick start

```bash
git clone https://github.com/Mystique1337/naija-petro.git
cd naija-petro
cp .env.example .env            # fill in your keys

# 1) Provision the vector store (run once against your Railway Postgres / Supabase)
psql "$SUPABASE_DB_URL" -f supabase/schema.sql

# 2) Push your .env into a Modal secret, then run the assistant
pip install -r requirements.txt && modal token new
bash scripts/setup_modal_secret.sh     # .env -> Modal secret "naija-petro-secrets"
modal serve modal_app.py               # dev URL (hot reload)
# modal deploy modal_app.py            # production URL

# 3) (optional) seed authoritative Nigerian docs once deployed
python scripts/seed_kb.py
```

See [`.env.example`](.env.example) for every setting and which credential each step needs.

## Dataset

20K+ synthetic instruction–response pairs generated with NVIDIA Data Designer from a scraped, de-duplicated petroleum corpus (arXiv, Semantic Scholar, OpenAlex, Crossref, DOE/OSTI, PetroWiki, SLB glossary, EIA, and more). Pipeline lives in `notebooks/01`–`02`; the dataset card is in [`hf_cards/dataset_card.md`](hf_cards/dataset_card.md).

## License

Apache-2.0, following the Qwen3 base models. Outputs are for research and educational support — **validate with qualified engineers before any operational decision.**
