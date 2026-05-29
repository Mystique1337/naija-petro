"""Modal deployment for Naija-Petro — the single entrypoint that wires everything.

    modal serve  modal_app.py        # local dev (hot-reload, ephemeral URL)
    modal deploy modal_app.py        # production
    modal run    modal_app.py::seed  # seed the KB with authoritative Nigerian docs

Components:
  * LLMService  — vLLM OpenAI-compatible server for naija-petro-8b (A10G)
  * Encoders    — nomic embeddings + bge reranker (small GPU)
  * enrich()    — background ingestion job (the "self-update after every query")
  * fastapi_app — the web front door (ASGI), with GPU work injected as deps
  * seed()      — one-off knowledge-base seeding
"""
from __future__ import annotations

import os
import subprocess
import time

import modal

from app.config import APP_NAME, settings

app = modal.App(APP_NAME)

# Persisted Hugging Face cache so weights download once across cold starts.
hf_cache = modal.Volume.from_name("naija-petro-hf-cache", create_if_missing=True)
HF_CACHE_DIR = "/root/.cache/huggingface"
FRONTEND_REMOTE = "/assets/frontend"

# One secret bundle, created from .env (see scripts/setup_modal_secret.sh).
secrets = [modal.Secret.from_name("naija-petro-secrets")]

EMBED_GPU = os.environ.get("EMBED_GPU", "L4")
VLLM_PORT = 8000


def _strip_dashes(text: str) -> str:
    """Remove em/en-dashes from model output (user preference)."""
    return text.replace("—", "-").replace("–", "-")

# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.11.0", "huggingface_hub[hf_transfer]>=0.27", "openai>=1.55")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_CACHE_DIR, "VLLM_USE_V1": "1"})
    .add_local_python_source("app")
)

encoders_image = (
    modal.Image.debian_slim(python_version="3.11")
    # nomic embeddings + bge reranker both via sentence-transformers (CrossEncoder),
    # so no FlagEmbedding/peft and no transformers pin needed.
    .pip_install("sentence-transformers>=3.0", "einops", "huggingface_hub>=0.27")
    .env({"HF_HOME": HF_CACHE_DIR})
    .add_local_python_source("app")
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]>=0.115", "asyncpg>=0.29", "pgvector>=0.3", "httpx>=0.27",
        "trafilatura>=1.12", "pymupdf4llm>=0.0.17", "langfuse>=4.0", "openai>=1.55",
    )
    .env({"FRONTEND_DIR": FRONTEND_REMOTE})
    .add_local_python_source("app")
    .add_local_dir("app/frontend", remote_path=FRONTEND_REMOTE)
)


# --------------------------------------------------------------------------- #
# LLM — vLLM OpenAI-compatible server (localhost inside the GPU container)
# --------------------------------------------------------------------------- #
@app.cls(
    image=vllm_image,
    gpu=settings.llm_gpu,
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=secrets,
    scaledown_window=settings.llm_scaledown_window,
    timeout=20 * 60,
)
@modal.concurrent(max_inputs=16)
class LLMService:
    @modal.enter()
    def start(self):
        import httpx
        from openai import AsyncOpenAI

        self.model_name = settings.model_repo
        # Minimal, stable flags (vLLM 0.22 removed --disable-log-requests; prefix
        # caching is on by default in the V1 engine).
        self._proc = subprocess.Popen([
            "vllm", "serve", self.model_name,
            "--port", str(VLLM_PORT),
            "--max-model-len", str(settings.max_model_len),
            "--gpu-memory-utilization", "0.90",
        ])
        base = f"http://localhost:{VLLM_PORT}"
        deadline = time.time() + 15 * 60
        while time.time() < deadline:
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            if self._proc.poll() is not None:
                raise RuntimeError("vLLM server exited during startup")
            time.sleep(2)
        else:
            raise TimeoutError("vLLM server did not become ready in time")
        self.client = AsyncOpenAI(base_url=f"{base}/v1", api_key="local")

    @modal.exit()
    def stop(self):
        if getattr(self, "_proc", None) is not None:
            self._proc.terminate()

    @modal.method()
    async def chat_stream(self, messages: list[dict], sampling: dict | None = None):
        s = sampling or {}
        # reasoning on -> Qwen3 emits <think>...</think> before the answer (the UI
        # renders it as a collapsible reasoning trace).
        think = bool(s.get("reasoning", True))
        stream = await self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=True,
            temperature=s.get("temperature", settings.temperature),
            top_p=s.get("top_p", settings.top_p),
            max_tokens=s.get("max_tokens", settings.max_new_tokens),
            extra_body={"chat_template_kwargs": {"enable_thinking": think},
                        "repetition_penalty": s.get("repetition_penalty", 1.1)},
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield _strip_dashes(chunk.choices[0].delta.content)

    @modal.method()
    async def complete(self, messages: list[dict], sampling: dict | None = None) -> str:
        s = sampling or {}
        resp = await self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=False,
            temperature=s.get("temperature", settings.temperature),
            top_p=s.get("top_p", settings.top_p),
            max_tokens=s.get("max_tokens", settings.max_new_tokens),
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(s.get("reasoning", False))}},
        )
        return _strip_dashes(resp.choices[0].message.content or "")


# --------------------------------------------------------------------------- #
# Encoders — embeddings + reranker
# --------------------------------------------------------------------------- #
@app.cls(
    image=encoders_image,
    gpu=EMBED_GPU,
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=secrets,
    scaledown_window=settings.llm_scaledown_window,
    timeout=10 * 60,
)
@modal.concurrent(max_inputs=32)
class Encoders:
    @modal.enter()
    def start(self):
        from app.rag.embeddings import EmbeddingModel, Reranker

        self.embedder = EmbeddingModel(settings.embed_model)
        self.reranker = Reranker(settings.rerank_model) if settings.enable_rerank else None

    @modal.method()
    def embed(self, texts: list[str], mode: str = "document") -> list[list[float]]:
        return self.embedder.encode(texts, mode)

    @modal.method()
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if self.reranker is None:
            return []
        return self.reranker.score(query, passages)


# --------------------------------------------------------------------------- #
# Dependency wiring (GPU work -> async callables for the web layer)
# --------------------------------------------------------------------------- #
async def _embed(texts: list[str], mode: str) -> list[list[float]]:
    return await Encoders().embed.remote.aio(texts, mode)


async def _rerank(query: str, passages: list[str]) -> list[float]:
    return await Encoders().rerank.remote.aio(query, passages)


async def _llm_stream(messages: list[dict], sampling: dict):
    async for tok in LLMService().chat_stream.remote_gen.aio(messages, sampling):
        yield tok


async def _llm_complete(messages: list[dict], sampling: dict) -> str:
    return await LLMService().complete.remote.aio(messages, sampling)


def _spawn_enrich(query: str) -> None:
    enrich.spawn(query)


# --------------------------------------------------------------------------- #
# Background enrichment — the self-updating step
# --------------------------------------------------------------------------- #
@app.function(image=web_image, secrets=secrets, timeout=10 * 60)
async def enrich(query: str) -> dict:
    from app.rag import ingest

    return await ingest.ingest_query(query, _embed)


# --------------------------------------------------------------------------- #
# Web front door
# --------------------------------------------------------------------------- #
@app.function(image=web_image, secrets=secrets, scaledown_window=300, timeout=900)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="naija-petro")
def fastapi_app():
    from app.api.server import Deps, create_app

    deps = Deps(embed=_embed, llm_stream=_llm_stream, rerank=_rerank,
                spawn_enrich=_spawn_enrich, llm_complete=_llm_complete)
    return create_app(deps)


# --------------------------------------------------------------------------- #
# KB seeding
# --------------------------------------------------------------------------- #
SEED_URLS = [
    # Petroleum Industry Act 2021 (canonical, searchable PDF)
    "https://ngfcp.nuprc.gov.ng/wp-content/uploads/2022/09/Petroleum-Industry-Act-2021-pdf-searchable.pdf",
    # EIA Nigeria country analysis
    "https://www.eia.gov/international/analysis/country/NGA",
    # Regulators / NOC
    "https://www.nuprc.gov.ng/",
    "https://www.nmdpra.gov.ng/",
    "https://nnpcgroup.com/",
    # Transparency
    "https://neiti.gov.ng/reports",
]


@app.function(image=web_image, secrets=secrets, timeout=30 * 60)
async def seed(urls: list[str] | None = None) -> dict:
    from app.rag import ingest

    return await ingest.ingest_urls(urls or SEED_URLS, _embed)
