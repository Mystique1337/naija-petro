"""Modal deployment for Naija-Petro.

Defines (top to bottom):
  * images + a shared HF-weights cache volume + the secret bundle
  * LLMService  — serves Shinzmann/naija-petro-8b with vLLM (OpenAI-compatible,
                  streaming) on a GPU. The OpenAI server runs on localhost inside
                  the container, so we get vLLM's native Qwen3 chat template +
                  streaming without depending on the churny engine Python API.
  * Encoders    — nomic embeddings + bge reranker on a small GPU   (added in wiring step)
  * fastapi_app — the web front door + background enrichment        (added in wiring step)

Run locally:   modal serve app/modal_app.py
Deploy:        modal deploy app/modal_app.py
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

# One secret bundle (created from .env — see README / scripts).
secrets = [modal.Secret.from_name("naija-petro-secrets")]

# --- vLLM serving image ---
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.11.0",
        "huggingface_hub[hf_transfer]>=0.27",
        "openai>=1.55",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_CACHE_DIR, "VLLM_USE_V1": "1"})
)

VLLM_PORT = 8000


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
        """Launch the vLLM OpenAI-compatible server on localhost and wait for ready."""
        import httpx
        from openai import AsyncOpenAI

        self.model_name = settings.model_repo
        cmd = [
            "vllm", "serve", self.model_name,
            "--port", str(VLLM_PORT),
            "--max-model-len", str(settings.max_model_len),
            "--gpu-memory-utilization", "0.90",
            "--enable-prefix-caching",
            "--disable-log-requests",
        ]
        self._proc = subprocess.Popen(cmd)

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
        """Stream assistant text deltas for the given chat messages."""
        s = sampling or {}
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            temperature=s.get("temperature", settings.temperature),
            top_p=s.get("top_p", settings.top_p),
            max_tokens=s.get("max_tokens", settings.max_new_tokens),
            extra_body={
                # Qwen3: keep answers direct (no chain-of-thought) and curb repetition.
                "chat_template_kwargs": {"enable_thinking": False},
                "repetition_penalty": s.get("repetition_penalty", 1.1),
            },
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @modal.method()
    async def complete(self, messages: list[dict], sampling: dict | None = None) -> str:
        """Non-streaming convenience (used by internal judges/utilities)."""
        s = sampling or {}
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
            temperature=s.get("temperature", settings.temperature),
            top_p=s.get("top_p", settings.top_p),
            max_tokens=s.get("max_tokens", settings.max_new_tokens),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content or ""
