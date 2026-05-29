"""Live ingestion: search → fetch → clean markdown → chunk → embed → upsert.

Used both for on-demand query enrichment and for seeding the knowledge base.
`embed_fn` is injected (async) so the heavy embedding model can live in a Modal
GPU container while this code runs in the lightweight web container.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import date, datetime
from typing import Awaitable, Callable

import httpx

from app.config import settings
from app.rag import db
from app.rag.chunking import chunk_text
from app.rag.sources import PREFERRED_DOMAINS, classify, domain_of

EmbedFn = Callable[[list[str], str], Awaitable[list[list[float]]]]

TAVILY_URL = "https://api.tavily.com/search"
_WS = re.compile(r"[ \t]+")
_BLANK = re.compile(r"\n{3,}")


def _normalise(text: str) -> str:
    text = _WS.sub(" ", (text or "").strip())
    return _BLANK.sub("\n\n", text)


def _hash(text: str) -> str:
    return hashlib.sha256(_normalise(text).lower().encode("utf-8")).hexdigest()


def _parse_date(value) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%a, %d %b %Y %H:%M:%S %Z"):
        try:
            return datetime.strptime(str(value)[: len(fmt) + 4], fmt).date()
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
async def tavily_search(query: str, max_results: int = 6, include_domains: list[str] | None = None) -> list[dict]:
    if not settings.tavily_api_key:
        return []
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_raw_content": True,
        "include_answer": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    async with httpx.AsyncClient(timeout=40) as client:
        resp = await client.post(TAVILY_URL, json=payload)
        resp.raise_for_status()
        return resp.json().get("results", [])


# --------------------------------------------------------------------------- #
# Fetch + extract to markdown
# --------------------------------------------------------------------------- #
def _extract_html(html: str, url: str) -> str:
    import trafilatura

    md = trafilatura.extract(
        html, url=url, output_format="markdown",
        include_links=False, include_comments=False, favor_recall=True,
    )
    return md or ""


def _extract_pdf(data: bytes) -> str:
    import tempfile
    import pymupdf4llm

    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(data)
        f.flush()
        return pymupdf4llm.to_markdown(f.name) or ""


async def fetch_markdown(url: str) -> str:
    """Fetch a URL and return clean markdown (HTML via trafilatura, PDF via PyMuPDF)."""
    try:
        async with httpx.AsyncClient(timeout=40, follow_redirects=True,
                                     headers={"User-Agent": "Naija-Petro/0.1 (+research)"}) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "").lower()
            if "application/pdf" in ctype or url.lower().endswith(".pdf"):
                return await asyncio.to_thread(_extract_pdf, resp.content)
            return await asyncio.to_thread(_extract_html, resp.text, url)
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
async def _ingest_one(result: dict, embed_fn: EmbedFn) -> dict:
    url = result.get("url", "")
    if not url:
        return {"inserted": False, "chunk_count": 0, "url": url}

    # Prefer Tavily's raw_content; fall back to fetching + extracting ourselves.
    content = (result.get("raw_content") or "").strip()
    if len(content) < 500:
        fetched = await fetch_markdown(url)
        if len(fetched) > len(content):
            content = fetched
    content = _normalise(content)
    if len(content) < 200:  # too thin to be useful
        return {"inserted": False, "chunk_count": 0, "url": url}

    chash = _hash(content)
    if await db.document_exists(chash):
        return {"inserted": False, "chunk_count": 0, "url": url}

    label, tier = classify(url)
    chunks_txt = chunk_text(content, settings.chunk_chars, settings.chunk_overlap)
    if not chunks_txt:
        return {"inserted": False, "chunk_count": 0, "url": url}

    embeddings = await embed_fn(chunks_txt, "document")
    chunks = [
        {"content": t, "embedding": e, "token_count": len(t.split()),
         "metadata": {"chunk": i}}
        for i, (t, e) in enumerate(zip(chunks_txt, embeddings))
    ]
    doc = {
        "title": result.get("title") or label,
        "source_url": url,
        "domain": domain_of(url),
        "source_tier": tier,
        "published_date": _parse_date(result.get("published_date")),
        "content": content,
        "content_hash": chash,
        "metadata": {"source_label": label, "search_score": result.get("score")},
    }
    res = await db.upsert_document(doc, chunks)
    res["url"] = url
    return res


async def ingest_query(query: str, embed_fn: EmbedFn, max_results: int | None = None) -> dict:
    """Search for `query`, preferring authoritative Nigerian sources, and ingest."""
    max_results = max_results or settings.tavily_max_results
    results = await tavily_search(query, max_results=max_results, include_domains=PREFERRED_DOMAINS)
    # Also run an open-web pass so we are not limited to the curated domains.
    if len(results) < max_results:
        results += await tavily_search(query, max_results=max_results)

    # De-dup by URL before fetching.
    seen, unique = set(), []
    for r in results:
        u = r.get("url")
        if u and u not in seen:
            seen.add(u)
            unique.append(r)

    outcomes = await asyncio.gather(*(_ingest_one(r, embed_fn) for r in unique), return_exceptions=True)
    ok = [o for o in outcomes if isinstance(o, dict)]
    ingested = [o for o in ok if o.get("inserted")]
    return {
        "searched": len(unique),
        "ingested_docs": len(ingested),
        "new_chunks": sum(o.get("chunk_count", 0) for o in ingested),
        "urls": [o["url"] for o in ingested],
    }


async def ingest_urls(urls: list[str], embed_fn: EmbedFn) -> dict:
    """Directly ingest specific URLs (used by the KB seeder)."""
    results = [{"url": u} for u in urls]
    outcomes = await asyncio.gather(*(_ingest_one(r, embed_fn) for r in results), return_exceptions=True)
    ok = [o for o in outcomes if isinstance(o, dict) and o.get("inserted")]
    return {"ingested_docs": len(ok), "new_chunks": sum(o.get("chunk_count", 0) for o in ok)}
