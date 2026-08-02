"""Embedding + reranking model wrappers.

Pure (no Modal): instantiated inside the Modal GPU container (see app/modal_app.py
`Encoders`). nomic-embed-text-v1.5 needs task prefixes, `search_query:` for queries
and `search_document:` for stored passages, and L2-normalised vectors for cosine.
"""
from __future__ import annotations

QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "


class EmbeddingModel:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    def encode(self, texts: list[str], mode: str = "document", batch_size: int = 32) -> list[list[float]]:
        """mode: 'query' or 'document'. Returns L2-normalised float lists."""
        if isinstance(texts, str):
            texts = [texts]
        prefix = QUERY_PREFIX if mode == "query" else DOC_PREFIX
        prefixed = [prefix + (t or "") for t in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()


class Reranker:
    """Cross-encoder reranker via sentence-transformers (fast tokenizer, no FlagEmbedding/peft)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
