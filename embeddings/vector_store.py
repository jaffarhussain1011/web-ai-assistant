"""
embeddings/vector_store.py
──────────────────────────
Manages FAISS vector index and the matching text chunks.

Responsibilities:
  - Chunk long documents into overlapping windows
  - Generate sentence-transformer embeddings
  - Build / save / load a FAISS flat-L2 index
  - Cosine-similarity search (via inner-product on normalised vectors)
  - Cache frequent queries in memory
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.parent
VECTORS_DIR = _HERE / "data" / "vectors"
INDEX_FILE   = VECTORS_DIR / "faiss.index"
CHUNKS_FILE  = VECTORS_DIR / "chunks.json"
META_FILE    = VECTORS_DIR / "meta.json"

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "all-MiniLM-L6-v2"   # ~22 MB, fast, 384-dim
CHUNK_SIZE     = 400    # characters per chunk
CHUNK_OVERLAP  = 80     # overlap to preserve context across splits
TOP_K_DEFAULT  = 5      # how many chunks to return per query


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping character windows."""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def chunk_documents(docs: list[str]) -> list[str]:
    """Chunk every document string and return a flat list of chunks."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))
    return all_chunks


# ── Embedding helper ─────────────────────────────────────────────────────────

def _load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", model_name)
    # Use the local cache without any HF Hub network checks if the model is
    # already downloaded.  Falls back to a one-time download if not cached.
    try:
        model = SentenceTransformer(model_name, local_files_only=True)
        logger.info("Embedding model loaded from local cache (offline)")
        return model
    except Exception:
        logger.info("Model not in local cache — downloading from HuggingFace (one-time only)…")
        return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """Return a float32 (N, dim) array of normalised embeddings."""
    logger.info("Embedding %d texts …", len(texts))
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine via inner product
        convert_to_numpy=True,
    )
    return vecs.astype("float32")


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    A FAISS-backed vector store.

    Usage (build):
        vs = VectorStore()
        vs.build(documents)
        vs.save()

    Usage (query):
        vs = VectorStore()
        vs.load()
        results = vs.search("how many users?")
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._chunks: list[str] = []
        # Simple in-memory query cache (question → answer strings)
        self._cache: dict[str, list[str]] = {}

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, documents: list[str]) -> None:
        """Chunk documents, embed them, and build a FAISS index."""
        self._chunks = chunk_documents(documents)
        logger.info("Total chunks: %d", len(self._chunks))

        vecs = embed_texts(self._chunks, self.model)
        dim  = vecs.shape[1]

        # IndexFlatIP → inner product on L2-normalised vecs == cosine similarity
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)
        logger.info("FAISS index built: %d vectors, dim=%d", self._index.ntotal, dim)

    # ── Persist ────────────────────────────────────────────────────────────

    def save(self, vectors_dir: Path = VECTORS_DIR) -> None:
        vectors_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(vectors_dir / "faiss.index"))
        with open(vectors_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)
        with open(vectors_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"model": self.model_name, "n_chunks": len(self._chunks)}, f)
        logger.info("Vector store saved to %s", vectors_dir)

    def load(self, vectors_dir: Path = VECTORS_DIR) -> None:
        idx_path    = vectors_dir / "faiss.index"
        chunks_path = vectors_dir / "chunks.json"
        meta_path   = vectors_dir / "meta.json"

        if not idx_path.exists():
            raise FileNotFoundError(
                f"No trained index found at {idx_path}. Run `python train.py` first."
            )

        self._index = faiss.read_index(str(idx_path))
        with open(chunks_path, encoding="utf-8") as f:
            self._chunks = json.load(f)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.model_name = meta.get("model", self.model_name)
        logger.info(
            "Vector store loaded: %d chunks, model=%s",
            len(self._chunks), self.model_name
        )

    # ── Search ─────────────────────────────────────────────────────────────

    def search(self, question: str, top_k: int = TOP_K_DEFAULT) -> list[str]:
        """
        Embed *question* and return the top-k most relevant text chunks.
        Results are cached in memory to avoid re-embedding the same query.
        """
        cache_key = f"{question}__k{top_k}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        q_vec = embed_texts([question], self.model)   # (1, dim)
        scores, indices = self._index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(self._chunks[idx])

        self._cache[cache_key] = results
        return results

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and len(self._chunks) > 0
