# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RAG MCP server — dense vector retrieval over a directory of text/markdown documents.

Pure FastMCP — every operation is an MCP tool at /mcp. Any agent that needs
to ground its answers in a local document corpus can connect here instead of
owning the retrieval code itself.

Document corpus
───────────────
The server indexes all ``.txt`` and ``.md`` files found in ``docs_dir`` at
startup.  The path is configured via the YAML ``docs_dir`` key or the
``RAG_DOCS_DIR`` environment variable (env var takes precedence).  Each
document is split into overlapping character-level chunks; each chunk is
embedded by the embedding server and stored as a normalised float32 vector.
Embeddings are cached to disk so restarts skip re-embedding unchanged corpora.

Tools (FastMCP, mounted at /mcp)
────────────────────────────────
  retrieve(query, top_k) → list[dict]
      Return the *top_k* document chunks whose embedding is most similar to
      the query embedding (cosine similarity, since both are L2-normalised).
      Each result has keys: text (str), source (str), score (float).
      Returns an empty list when no chunks score above ``min_score``.

  list_documents() → list[str]
      Names of all indexed source files (without directory prefix).

Config (rag_mcp_server.yaml)
────────────────────────────
    docs_dir:         ./docs    # path to .txt / .md files; relative to YAML
    host:             0.0.0.0
    port:             8240
    chunk_size:       400       # target chars per chunk
    chunk_overlap:    80        # overlap between adjacent chunks
    embedding_server: http://localhost:8109   # embedding-server URL
    embedding_dim:    768       # Matryoshka dim to truncate vectors to
    min_score:        0.3       # cosine threshold below which chunks are dropped
    request_timeout_s: 30.0    # HTTP timeout for embedding calls
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import pathlib
import re
import time
from dataclasses import dataclass

import httpx
import numpy as np
import uvicorn
import yaml
from fastmcp import FastMCP
from loguru import logger
from xr_ai_logging import setup_logging


# ── chunker ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Chunk:
    text:   str
    source: str   # filename, for attribution


# Bump this whenever the chunker or preprocessing changes — it is folded
# into the corpus hash so cached embeddings auto-invalidate on code changes.
_CHUNKER_VERSION = "v2-strip-html-comments"


def _strip_noise(text: str) -> str:
    """Remove license / TODO comment blocks that would otherwise be embedded.

    HTML / Markdown comment blocks (``<!-- … -->``) at the top of files
    usually carry SPDX license boilerplate or editorial TODOs.  Embedding
    them gives a high-IDF top-match for any query whose tokens happen to
    appear in the surrounding header — at the cost of context budget and
    relevance.  Strip them before chunking.
    """
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _split_chunks(text: str, size: int, overlap: int) -> list[str]:
    text = _strip_noise(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if len(candidate) <= size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            if len(para) > size:
                words = para.split()
                window = ""
                for word in words:
                    trial = (window + " " + word).strip()
                    if len(trial) <= size:
                        window = trial
                    else:
                        if window:
                            chunks.append(window)
                        overlap_text = window[-overlap:] if overlap else ""
                        window = (overlap_text + " " + word).strip()
                buf = window if window else ""
            else:
                buf = para
    if buf:
        chunks.append(buf)
    return chunks


# ── embedding client ──────────────────────────────────────────────────────────

class EmbeddingClient:
    """Minimal client for the vLLM embedding server's /v1/embeddings endpoint."""

    def __init__(self, base_url: str, dim: int, timeout_s: float = 30.0) -> None:
        self._url     = base_url.rstrip("/") + "/v1/embeddings"
        self._health  = base_url.rstrip("/") + "/health"
        self._dim     = dim
        self._timeout = timeout_s

    def _embed_sync(self, texts: list[str]) -> np.ndarray:
        """POST to /v1/embeddings and return an (N, dim) float32 array."""
        resp = httpx.post(
            self._url,
            json={"model": "embed", "input": texts},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # vLLM returns items in the order they were sent; each has "embedding".
        vecs = np.array([item["embedding"] for item in data], dtype=np.float32)
        # Matryoshka truncation: first self._dim dims are a valid sub-embedding.
        vecs = vecs[:, : self._dim]
        # L2-normalise each row so dot-product == cosine similarity.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vecs / norms).astype(np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed document passages with the required ``"passage: "`` prefix."""
        prefixed = ["passage: " + t for t in texts]
        return self._embed_sync(prefixed)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query with the required ``"query: "`` prefix."""
        return self._embed_sync(["query: " + text])[0]

    def wait_ready(self, max_wait_s: float = 120.0) -> None:
        """Poll /health until the server is up.  Raises on timeout."""
        deadline = time.monotonic() + max_wait_s
        while True:
            try:
                resp = httpx.get(self._health, timeout=5.0)
                if resp.is_success:
                    return
            except httpx.TransportError:
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"embedding-server at {self._health!r} did not become ready "
                    f"within {max_wait_s:.0f} s"
                )
            logger.info("rag-mcp: waiting for embedding-server …  ({:.0f}s left)", remaining)
            time.sleep(3.0)


# ── disk cache ────────────────────────────────────────────────────────────────

def _corpus_hash(docs_dir: pathlib.Path, embedding_dim: int, model: str) -> str:
    """Return a short hex digest that changes when any doc file or the
    chunker / preprocessing pipeline changes."""
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(str(embedding_dim).encode())
    h.update(_CHUNKER_VERSION.encode())
    for p in sorted(docs_dir.glob("*.txt")) + sorted(docs_dir.glob("*.md")):
        h.update(p.name.encode())
        h.update(str(p.stat().st_mtime_ns).encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


def _cache_path(docs_dir: pathlib.Path) -> pathlib.Path:
    return docs_dir / ".rag_embed_cache.npz"


def _load_cache(
    docs_dir: pathlib.Path, corpus_hash: str
) -> tuple[list[Chunk], np.ndarray] | None:
    """Return (chunks, matrix) from cache, or None if cache is absent/stale."""
    p = _cache_path(docs_dir)
    if not p.exists():
        return None
    try:
        data = np.load(p, allow_pickle=True)
        if data["hash"].item() != corpus_hash:
            return None
        texts   = data["texts"].tolist()
        sources = data["sources"].tolist()
        matrix  = data["matrix"]
        chunks  = [Chunk(text=t, source=s) for t, s in zip(texts, sources)]
        logger.info("rag-mcp: loaded {} embeddings from cache", len(chunks))
        return chunks, matrix
    except Exception as exc:
        logger.warning("rag-mcp: cache read failed ({}), re-indexing", exc)
        return None


def _save_cache(
    docs_dir: pathlib.Path,
    corpus_hash: str,
    chunks: list[Chunk],
    matrix: np.ndarray,
) -> None:
    p = _cache_path(docs_dir)
    try:
        np.savez(
            p,
            hash=np.array(corpus_hash),
            texts=np.array([c.text for c in chunks]),
            sources=np.array([c.source for c in chunks]),
            matrix=matrix,
        )
        logger.info("rag-mcp: saved {} embeddings to cache at {}", len(chunks), p)
    except Exception as exc:
        logger.warning("rag-mcp: cache write failed: {}", exc)


# ── dense index ───────────────────────────────────────────────────────────────

class DenseIndex:
    """Dense vector index built from a corpus of chunked documents.

    Construction calls the embedding server synchronously (blocking), so it
    must happen before the asyncio event loop starts.  After construction the
    index is read-only and safe to call from coroutines.
    """

    def __init__(
        self,
        docs_dir:      pathlib.Path,
        embedder:      EmbeddingClient,
        chunk_size:    int = 400,
        chunk_overlap: int = 80,
    ) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.file_count = 0

        raw: list[tuple[str, str]] = []
        for ext in ("*.txt", "*.md"):
            for p in sorted(docs_dir.glob(ext)):
                try:
                    raw.append((p.name, p.read_text(encoding="utf-8", errors="replace")))
                    self.file_count += 1
                except OSError:
                    continue

        all_chunks: list[Chunk] = []
        for source, text in raw:
            for chunk_text in _split_chunks(text, chunk_size, chunk_overlap):
                all_chunks.append(Chunk(text=chunk_text, source=source))

        if not all_chunks:
            logger.warning("rag-mcp: no documents found in {}", docs_dir)
            return

        corpus_hash = _corpus_hash(docs_dir, embedder._dim, "nvidia/llama-nemotron-embed-1b-v2")
        cached = _load_cache(docs_dir, corpus_hash)
        if cached is not None:
            self._chunks, self._matrix = cached
            return

        logger.info("rag-mcp: embedding {} chunks …", len(all_chunks))
        texts = [c.text for c in all_chunks]
        matrix = self._embed_with_retry(embedder, texts)
        _save_cache(docs_dir, corpus_hash, all_chunks, matrix)
        self._chunks = all_chunks
        self._matrix = matrix

    @staticmethod
    def _embed_with_retry(embedder: EmbeddingClient, texts: list[str]) -> np.ndarray:
        """Embed in one call with up to 3 retries and exponential backoff."""
        for attempt in range(3):
            try:
                return embedder.embed_passages(texts)
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"embedding failed after 3 attempts: {exc}"
                    ) from exc
                wait = 2 ** attempt
                logger.error(
                    "rag-mcp: embedding attempt {} failed: {} — retrying in {}s",
                    attempt + 1, exc, wait,
                )
                time.sleep(wait)
        raise AssertionError("unreachable")

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def retrieve(
        self, q_vec: np.ndarray, top_k: int, min_score: float
    ) -> list[tuple[Chunk, float]]:
        """Return up to *top_k* (chunk, score) pairs above *min_score*."""
        if not self._chunks:
            return []
        scores = self._matrix @ q_vec        # cosine similarity (both L2-normed)
        order  = np.argsort(scores)[::-1][:top_k]
        return [
            (self._chunks[i], float(scores[i]))
            for i in order
            if float(scores[i]) > min_score
        ]

    def sources(self) -> list[str]:
        seen: list[str] = []
        for chunk in self._chunks:
            if chunk.source not in seen:
                seen.append(chunk.source)
        return seen


# ── MCP server ────────────────────────────────────────────────────────────────

def build_mcp(index: DenseIndex, embedder: EmbeddingClient, min_score: float) -> "FastMCP":
    mcp = FastMCP("rag-mcp")

    @mcp.tool()
    def retrieve(query: str, top_k: int = 4) -> list[dict]:
        """
        Retrieve the most relevant document chunks for *query*.

        Returns up to *top_k* results, each with keys:
          text   (str)   — the chunk text
          source (str)   — source filename
          score  (float) — cosine similarity score (0.0–1.0)

        Returns an empty list if the corpus is empty or all chunks score below
        the configured minimum similarity threshold.
        """
        try:
            q_vec = embedder.embed_query(query)
        except Exception as exc:
            logger.warning("rag-mcp: embed query failed: {} — returning empty", exc)
            return []
        results = index.retrieve(q_vec, top_k=top_k, min_score=min_score)
        return [
            {"text": chunk.text, "source": chunk.source, "score": round(score, 4)}
            for chunk, score in results
        ]

    @mcp.tool()
    def list_documents() -> list[str]:
        """
        Return the filenames of all documents in the indexed corpus.

        Useful for callers that want to confirm which files are loaded before
        issuing a retrieve call.
        """
        return index.sources()

    return mcp


def build_app(index: DenseIndex, embedder: EmbeddingClient, min_score: float):
    return build_mcp(index, embedder, min_score).http_app(path="/mcp")


async def _serve(
    docs_dir:      pathlib.Path,
    host:          str,
    port:          int,
    chunk_size:    int,
    chunk_overlap: int,
    embedder:      EmbeddingClient,
    min_score:     float,
    ready_file:    pathlib.Path | None,
) -> None:
    logger.info("rag-mcp: building dense index from {}", docs_dir)
    index = DenseIndex(
        docs_dir, embedder,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    logger.info(
        "rag-mcp: {} chunk(s) from {} file(s)",
        index.chunk_count, index.file_count,
    )

    app    = build_app(index, embedder, min_score)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    logger.info("rag-mcp-server  mcp=/mcp  port={}  docs={}", port, docs_dir)
    if ready_file:
        ready_file.touch()
    await server.serve()


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    setup_logging("rag-mcp")

    env_docs = os.environ.get("RAG_DOCS_DIR")
    if env_docs:
        docs_dir = pathlib.Path(env_docs)
    elif "docs_dir" in cfg:
        docs_dir = (ns.config.parent / cfg["docs_dir"]).resolve() if ns.config else pathlib.Path(cfg["docs_dir"])
    else:
        docs_dir = pathlib.Path("docs")

    host          = cfg.get("host",             "0.0.0.0")
    port          = int(cfg.get("port",          8240))
    chunk_size    = int(cfg.get("chunk_size",    400))
    chunk_overlap = int(cfg.get("chunk_overlap",  80))
    embed_url     = cfg.get("embedding_server",  "http://localhost:8109")
    embed_dim     = int(cfg.get("embedding_dim", 768))
    min_score     = float(cfg.get("min_score",   0.3))
    req_timeout   = float(cfg.get("request_timeout_s", 30.0))

    embedder = EmbeddingClient(embed_url, dim=embed_dim, timeout_s=req_timeout)

    logger.info("rag-mcp: waiting for embedding-server at {} …", embed_url)
    embedder.wait_ready(max_wait_s=120.0)
    logger.info("rag-mcp: embedding-server ready")

    asyncio.run(_serve(
        docs_dir, host, port,
        chunk_size, chunk_overlap,
        embedder, min_score,
        ns.ready_file,
    ))


if __name__ == "__main__":
    run()
