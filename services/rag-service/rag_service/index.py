# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Document loading, chunking, embedding, and cosine retrieval."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from xr_ai_models import EmbeddingService


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    clean = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    if not clean:
        return []
    chunks: list[str] = []
    buffer = ""
    for paragraph in re.split(r"\n\n+", clean):
        paragraph = paragraph.strip()
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        buffer = ""
        words: list[str] = []
        for word in paragraph.split():
            candidate = " ".join([*words, word])
            if len(candidate) <= chunk_size:
                words.append(word)
                continue
            if words:
                chunk = " ".join(words)
                chunks.append(chunk)
                prefix = chunk[-overlap:].lstrip() if overlap else ""
                words = ([prefix] if prefix else []) + [word]
            else:
                chunks.append(word[:chunk_size])
                words = [word[chunk_size:]] if word[chunk_size:] else []
        buffer = " ".join(words)
    if buffer:
        chunks.append(buffer)
    return chunks


class DenseIndex:
    def __init__(
        self,
        *,
        chunks: list[Chunk],
        vectors: np.ndarray,
        documents: list[str],
        embedder: EmbeddingService,
        embedding_dim: int,
        query_prefix: str,
        min_score: float,
    ) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.documents = documents
        self._embedder = embedder
        self._embedding_dim = embedding_dim
        self._query_prefix = query_prefix
        self._min_score = min_score

    async def health(self) -> bool:
        try:
            return await self._embedder.health()
        except Exception:
            return False

    @classmethod
    async def build(
        cls,
        documents_dir: Path,
        embedder: EmbeddingService,
        *,
        cache_dir: Path,
        chunk_size: int = 900,
        overlap: int = 120,
        embedding_dim: int = 768,
        batch_size: int = 32,
        passage_prefix: str = "passage: ",
        query_prefix: str = "query: ",
        cache_key: str = "",
        min_score: float = 0.3,
    ) -> "DenseIndex":
        if chunk_size <= 0 or not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        if embedding_dim <= 0 or batch_size <= 0:
            raise ValueError("embedding_dim and batch_size must be positive")
        paths = sorted(
            path for path in documents_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}
        )
        documents = [str(path.relative_to(documents_dir)) for path in paths]
        chunks = [
            Chunk(source=source, text=part)
            for path, source in zip(paths, documents, strict=True)
            for part in _chunk_text(path.read_text(errors="replace"), chunk_size=chunk_size, overlap=overlap)
        ]
        if not chunks:
            raise ValueError(f"no non-empty .md or .txt documents found in {documents_dir}")

        digest = hashlib.sha256()
        digest.update(
            f"{chunk_size}:{overlap}:{embedding_dim}:{passage_prefix}:{cache_key}".encode()
        )
        for chunk in chunks:
            digest.update(chunk.source.encode())
            digest.update(chunk.text.encode())
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{digest.hexdigest()}.npz"
        vectors: np.ndarray | None = None
        if cache_path.exists():
            try:
                with np.load(cache_path) as cached:
                    vectors = cached["vectors"]
                _validate_vectors(
                    vectors,
                    row_count=len(chunks),
                    embedding_dim=embedding_dim,
                )
            except (OSError, KeyError, ValueError):
                vectors = None
        if vectors is None:
            rows: list[list[float]] = []
            texts = [passage_prefix + chunk.text for chunk in chunks]
            for offset in range(0, len(texts), batch_size):
                rows.extend(await embedder.embed(texts[offset:offset + batch_size]))
            vectors = np.asarray(rows, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] < embedding_dim:
                raise ValueError(
                    f"embedding service returned shape {vectors.shape}; "
                    f"expected {len(chunks)} rows with at least {embedding_dim} columns"
                )
            vectors = vectors[:, :embedding_dim]
            _validate_vectors(
                vectors,
                row_count=len(chunks),
                embedding_dim=embedding_dim,
            )
            vectors = _normalize(vectors)
            temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp.npz")
            np.savez_compressed(temporary, vectors=vectors)
            temporary.replace(cache_path)
        return cls(
            chunks=chunks,
            vectors=vectors,
            documents=documents,
            embedder=embedder,
            embedding_dim=embedding_dim,
            query_prefix=query_prefix,
            min_score=min_score,
        )

    async def retrieve(self, query: str, *, top_k: int) -> list[dict]:
        rows = np.asarray(
            await self._embedder.embed([self._query_prefix + query]),
            dtype=np.float32,
        )
        if rows.ndim != 2 or rows.shape[1] < self._embedding_dim:
            raise ValueError(
                f"embedding service returned shape {rows.shape}; expected one row "
                f"with at least {self._embedding_dim} columns"
            )
        vector = rows[:, : self._embedding_dim]
        _validate_vectors(vector, row_count=1, embedding_dim=self._embedding_dim)
        scores = self.vectors @ _normalize(vector)[0]
        indices = [
            index
            for index in np.argsort(scores)[::-1]
            if scores[index] >= self._min_score
        ][:top_k]
        return [
            {
                "text": self.chunks[index].text,
                "source": self.chunks[index].source,
                "score": float(scores[index]),
            }
            for index in indices
        ]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _validate_vectors(
    vectors: np.ndarray,
    *,
    row_count: int,
    embedding_dim: int,
) -> None:
    expected = (row_count, embedding_dim)
    if vectors.shape != expected:
        raise ValueError(f"embedding vectors have shape {vectors.shape}; expected {expected}")
    if not np.isfinite(vectors).all():
        raise ValueError("embedding vectors contain non-finite values")
