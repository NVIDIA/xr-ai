# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text embedding backend.

Upstream TextSLAM ships BGE-small via sentence-transformers. This XR sample
avoids pulling a heavyweight embedding model into the worker (XR-AI has no
embedding service yet), so it ships a dependency-free ``HashingEmbedder``: a
deterministic hashed bag-of-words with signed buckets, L2-normalized. Its cosine
captures lexical overlap between captions/object-inventories well enough to drive
the place data-association in ``scoring`` — which is the only thing the map asks
of the embedder.

The ``Embedder`` protocol is the seam to later swap in BGE or an
OpenAI-compatible ``/v1/embeddings`` endpoint (the way ``xr-ai-models`` routes
every other model call) without touching the map. Place matching is *symmetric*
(description vs description), so there is no query/passage asymmetry to handle —
both sides get identical treatment, which is what makes the cosine meaningful.
"""
from __future__ import annotations

import re
import zlib
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) array of L2-normalized embeddings."""
        ...


class HashingEmbedder:
    """Deterministic hashed bag-of-words embedder (no model, no network).

    Each token is hashed (CRC32, so it is stable across processes — unlike the
    salted built-in ``hash``) to a bucket and a sign; counts accumulate into a
    fixed-width vector that is then L2-normalized. Cosine between two such vectors
    is a smoothed lexical overlap, which is what the caption-similarity signal
    needs here.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            h = zlib.crc32(tok.encode("utf-8"))
            bucket = h % self.dim
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed_one(t) for t in texts])
