# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper around DINOv2 for L2-normalized global image descriptors.

The model is loaded lazily on first call so the MCP server starts in well
under a second; subsequent calls reuse the same model + processor and
serialize behind a lock because the underlying weights aren't thread-safe.

All output descriptors are unit-normalized so downstream callers can
compute cosine similarity by a simple dot product.
"""
from __future__ import annotations

import threading
import time
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Duck-typed interface so tests can substitute a fake."""
    embedding_dim: int
    def __call__(self, image_rgb: np.ndarray) -> np.ndarray: ...


class DinoV2Embedder:
    """Returns the L2-normalized CLS token from a DINOv2 backbone.

    Apache-2.0 weights; pulled from HuggingFace on first call and cached
    under the standard ``~/.cache/huggingface`` location.
    """

    def __init__(self, *, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device     = device
        self._model      = None  # lazy
        self._processor  = None
        self._torch      = None
        self._lock       = threading.Lock()
        # Filled in after the first load.
        self.embedding_dim: int = 0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from loguru import logger
        from transformers import AutoImageProcessor, AutoModel
        device = _resolve_device(self._device)
        logger.info(
            "DINOv2: loading {} on {} (first call — may download weights from HuggingFace)",
            self._model_name, device,
        )
        t0 = time.monotonic()
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        m = AutoModel.from_pretrained(self._model_name).to(device).eval()
        self._model        = m
        self._device       = device
        self._torch        = torch
        self.embedding_dim = int(m.config.hidden_size)
        logger.info("DINOv2: ready  dim={}  ({:.1f}s)", self.embedding_dim, time.monotonic() - t0)

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        torch = self._torch
        # The HF processor accepts a numpy HWC array directly.
        inputs = self._processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._lock, torch.no_grad():
            out = self._model(**inputs)
        # Use the CLS token (first position) — DINOv2 doesn't ship a separate
        # pooler, so the standard practice is the [CLS] hidden state.
        cls = out.last_hidden_state[:, 0, :]
        cls = cls / cls.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return cls.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
