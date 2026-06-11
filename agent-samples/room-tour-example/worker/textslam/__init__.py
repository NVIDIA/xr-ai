# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""textslam -- SLAM in text space (XR-AI port of github.com/nvddr/textslam).

Perceive sparse monocular frames into structured text (caption + objects + OCR),
discard the pixels, and build a semantic-topological place graph purely from the
text. Supports relocalization ("where am I") and nearby-context recall the way a
human does -- from scene content, with no metric poses.

This is a direct port of the upstream library's core (``types``, ``scoring``,
``relations``, ``landmarks``, ``topomap`` are carried over intact). The two model
backends are swapped for the XR-AI stack: ``perception.VLMPerceptor`` uses the
shared VLM via ``xr-ai-models`` instead of Florence-2, and
``embedding.HashingEmbedder`` replaces BGE so the worker pulls no embedding model.
"""
from .embedding import Embedder, HashingEmbedder
from .perception import Perceptor, VLMPerceptor
from .scoring import ScoreBreakdown, ScoreWeights
from .topomap import IngestResult, Localization, SemanticTopoMap
from .types import Detection, Observation, PlaceNode, SceneDescription

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "Perceptor",
    "VLMPerceptor",
    "ScoreBreakdown",
    "ScoreWeights",
    "IngestResult",
    "Localization",
    "SemanticTopoMap",
    "Detection",
    "Observation",
    "PlaceNode",
    "SceneDescription",
]
