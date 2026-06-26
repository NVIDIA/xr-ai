# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data-association scoring -- the single load-bearing seam of text-space SLAM.

Whether two text descriptions refer to the *same place* is the entire problem.
A naive whole-blob cosine similarity fails two ways:

  * perceptual aliasing -- two different look-alike places (two hallways) score
    high and get wrongly merged / a false loop closure fires;
  * viewpoint variance -- two views of the same place score low and the place
    fragments into many nodes.

So we score with three complementary signals, each capturing something the
others miss, and we return a full breakdown so every decision is inspectable:

  * caption cosine -- semantic gist (text embedding), best-of over the node's
    observations (never a centroid -- a mean blurs multi-view places);
  * object-set Jaccard -- "are the same things here" (furniture, fixtures);
  * OCR / signage overlap -- room numbers, labels, signs. This is what humans
    actually relocalize on, so it is weighted heavily and, when both sides
    carry text, can decide the match on its own.

This module is deliberately self-contained and model-free so it is unit-testable
and swappable without touching the map.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import Observation, PlaceNode


@dataclass(slots=True)
class ScoreWeights:
    caption: float = 0.5
    objects: float = 0.3
    ocr: float = 0.2
    # object spatial-arrangement signal (stage 2). 0 = off (default), so legacy
    # behavior is unchanged; object-only mode turns it on.
    relations: float = 0.0
    # When both observation and node carry OCR tokens, signage dominates: blend
    # the base score toward the OCR overlap by this factor. Signage is the most
    # discriminative relocalization cue a human uses.
    ocr_present_boost: float = 0.6


@dataclass(slots=True)
class SignalThresholds:
    """Per-signal "this signal agrees" thresholds, for the loop-closure
    agreement gate. A loop closure must be backed by >= ``min_signals`` of these,
    OR a strong unique-OCR match (``strong_ocr``) which can stand alone."""

    caption: float = 0.55
    objects: float = 0.30
    ocr: float = 0.34
    min_signals: int = 2
    strong_ocr: float = 0.50


@dataclass(slots=True)
class ScoreBreakdown:
    node_id: int
    total: float
    caption_cos: float
    object_jaccard: float
    ocr_overlap: float
    relation_overlap: float = 0.0
    ocr_active: bool = False
    best_obs_index: int = -1
    # adjusted = total * sequence-prior weight (set by the map; total stays raw)
    adjusted: float = -1.0
    prior_weight: float = 1.0
    notes: dict = field(default_factory=dict)

    def strong_signals(self, th: SignalThresholds) -> int:
        return int(self.caption_cos >= th.caption) + \
               int(self.object_jaccard >= th.objects) + \
               int(self.ocr_overlap >= th.ocr)

    def passes_agreement(self, th: SignalThresholds) -> bool:
        """Enough independent evidence to trust a (graph-distant) loop closure."""
        if self.ocr_overlap >= th.strong_ocr:
            return True  # a strong unique-signage match stands on its own
        return self.strong_signals(th) >= th.min_signals

    def __str__(self) -> str:
        flag = " [OCR]" if self.ocr_active else ""
        adj = f" adj={self.adjusted:.3f}(x{self.prior_weight:.2f})" if self.adjusted >= 0 else ""
        return (
            f"node {self.node_id:>3}: total={self.total:.3f}{adj} "
            f"(cap={self.caption_cos:.3f} obj={self.object_jaccard:.3f} "
            f"ocr={self.ocr_overlap:.3f}){flag}"
        )


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _best_caption_cos(obs_emb: np.ndarray, node: PlaceNode) -> tuple[float, int]:
    """Max cosine over the node's observations (embeddings are L2-normalized)."""
    mat = node.embedding_matrix()  # (n, d)
    sims = mat @ obs_emb  # cosine, since normalized
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx


def score_against_node(
    obs: Observation,
    node: PlaceNode,
    weights: ScoreWeights | None = None,
) -> ScoreBreakdown:
    w = weights or ScoreWeights()
    desc = obs.description

    cap_cos, best_idx = _best_caption_cos(obs.embedding, node)
    # cosine is in [-1, 1]; clamp negatives to 0 for a clean [0,1] blend
    cap_cos = max(0.0, cap_cos)

    obj_jac = _jaccard(desc.object_labels(), node.label_set())

    obs_ocr = desc.ocr_tokens()
    node_ocr = node.ocr_set()
    ocr_overlap = _jaccard(obs_ocr, node_ocr)
    ocr_active = bool(obs_ocr and node_ocr)

    rel_overlap = 0.0
    if w.relations > 0.0:
        from .relations import obs_relations, overlap_coeff, place_relations

        rel_overlap = overlap_coeff(obs_relations(desc), place_relations(node))

    base = (
        w.caption * cap_cos
        + w.objects * obj_jac
        + w.ocr * ocr_overlap
        + w.relations * rel_overlap
    )
    # renormalize by the weights actually in play
    denom = w.caption + w.objects + w.ocr + w.relations
    base = base / denom if denom else 0.0

    if ocr_active:
        # signage is decisive when present on both sides: pull toward OCR overlap
        total = (1.0 - w.ocr_present_boost) * base + w.ocr_present_boost * ocr_overlap
    else:
        total = base

    return ScoreBreakdown(
        node_id=node.node_id,
        total=float(total),
        caption_cos=float(cap_cos),
        object_jaccard=float(obj_jac),
        ocr_overlap=float(ocr_overlap),
        relation_overlap=float(rel_overlap),
        ocr_active=ocr_active,
        best_obs_index=best_idx,
    )


def score_against_all(
    obs: Observation,
    nodes: dict[int, PlaceNode],
    weights: ScoreWeights | None = None,
) -> list[ScoreBreakdown]:
    """Score an observation against every node, sorted best-first."""
    out = [score_against_node(obs, n, weights) for n in nodes.values()]
    out.sort(key=lambda b: b.total, reverse=True)
    return out
