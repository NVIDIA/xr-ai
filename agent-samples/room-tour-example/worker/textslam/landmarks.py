# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Landmark index: relocalize by distinctive tokens, the way a human uses a
room number or an unusual object to know where they are.

This is the practical payload of the entity layer for relocalization, and the
"index more over time" idea, in one structure: an inverted index from token
(object label or OCR word) to the places that contain it, weighted by inverse
document frequency. A token in every place ("wall", "floor") carries almost no
information; a token in one place ("214", "fire-extinguisher") nearly pins the
location. Pure text -- no appearance vectors.

It is additive: as more observations arrive, place token sets grow and IDF
sharpens, so relocalization gets *more* precise over time, not less.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .types import PlaceNode, tokenize


@dataclass(slots=True)
class LandmarkIndex:
    # token -> set of place ids containing it
    postings: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    # token -> {place_id: term weight in that place}
    place_token_w: dict[int, dict[str, float]] = field(default_factory=dict)
    n_places: int = 0
    include_ocr: bool = True

    @classmethod
    def build(
        cls,
        nodes: dict[int, PlaceNode],
        include_ocr: bool = True,
        min_support: int = 1,
    ) -> LandmarkIndex:
        idx = cls()
        idx.n_places = len(nodes)
        idx.include_ocr = include_ocr
        for nid, node in nodes.items():
            ents = node.entities()
            # term weight: object labels weighted by support (persistence). OCR
            # tokens (signage) optionally added with a flat strong weight -- set
            # include_ocr=False for an object-only index. ``min_support`` drops
            # one-off / transient detections (a person walking by, a single
            # spurious box) so IDF can't over-trust noise -- important when
            # observations per place are few (sparse sampling).
            tw: dict[str, float] = {}
            for lbl, e in ents.items():
                if e.support < min_support:
                    continue
                tw[lbl] = float(e.support)
            if include_ocr:
                for tok in node.ocr_set():
                    tw[tok] = max(tw.get(tok, 0.0), 2.0)
            idx.place_token_w[nid] = tw
            for tok in tw:
                idx.postings[tok].add(nid)
        return idx

    def idf(self, token: str) -> float:
        df = len(self.postings.get(token, ()))
        if df == 0:
            return 0.0
        # smoothed idf; rare token -> high
        return math.log((self.n_places + 1) / (df + 0.5))

    def score(self, query_tokens: set[str]) -> dict[int, float]:
        """IDF-weighted vote: for each query token, add its IDF to every place
        that contains it (scaled by that place's term weight). Returns
        place_id -> raw landmark score (un-normalized)."""
        scores: dict[int, float] = defaultdict(float)
        for tok in query_tokens:
            w_idf = self.idf(tok)
            if w_idf <= 0.0:
                continue
            for nid in self.postings.get(tok, ()):  # inverted-index lookup
                scores[nid] += w_idf * self.place_token_w[nid].get(tok, 1.0)
        return dict(scores)

    def score_normalized(self, query_tokens: set[str]) -> dict[int, float]:
        """Same as ``score`` but squashed to ~[0,1] for blending with the
        place-similarity score. Normalized by the best place's score."""
        raw = self.score(query_tokens)
        if not raw:
            return {}
        top = max(raw.values())
        return {nid: s / top for nid, s in raw.items()} if top > 0 else {}


def description_tokens(desc, include_ocr: bool = True) -> set[str]:
    """Landmark tokens of a query description: object labels (+ OCR tokens)."""
    toks = set(desc.object_labels())
    if include_ocr:
        for line in desc.ocr:
            toks |= tokenize(line)
    return toks
