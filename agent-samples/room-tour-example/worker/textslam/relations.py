# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Object spatial-arrangement layer: pose-free qualitative relations from boxes.

This is what breaks object-set saturation -- two places with the same objects
(desk, monitor, keyboard) are told apart by *how the objects are arranged*. The
relations form the edges of the object scene-graph.

Only the **viewpoint-invariant** tier (per DESIGN.md H2) is computed here, from
2-D normalized boxes (x1, y1, x2, y2), y-down:

  * ``near(a, b)``   -- centers close (symmetric). Proximity is largely
                        viewpoint-stable.
  * ``above(a, b)``  -- a is above b and they overlap horizontally
                        (gravity-aligned; stable under an upright camera).
  * ``inside(a, b)`` -- a's box is largely contained in b's ("on"/part-of:
                        mouse on mousepad, logo on monitor).

Heading-dependent ``left_of/right_of`` is deliberately NOT here -- it cannot be
aggregated across viewpoints without an orientation anchor (see DESIGN.md H2).

A relation is a canonical triple ``(rel, a, b)`` so it can go in a set and be
compared / voted across observations. ``near`` sorts its labels (symmetric);
``above``/``inside`` keep direction (upper/inner first).

Note: the XR ``VLMPerceptor`` emits labels without bounding boxes, so in this
sample relations are empty and the relations weight stays 0 (see ``scoring``).
The layer is ported intact so a box-producing perceptor can switch it on.
"""
from __future__ import annotations

NEAR_THR = 0.18      # center-distance (normalized) below which two objects are "near"
V_MARGIN = 0.06      # min vertical center gap to call one object "above" another
H_OVERLAP_MIN = 0.20 # min horizontal overlap (fraction of narrower box) for "above"
INSIDE_MIN = 0.70    # min fraction of a's area inside b to call a "inside" b

Triple = tuple[str, str, str]


def _center(b):
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def _overlap_1d(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _pair_relations(la: str, ba, lb: str, bb) -> list[Triple]:
    out: list[Triple] = []
    (ax, ay), (bx, by) = _center(ba), _center(bb)

    # near (symmetric) -- canonicalize by sorting labels
    if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < NEAR_THR:
        lo, hi = sorted((la, lb))
        out.append(("near", lo, hi))

    # above (directional, gravity-aligned) -- needs horizontal overlap so it's a
    # genuine vertical stack, not two objects on opposite sides
    wa, wb = ba[2] - ba[0], bb[2] - bb[0]
    hx = _overlap_1d(ba[0], ba[2], bb[0], bb[2])
    if hx >= H_OVERLAP_MIN * max(1e-6, min(wa, wb)):
        if ay < by - V_MARGIN:
            out.append(("above", la, lb))
        elif by < ay - V_MARGIN:
            out.append(("above", lb, la))

    # inside / on (directional) -- a contained in b
    area_a = max(1e-9, (ba[2] - ba[0]) * (ba[3] - ba[1]))
    area_b = max(1e-9, (bb[2] - bb[0]) * (bb[3] - bb[1]))
    inter = _overlap_1d(ba[0], ba[2], bb[0], bb[2]) * _overlap_1d(ba[1], ba[3], bb[1], bb[3])
    if area_a <= area_b and inter / area_a >= INSIDE_MIN:
        out.append(("inside", la, lb))
    elif area_b < area_a and inter / area_b >= INSIDE_MIN:
        out.append(("inside", lb, la))

    return out


def obs_relations(desc) -> set[Triple]:
    """All viewpoint-invariant relation triples in one observation."""
    objs = [(o.label.strip().lower(), o.bbox) for o in desc.objects if o.bbox]
    rels: set[Triple] = set()
    for i in range(len(objs)):
        la, ba = objs[i]
        for j in range(i + 1, len(objs)):
            lb, bb = objs[j]
            rels.update(_pair_relations(la, ba, lb, bb))
    return rels


def place_relations(node) -> set[Triple]:
    """Union of relation triples seen across a place's observations.

    A place's arrangement is the set of relations observed in it (voting/counts
    could refine this, but presence-union is the robust first cut)."""
    rels: set[Triple] = set()
    for o in node.observations:
        rels |= obs_relations(o.description)
    return rels


def overlap_coeff(query: set[Triple], place: set[Triple]) -> float:
    """How much of the query's arrangement the place explains: |Q∩P| / |Q|.

    Overlap-coefficient (not Jaccard) because the place aggregates many
    observations (large P) while a query is one view (small Q) -- Jaccard would
    unfairly penalize the size mismatch."""
    if not query:
        return 0.0
    return len(query & place) / len(query)
