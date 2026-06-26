# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The semantic topological map: incremental build, relocalization, persistence.

Nodes are places; edges are transitions between places observed over time. There
are no metric poses anywhere -- adjacency is purely "I saw B right after A", and
a loop-closure edge is "I returned to a place I'd seen before". This is a
cognitive/topological map, not a metric one.

Map-building data-association and relocalization are *the same operation*
(embed -> score against nodes -> best). ``relocalize`` is just that step exposed
read-only.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass

import numpy as np

from .scoring import (
    ScoreBreakdown,
    ScoreWeights,
    SignalThresholds,
    score_against_all,
)
from .types import Observation, PlaceNode, SceneDescription


def calibrate_threshold(descriptions, embedder, weights, percentile: float = 85.0) -> float:
    """Data-driven match threshold from the build set's similarity distribution.

    A fixed threshold is perceptor/scene-sensitive: a perceptor whose object-sets
    are very similar across frames (multi-room validation: Florence on Replica)
    over-merges everything at 0.62, while a varied one is fine. We set the
    threshold at a high percentile of the *pairwise* similarity among build
    observations, so only the most-similar pairs merge -- this auto-raises for
    saturated perceptors and stays moderate for varied ones. Clamped to a sane
    [0.5, 0.9] band."""
    import numpy as np

    from .relations import obs_relations

    descs = list(descriptions)
    n = len(descs)
    if n < 3:
        return 0.62
    embs = np.asarray(embedder.embed([d.to_embedding_text() for d in descs]), dtype=np.float32)
    labels = [d.object_labels() for d in descs]
    ocrs = [d.ocr_tokens() for d in descs]
    rels = [obs_relations(d) for d in descs]
    denom = weights.caption + weights.objects + weights.ocr + weights.relations

    def jac(a, b):
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)

    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            cap = max(0.0, float(embs[i] @ embs[j]))
            s = (
                weights.caption * cap
                + weights.objects * jac(labels[i], labels[j])
                + weights.ocr * jac(ocrs[i], ocrs[j])
                + weights.relations * jac(rels[i], rels[j])
            )
            scores.append(s / denom if denom else 0.0)
    return float(min(0.9, max(0.5, np.percentile(scores, percentile))))


@dataclass(slots=True)
class IngestResult:
    """What happened when one description was folded into the map."""

    node_id: int
    is_new_node: bool
    is_loop_closure: bool
    best_score: float
    scores: list[ScoreBreakdown]  # every node, best-first (instrumentation)
    reason: str = ""  # why this decision (stage-0 gating is logged here)

    @property
    def top(self) -> ScoreBreakdown | None:
        return self.scores[0] if self.scores else None


@dataclass(slots=True)
class Localization:
    """A relocalization decision with calibrated confidence.

    Separates the two ways localization fails: a *weak* top match (nothing in
    the map looks like here -> ``no_match``) and *confusion* (top-1 and top-2 are
    too close -> ``ambiguous``, torn between places). Only ``confident`` sets
    ``localized=True``; callers can also gate on ``confidence`` directly."""

    localized: bool
    node_id: int | None        # best guess (None only if the map is empty)
    confidence: float          # [0,1]
    status: str                # "confident" | "ambiguous" | "no_match"
    top_score: float
    margin: float              # top1 - top2 (separation from the runner-up)
    candidates: list[tuple[int, float]]  # top-k (node_id, score) -- the confusion set
    reason: str

    def to_dict(self) -> dict:
        return {
            "localized": self.localized, "node_id": self.node_id,
            "confidence": round(self.confidence, 3), "status": self.status,
            "top_score": round(self.top_score, 3), "margin": round(self.margin, 3),
            "candidates": self.candidates, "reason": self.reason,
        }


@dataclass(slots=True)
class HierarchicalLocalization:
    """Coarse-to-fine localization result. When the fine (place) level is too
    uncertain but the candidate places collapse into one region, we report the
    region instead of failing -- a coarser but still-useful answer."""

    level: str                  # "place" | "region" | "none"
    place_id: int | None
    region_id: int | None
    region_members: list[int]
    confidence: float
    candidates: list[tuple[int, float]]
    reason: str

    def to_dict(self) -> dict:
        return {
            "level": self.level, "place_id": self.place_id, "region_id": self.region_id,
            "region_members": self.region_members, "confidence": round(self.confidence, 3),
            "candidates": self.candidates, "reason": self.reason,
        }


@dataclass(slots=True)
class SequencePrior:
    """Topological motion prior: you are most likely near where you just were.

    Candidate nodes are weighted by graph-hop distance from the current node, so
    a look-alike place that is graph-distant is nudged below threshold while a
    genuine revisit (high raw score) still wins. The prior is intentionally
    *gentle* -- it must not suppress real loop closures, which are by definition
    jumps to graph-distant nodes; strong evidence overcomes it (Bayesian:
    posterior is evidence x prior, not a hard gate).
    """

    enabled: bool = True
    near_hops: int = 1     # within this many hops -> full weight 1.0
    floor: float = 0.80    # weight applied to far / disconnected candidates
    decay: float = 0.10    # linear falloff per hop beyond near_hops, down to floor

    def weight(self, hops: int | None) -> float:
        if not self.enabled or hops is None:  # disconnected or no context
            return self.floor if self.enabled else 1.0
        if hops <= self.near_hops:
            return 1.0
        return max(self.floor, 1.0 - self.decay * (hops - self.near_hops))


# Objects that signify a passage between places -- used to tag portal transitions.
PORTAL_LABELS = frozenset(
    {"door", "doorway", "opening", "hallway", "corridor", "gate", "stairs",
     "staircase", "elevator", "window"}
)


@dataclass(slots=True)
class Edge:
    a: int
    b: int
    kind: str  # "temporal" | "loop" | "shared_landmark"
    count: int = 1
    via: str = ""  # for shared_landmark/portal: the connecting object label


class SemanticTopoMap:
    """Incremental place graph built from text-only scene descriptions.

    Parameters
    ----------
    embedder:
        Object with ``embed(list[str]) -> np.ndarray`` (L2-normalized rows).
    match_threshold:
        Associate to an existing node when best score >= this. Below it, a new
        node is created. This is the aliasing/fragmentation knob -- tune it by
        watching the logged score distribution, not by guessing.
    weights:
        Scoring weights (see ``scoring.ScoreWeights``).
    """

    def __init__(
        self,
        embedder,
        match_threshold: float = 0.62,
        weights: ScoreWeights | None = None,
        ratio_margin: float = 0.04,
        sequence_prior: SequencePrior | None = None,
        signal_thresholds: SignalThresholds | None = None,
        landmark_weight: float = 0.25,
    ) -> None:
        self._embedder = embedder
        self.match_threshold = match_threshold
        self.weights = weights or ScoreWeights()
        # stage-0 robustness knobs
        self.ratio_margin = ratio_margin              # ambiguity / Lowe-style gate
        self.sequence_prior = sequence_prior or SequencePrior()
        self.signal_thresholds = signal_thresholds or SignalThresholds()
        # stage-1 landmark index (built on demand from current nodes)
        self.landmark_weight = landmark_weight
        self.landmark_index = None  # type: ignore[assignment]
        # hierarchical regions (place -> region id), built on demand
        self.regions: dict[int, int] = {}
        self.region_members: dict[int, list[int]] = {}

        self.nodes: dict[int, PlaceNode] = {}
        self.edges: dict[tuple[int, int], Edge] = {}
        self.visit_order: list[int] = []  # node_id per ingested frame, in time
        self.current_node_id: int | None = None
        self._next_node_id = 0
        self._next_obs_id = 0

    # ---- construction --------------------------------------------------

    def _embed_description(self, desc: SceneDescription) -> Observation:
        vec = self._embedder.embed([desc.to_embedding_text()])[0]
        vec = np.asarray(vec, dtype=np.float32)
        obs = Observation(
            obs_id=f"obs-{self._next_obs_id:06d}",
            description=desc,
            embedding=vec,
            seq_index=len(self.visit_order),
        )
        self._next_obs_id += 1
        return obs

    def _new_node(self, obs: Observation) -> PlaceNode:
        node = PlaceNode(node_id=self._next_node_id)
        node.add(obs)
        self.nodes[node.node_id] = node
        self._next_node_id += 1
        return node

    def _add_edge(self, a: int, b: int, kind: str, via: str = "") -> None:
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        if key in self.edges:
            e = self.edges[key]
            e.count += 1
            # promote toward the more informative kind; don't overwrite a
            # traversal/loop edge with a derived shared_landmark one
            if kind == "loop":
                e.kind = "loop"
            if via and not e.via:
                e.via = via
        else:
            self.edges[key] = Edge(a=key[0], b=key[1], kind=kind, via=via)

    def _hops_from(self, src: int | None) -> dict[int, int]:
        """BFS graph-hop distance from ``src`` to every reachable node."""
        if src is None:
            return {}
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def _apply_prior(self, scores: list[ScoreBreakdown], current: int | None) -> list[ScoreBreakdown]:
        """Annotate each breakdown with the sequence-prior-adjusted score and
        re-sort by it. ``total`` (raw evidence) is left untouched for logging."""
        hops = self._hops_from(current)
        for b in scores:
            w = self.sequence_prior.weight(hops.get(b.node_id))
            b.prior_weight = w
            b.adjusted = b.total * w
        scores.sort(key=lambda b: b.adjusted, reverse=True)
        return scores

    def ingest(self, desc: SceneDescription) -> IngestResult:
        """Fold one scene description into the map (online SLAM step).

        Stage-0 gating, in order: sequence prior re-ranks candidates; a ratio
        test rejects ambiguous matches; and a graph-distant association (a loop
        closure) additionally requires multi-signal agreement. Any rejection
        falls through to creating a new place.
        """
        obs = self._embed_description(desc)
        prev = self.current_node_id

        if not self.nodes:
            node = self._new_node(obs)
            self.current_node_id = node.node_id
            self.visit_order.append(node.node_id)
            return IngestResult(node.node_id, True, False, 1.0, [], "first frame")

        scores = self._apply_prior(score_against_all(obs, self.nodes, self.weights), prev)
        best = scores[0]
        second_adj = scores[1].adjusted if len(scores) > 1 else 0.0

        reason = ""
        accept = best.total >= self.match_threshold
        if accept and (best.adjusted - second_adj) < self.ratio_margin:
            accept, reason = False, f"ambiguous (margin {best.adjusted - second_adj:.3f}<{self.ratio_margin})"

        is_loop = bool(accept and prev is not None and prev != best.node_id
                       and best.node_id in self.visit_order)
        if is_loop and not best.passes_agreement(self.signal_thresholds):
            # graph-distant jump without enough independent evidence -> distrust it
            accept, is_loop, reason = False, False, (
                f"loop rejected: only {best.strong_signals(self.signal_thresholds)} "
                f"signal(s) agree"
            )

        if accept:
            node = self.nodes[best.node_id]
            node.add(obs)
            if prev is not None and prev != node.node_id:
                self._add_edge(prev, node.node_id, "loop" if is_loop else "temporal")
            self.current_node_id = node.node_id
            self.visit_order.append(node.node_id)
            return IngestResult(node.node_id, False, is_loop, best.total, scores,
                                "loop closure" if is_loop else "associated")

        # no accepted match -> new place, temporal edge from where we just were
        node = self._new_node(obs)
        if prev is not None:
            self._add_edge(prev, node.node_id, "temporal")
        self.current_node_id = node.node_id
        self.visit_order.append(node.node_id)
        return IngestResult(node.node_id, True, False, best.total, scores,
                            reason or "below threshold")

    # ---- query ---------------------------------------------------------

    def index_landmarks(self, include_ocr: bool = True, min_support: int = 1) -> None:
        """(Re)build the landmark inverted index from current places. Call after
        building / updating the map; cheap, and sharpens as the map grows. Set
        ``include_ocr=False`` for an object-only index; raise ``min_support`` to
        drop one-off / transient detections from the index."""
        from .landmarks import LandmarkIndex

        self.landmark_index = LandmarkIndex.build(
            self.nodes, include_ocr=include_ocr, min_support=min_support
        )

    def relocalize(
        self,
        desc: SceneDescription,
        top_k: int = 5,
        current_hint: int | None = None,
        use_landmarks: bool = False,  # env-gated (needs anchored signage); off by default
    ) -> list[ScoreBreakdown]:
        """Where am I? Score a (held-out) description against the built map.

        Ranking score (``adjusted``) blends place similarity (``total``) with the
        landmark-index vote (distinctive shared tokens), then optionally the
        sequence prior. ``total`` stays the raw place-similarity evidence.
        Pass ``current_hint`` for tracking-style relocalization; omit it for the
        kidnapped-robot case (how place-recognition recall is measured).
        """
        if not self.nodes:
            return []
        from .landmarks import description_tokens

        obs = Observation(
            obs_id="query",
            description=desc,
            embedding=np.asarray(self._embedder.embed([desc.to_embedding_text()])[0], dtype=np.float32),
        )
        scores = score_against_all(obs, self.nodes, self.weights)

        lm = {}
        if use_landmarks and self.landmark_index is not None:
            qtok = description_tokens(desc, include_ocr=self.landmark_index.include_ocr)
            lm = self.landmark_index.score_normalized(qtok)
        w = self.landmark_weight if lm else 0.0
        hops = self._hops_from(current_hint) if current_hint is not None else None
        for b in scores:
            lm_s = lm.get(b.node_id, 0.0)
            b.notes["landmark"] = lm_s
            base = (1.0 - w) * b.total + w * lm_s
            if hops is not None:
                pw = self.sequence_prior.weight(hops.get(b.node_id))
                b.prior_weight = pw
                base *= pw
            b.adjusted = base
        scores.sort(key=lambda b: b.adjusted, reverse=True)
        return scores[:top_k]

    def localize(
        self,
        desc: SceneDescription,
        *,
        min_score: float = 0.40,
        min_margin: float = 0.05,
        top_k: int = 5,
        current_hint: int | None = None,
        use_landmarks: bool = False,
    ) -> Localization:
        """Relocalize with a confidence/confusion verdict.

        ``confidence`` blends absolute match strength (top score) with how
        clearly it beats the runner-up (margin) -- a high top score with a tiny
        margin is *confused*, not confident. ``min_score`` / ``min_margin`` are
        the tunable gates; lower them for more coverage, raise for higher
        precision. The full ranked ``candidates`` list is returned so an
        ``ambiguous`` result can name the places it's torn between."""
        scores = self.relocalize(desc, top_k=top_k, current_hint=current_hint,
                                 use_landmarks=use_landmarks)
        if not scores:
            return Localization(False, None, 0.0, "no_match", 0.0, 0.0, [], "empty map")
        top = scores[0].adjusted
        second = scores[1].adjusted if len(scores) > 1 else 0.0
        margin = top - second
        cands = [(s.node_id, round(s.adjusted, 3)) for s in scores]
        score_ok = top >= min_score
        margin_ok = margin >= min_margin
        mfac = 1.0 if min_margin <= 0 else min(1.0, margin / min_margin)
        confidence = max(0.0, min(1.0, top * mfac))
        best = scores[0].node_id
        if not score_ok:
            status = "no_match"
            reason = f"best match weak (top {top:.2f} < {min_score})"
        elif not margin_ok:
            status = "ambiguous"
            reason = (f"confused between node {scores[0].node_id} and "
                      f"{scores[1].node_id} (margin {margin:.2f} < {min_margin})")
        else:
            status = "confident"
            reason = f"clear match to node {best} (top {top:.2f}, margin {margin:.2f})"
        return Localization(status == "confident", best, confidence, status,
                            top, margin, cands, reason)

    def build_regions(self, threshold: float = 0.35, max_passes: int = 100) -> int:
        """Group places into regions by graph-constrained agglomerative
        clustering: repeatedly merge the graph-adjacent region pair with the
        highest single-link place similarity, until no adjacent pair clears
        ``threshold``. Regions are therefore *contiguous* (connected in the place
        graph) and internally similar -- the coarse layer above places. Returns
        the number of regions. Threshold is looser than consolidation: this
        groups places, it does not merge their identity."""
        from collections import defaultdict

        region = {n: n for n in self.nodes}

        def members_of() -> dict[int, list[int]]:
            m: dict[int, list[int]] = defaultdict(list)
            for n, r in region.items():
                m[r].append(n)
            return m

        for _ in range(max_passes):
            members = members_of()
            pairs = set()
            for (a, b) in self.edges:
                ra, rb = region[a], region[b]
                if ra != rb:
                    pairs.add((min(ra, rb), max(ra, rb)))
            best, best_sim = None, threshold
            for (ra, rb) in pairs:
                sim = max(self._node_similarity(x, y)
                          for x in members[ra] for y in members[rb])
                if sim >= best_sim:
                    best, best_sim = (ra, rb), sim
            if best is None:
                break
            ra, rb = best
            for n in self.nodes:
                if region[n] == rb:
                    region[n] = ra
        uniq = {r: i for i, r in enumerate(sorted(set(region.values())))}
        self.regions = {n: uniq[r] for n, r in region.items()}
        m = members_of()
        self.region_members = {uniq[r]: sorted(ns) for r, ns in m.items()}
        return len(self.region_members)

    def region_summary(self, region_id: int) -> str:
        """Coarse description of a region: its most common objects + any portals."""
        from collections import Counter
        labels: Counter = Counter()
        portals: set[str] = set()
        for nid in self.region_members.get(region_id, []):
            for lbl, e in self.nodes[nid].entities().items():
                labels[lbl] += e.support
            portals |= self.place_portals(nid)
        top = [lab for lab, _ in labels.most_common(6)]
        s = "area with " + ", ".join(top) if top else "an area"
        if portals:
            s += f" (portals: {', '.join(sorted(portals))})"
        return s

    def localize_hierarchical(
        self,
        desc: SceneDescription,
        *,
        min_score: float = 0.40,
        min_margin: float = 0.05,
        region_share: float = 0.60,
        top_k: int = 8,
        current_hint: int | None = None,
        use_landmarks: bool = False,
    ) -> HierarchicalLocalization:
        """Coarse-to-fine. Try place-level (confident) first; if the place is
        ambiguous but the candidate evidence concentrates in one region (>=
        ``region_share`` of the candidate score mass), report that region;
        otherwise give up (``none``)."""
        if not self.regions:
            self.build_regions()
        loc = self.localize(desc, min_score=min_score, min_margin=min_margin,
                            top_k=top_k, current_hint=current_hint, use_landmarks=use_landmarks)
        if loc.localized:
            rid = self.regions.get(loc.node_id)
            return HierarchicalLocalization(
                "place", loc.node_id, rid, self.region_members.get(rid, []),
                loc.confidence, loc.candidates, "confident place")
        if loc.status == "no_match":
            return HierarchicalLocalization("none", None, None, [], loc.confidence,
                                            loc.candidates, loc.reason)
        # ambiguous place -> fall back to region via candidate score mass
        from collections import defaultdict
        rs: dict[int, float] = defaultdict(float)
        for nid, score in loc.candidates:
            rs[self.regions.get(nid, -1)] += score
        total = sum(rs.values()) or 1.0
        ranked = sorted(rs.items(), key=lambda x: -x[1])
        rid, rmass = ranked[0]
        share = rmass / total
        if rid >= 0 and share >= region_share:
            return HierarchicalLocalization(
                "region", None, rid, self.region_members.get(rid, []),
                share, loc.candidates,
                f"place ambiguous; {share:.0%} of evidence in region {rid}")
        return HierarchicalLocalization("none", None, None, [], loc.confidence,
                                        loc.candidates,
                                        "ambiguous across regions — can't localize")

    def neighbors(self, node_id: int) -> list[int]:
        out = []
        for (a, b) in self.edges:
            if a == node_id:
                out.append(b)
            elif b == node_id:
                out.append(a)
        return sorted(set(out))

    def loop_closures(self) -> list[Edge]:
        return [e for e in self.edges.values() if e.kind == "loop"]

    # ---- consolidation (merge over-fragmented places) ------------------

    def _node_similarity(self, a: int, b: int) -> float:
        """Place-vs-place similarity using the same signals as association:
        best-of-observation caption cosine + object-set Jaccard + relation
        overlap + OCR overlap, blended by the map weights."""
        from .relations import place_relations

        na, nb = self.nodes[a], self.nodes[b]
        w = self.weights

        def jac(x, y):
            return len(x & y) / len(x | y) if (x or y) else 0.0

        cap = 0.0
        if w.caption > 0:
            ma, mb = na.embedding_matrix(), nb.embedding_matrix()
            cap = max(0.0, float((ma @ mb.T).max()))
        obj = jac(na.label_set(), nb.label_set())
        rel = jac(place_relations(na), place_relations(nb))
        ocr = jac(na.ocr_set(), nb.ocr_set())
        denom = w.caption + w.objects + w.ocr + w.relations
        return (w.caption * cap + w.objects * obj + w.ocr * ocr + w.relations * rel) / denom if denom else 0.0

    def _merge_nodes(self, keep: int, drop: int) -> None:
        """Fold node ``drop`` into ``keep``: move observations, redirect edges,
        fix visit order / current pointer."""
        self.nodes[keep].observations.extend(self.nodes[drop].observations)
        del self.nodes[drop]
        # redirect edges through the merge
        new_edges: dict[tuple[int, int], Edge] = {}
        for (x, y), e in self.edges.items():
            x = keep if x == drop else x
            y = keep if y == drop else y
            if x == y:
                continue
            key = (x, y) if x < y else (y, x)
            if key in new_edges:
                ex = new_edges[key]
                ex.count += e.count
                if e.kind == "loop":
                    ex.kind = "loop"
                if e.via and not ex.via:
                    ex.via = e.via
            else:
                new_edges[key] = Edge(key[0], key[1], e.kind, e.count, e.via)
        self.edges = new_edges
        self.visit_order = [keep if v == drop else v for v in self.visit_order]
        if self.current_node_id == drop:
            self.current_node_id = keep

    def consolidate(self, threshold: float | None = None, max_passes: int = 5) -> int:
        """Merge over-fragmented places: repeatedly merge mutual-best node pairs
        whose similarity >= ``threshold``.

        Mutual-best (A's most-similar node is B and vice-versa) is conservative --
        it only merges pairs that each consider the other their top match, which
        avoids chaining unrelated places together. The threshold sits above
        ``match_threshold`` (only confident same-place merges). Returns the number
        of merges performed. Rebuild the landmark index afterwards."""
        if threshold is None:
            threshold = min(0.95, self.match_threshold + 0.12)
        total = 0
        for _ in range(max_passes):
            ids = list(self.nodes)
            if len(ids) < 2:
                break
            best: dict[int, tuple[int, float]] = {}
            for i in range(len(ids)):
                bi, bs = -1, -1.0
                for j in range(len(ids)):
                    if i == j:
                        continue
                    s = self._node_similarity(ids[i], ids[j])
                    if s > bs:
                        bi, bs = ids[j], s
                best[ids[i]] = (bi, bs)
            # collect disjoint mutual-best pairs above threshold
            merged: set[int] = set()
            pairs = []
            for a in ids:
                b, s = best[a]
                if s >= threshold and best.get(b, (-1, 0))[0] == a:
                    lo, hi = sorted((a, b))
                    if lo not in merged and hi not in merged:
                        pairs.append((lo, hi))
                        merged.add(lo)
                        merged.add(hi)
            if not pairs:
                break
            for keep, drop in pairs:
                self._merge_nodes(keep, drop)
                total += 1
        if total:
            self.landmark_index = None  # stale; caller re-indexes
        return total

    def cap_observations(self, k: int) -> int:
        """Decay/forget: keep only the ``k`` most-recent observations per place,
        bounding memory on long runs (and letting a place drift as the
        environment changes). Returns #observations dropped."""
        dropped = 0
        for n in self.nodes.values():
            if len(n.observations) > k:
                keep = sorted(n.observations, key=lambda o: o.seq_index)[-k:]
                dropped += len(n.observations) - len(keep)
                n.observations = keep
        if dropped:
            self.landmark_index = None
        return dropped

    # ---- stage 3: place connectivity (inter-place spatial network) -----

    def link_shared_landmarks(self, max_df: int = 3, min_shared: int = 2) -> int:
        """Link places that share **multiple** rare objects into a
        `shared_landmark` edge.

        A single shared "rare" object over-links in large spaces -- a printer or
        fridge legitimately recurs in distant rooms (multi-room validation showed
        precision collapse from 93%→46% with the single-object rule). Requiring
        ``min_shared`` distinct rare objects (df in [2, max_df]) between the two
        places is much stronger co-location evidence. Requires
        ``index_landmarks`` first. Returns #edges added."""
        if self.landmark_index is None:
            self.index_landmarks(include_ocr=False)
        # accumulate the set of rare tokens each place-pair shares
        pair_shared: dict[tuple[int, int], set[str]] = {}
        for tok, places in self.landmark_index.postings.items():
            if tok in PORTAL_LABELS or not (2 <= len(places) <= max_df):
                continue
            ps = sorted(places)
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    pair_shared.setdefault((ps[i], ps[j]), set()).add(tok)
        added = 0
        for (a, b), toks in pair_shared.items():
            if len(toks) >= min_shared and (a, b) not in self.edges:
                self._add_edge(a, b, "shared_landmark", via=",".join(sorted(toks)))
                added += 1
        return added

    def place_portals(self, node_id: int) -> set[str]:
        """Portal objects (door/opening/...) visible in a place -- passage cues."""
        return self.nodes[node_id].label_set() & PORTAL_LABELS

    def connected_components(self) -> list[set[int]]:
        seen: set[int] = set()
        comps: list[set[int]] = []
        for nid in self.nodes:
            if nid in seen:
                continue
            comp = set(self._hops_from(nid))  # reachable set via BFS
            comps.append(comp)
            seen |= comp
        return comps

    def shortest_path(self, a: int, b: int) -> list[int] | None:
        """BFS shortest path of places from a to b over all edge types."""
        if a == b:
            return [a]
        prev: dict[int, int] = {a: a}
        q = deque([a])
        while q:
            u = q.popleft()
            for v in self.neighbors(u):
                if v not in prev:
                    prev[v] = u
                    if v == b:
                        path = [b]
                        while path[-1] != a:
                            path.append(prev[path[-1]])
                        return path[::-1]
                    q.append(v)
        return None

    def route_text(self, path: list[int]) -> str:
        """Human/LLM-readable route over a place path, annotating portals."""
        if not path:
            return "(no route)"
        lines = []
        for i, nid in enumerate(path):
            portals = self.place_portals(nid)
            tag = f" [portal: {', '.join(sorted(portals))}]" if portals else ""
            lines.append(f"{i+1}. place {nid}{tag}: {self.nodes[nid].summary()}")
        return "\n".join(lines)

    # ---- persistence (text + vectors only -- tiny) ---------------------

    def to_dict(self) -> dict:
        return {
            "match_threshold": self.match_threshold,
            "weights": {
                "caption": self.weights.caption,
                "objects": self.weights.objects,
                "ocr": self.weights.ocr,
                "ocr_present_boost": self.weights.ocr_present_boost,
            },
            "current_node_id": self.current_node_id,
            "visit_order": self.visit_order,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [
                {"a": e.a, "b": e.b, "kind": e.kind, "count": e.count, "via": e.via}
                for e in self.edges.values()
            ],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, embedder) -> SemanticTopoMap:
        with open(path) as f:
            d = json.load(f)
        w = d.get("weights", {})
        m = cls(
            embedder,
            match_threshold=d.get("match_threshold", 0.62),
            weights=ScoreWeights(**w) if w else None,
        )
        for nd in d["nodes"]:
            node = PlaceNode.from_dict(nd)
            m.nodes[node.node_id] = node
            m._next_node_id = max(m._next_node_id, node.node_id + 1)
            for o in node.observations:
                # keep obs id counter ahead
                try:
                    n = int(o.obs_id.split("-")[-1])
                    m._next_obs_id = max(m._next_obs_id, n + 1)
                except ValueError:
                    pass
        for ed in d["edges"]:
            m.edges[(ed["a"], ed["b"])] = Edge(
                ed["a"], ed["b"], ed["kind"], ed.get("count", 1), ed.get("via", "")
            )
        m.visit_order = d.get("visit_order", [])
        m.current_node_id = d.get("current_node_id")
        return m

    def stats(self) -> dict:
        return {
            "frames_ingested": len(self.visit_order),
            "places": len(self.nodes),
            "edges": len(self.edges),
            "loop_closures": len(self.loop_closures()),
            "observations": sum(len(n.observations) for n in self.nodes.values()),
        }
