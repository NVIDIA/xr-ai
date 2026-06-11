# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core data types for text-space SLAM.

The defining constraint of this system: once a frame is perceived into a
``SceneDescription``, the pixels are discarded. Every downstream structure
here is pure text + a text embedding. The "map" is therefore tiny and
human-readable -- it is how a person remembers where they are, not a point
cloud.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# Tokenizer for OCR / signage overlap. Lowercased alphanumeric runs of length
# >= 2 (drops single chars and punctuation noise).
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

# Low-information OCR tokens: English stopwords + web boilerplate. These show up
# as spurious high-IDF "landmarks" (screen text, watermarks) and mislead
# relocalization, so they are dropped from token sets.
_STOPWORDS = frozenset(
    "the a an of and or to in on at for with from by is are was it this that "
    "as be or no www com http https org net html www2".split()
)


def tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


@dataclass(slots=True)
class Detection:
    """One detected object: a label and a normalized bbox (x1, y1, x2, y2)."""

    label: str
    bbox: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class SceneDescription:
    """Structured, text-only perception of a single frame.

    This is the entire footprint a frame leaves behind. The image is gone.
    """

    frame_id: str
    caption: str = ""
    detailed_caption: str = ""
    objects: list[Detection] = field(default_factory=list)
    ocr: list[str] = field(default_factory=list)
    source: str = ""

    def object_labels(self) -> set[str]:
        return {o.label.strip().lower() for o in self.objects if o.label.strip()}

    def ocr_tokens(self) -> set[str]:
        toks: set[str] = set()
        for line in self.ocr:
            toks |= tokenize(line)
        return toks

    def to_embedding_text(self) -> str:
        """Canonical text fed to the embedder.

        Caption-led, with the object inventory appended. Kept deterministic
        (sorted labels) so the same scene yields the same string. OCR is *not*
        folded in here -- it is scored separately and more heavily (signage is
        the strongest relocalization cue), see ``scoring``.
        """
        cap = self.detailed_caption or self.caption
        labels = sorted(self.object_labels())
        parts = [cap.strip()]
        if labels:
            parts.append("Objects: " + ", ".join(labels) + ".")
        return " ".join(p for p in parts if p).strip()

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "caption": self.caption,
            "detailed_caption": self.detailed_caption,
            "objects": [{"label": o.label, "bbox": o.bbox} for o in self.objects],
            "ocr": self.ocr,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SceneDescription:
        return cls(
            frame_id=d["frame_id"],
            caption=d.get("caption", ""),
            detailed_caption=d.get("detailed_caption", ""),
            objects=[
                Detection(label=o["label"], bbox=tuple(o["bbox"]) if o.get("bbox") else None)
                for o in d.get("objects", [])
            ],
            ocr=d.get("ocr", []),
            source=d.get("source", ""),
        )


@dataclass(slots=True)
class Observation:
    """A perceived frame plus its embedding, as stored inside a place node."""

    obs_id: str
    description: SceneDescription
    embedding: np.ndarray  # L2-normalized
    seq_index: int = -1

    def to_dict(self) -> dict:
        return {
            "obs_id": self.obs_id,
            "description": self.description.to_dict(),
            "embedding": self.embedding.astype(np.float32).tolist(),
            "seq_index": self.seq_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Observation:
        return cls(
            obs_id=d["obs_id"],
            description=SceneDescription.from_dict(d["description"]),
            embedding=np.asarray(d["embedding"], dtype=np.float32),
            seq_index=d.get("seq_index", -1),
        )


@dataclass(slots=True)
class Entity:
    """A place-scoped landmark: a recurring object within one place.

    Identity is text-only (label + supporting OCR), not an appearance vector.
    ``support`` = how many of the place's observations contain this label, which
    is the persistence signal (high support = stable landmark; support 1 =
    likely transient). Cross-place recurrence (the same physical door seen from
    two places) is handled at query time via the landmark index, not by a shared
    Entity object -- that belongs to the connectivity stage.
    """

    label: str
    support: int = 1
    ocr: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"label": self.label, "support": self.support, "ocr": self.ocr}

    @classmethod
    def from_dict(cls, d: dict) -> Entity:
        return cls(label=d["label"], support=d.get("support", 1), ocr=d.get("ocr", []))


@dataclass(slots=True)
class PlaceNode:
    """A place: a cluster of observations believed to be the same location.

    We deliberately keep *all* observations (no centroid). Matching is
    best-of-observations, which survives viewpoint variance far better than a
    blurred mean vector.
    """

    node_id: int
    observations: list[Observation] = field(default_factory=list)

    def add(self, obs: Observation) -> None:
        self.observations.append(obs)

    def entities(self) -> dict[str, Entity]:
        """Aggregate this place's observations into place-scoped landmarks.

        Computed from the observations (the source of truth), so it always
        reflects current evidence. Support = #observations containing the label.
        """
        counts: dict[str, int] = {}
        for o in self.observations:
            for lbl in o.description.object_labels():
                counts[lbl] = counts.get(lbl, 0) + 1
        return {lbl: Entity(label=lbl, support=n) for lbl, n in counts.items()}

    def tokens(self) -> set[str]:
        """All landmark tokens for indexing: object labels + OCR tokens."""
        return self.label_set() | self.ocr_set()

    def label_set(self) -> set[str]:
        s: set[str] = set()
        for o in self.observations:
            s |= o.description.object_labels()
        return s

    def ocr_set(self) -> set[str]:
        s: set[str] = set()
        for o in self.observations:
            s |= o.description.ocr_tokens()
        return s

    def embedding_matrix(self) -> np.ndarray:
        return np.stack([o.embedding for o in self.observations])

    def summary(self) -> str:
        """Short human/LLM-readable description of the place.

        Prefers a caption if the perceptor produced one; otherwise (object-only
        detectors) falls back to the most-persistent objects, so detector-based
        maps are still describable."""
        rep = max(
            self.observations,
            key=lambda o: len(o.description.to_embedding_text()),
        )
        text = (rep.description.detailed_caption or rep.description.caption).strip()
        if not text:
            ents = sorted(self.entities().values(), key=lambda e: -e.support)
            top = [e.label for e in ents[:6]]
            text = "area with " + ", ".join(top) if top else "an area"
        ocr = sorted(self.ocr_set())
        if ocr:
            text += f" (visible text: {', '.join(ocr)})"
        return text

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "observations": [o.to_dict() for o in self.observations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> PlaceNode:
        return cls(
            node_id=d["node_id"],
            observations=[Observation.from_dict(o) for o in d["observations"]],
        )
