# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Perception backend: image -> structured, text-only SceneDescription.

Upstream TextSLAM uses Florence-2 (one model emitting caption + detections +
OCR). This XR sample takes the path TextSLAM's own docstring names as the
production target: an OpenAI-compatible VLM served via ``xr-ai-models`` (Cosmos),
prompted for structured JSON. One VLM call per frame yields the caption, an
object inventory, and any visible text; after ``perceive`` returns, the caller
drops the image — only the text survives into the map.

The VLM emits labels without bounding boxes, so ``Detection.bbox`` is ``None``
and the object spatial-relations signal (``scoring.ScoreWeights.relations``)
stays off — caption + object-set + OCR carry the association here.

The ``Perceptor`` protocol is the seam; it is ``async`` because every XR-AI model
call is async (unlike upstream's in-process Florence-2).
"""
from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import httpx
from loguru import logger

from .types import Detection, SceneDescription

# What the VLM is asked to produce per frame. Caption + distinctive objects +
# any signage/screen text — exactly the three signals ``scoring`` blends.
PERCEPTION_PROMPT = (
    "Describe this view so a place-recognition system can tell this spot apart "
    "from other rooms. Reply ONLY as JSON, no prose:\n"
    '{"caption": "<one sentence describing the place and its layout>",\n'
    ' "objects": ["distinctive objects, furniture, appliances, fixtures"],\n'
    ' "text": ["any visible text — signs, labels, room numbers, brand names, on '
    'screens; empty list if none"]}\n'
    "Use short noun phrases. Prefer distinctive, fixed things over generic "
    "walls/floors."
)


def _extract_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of a VLM reply (tolerant of prose/markdown)."""
    if not text:
        return None
    body = text.strip()
    if body.startswith("```"):
        parts = body.split("```")
        if len(parts) >= 2:
            body = parts[1].lstrip("json").strip()
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(body[start:end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _strlist(value: object, limit: int = 8) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = [str(v).strip() for v in value if str(v).strip()]
    out = [v for v in out if v.lower() not in {"none", "no text", "n/a", "na", "[]"}]
    return out[:limit]


@runtime_checkable
class Perceptor(Protocol):
    async def perceive(self, image_url: str, frame_id: str = "") -> SceneDescription:
        ...


class VLMPerceptor:
    """Perceive a frame into text via an XR-AI ``VLMService`` (Cosmos).

    ``image_url`` is a JPEG data URL (see ``pixels.encode_image``). Degrades to an
    empty description on VLM/transport error rather than crashing the tour loop.
    """

    def __init__(self, vlm, system_prompt: str, max_objects: int = 8):
        self._vlm = vlm
        self._system_prompt = system_prompt
        self._max_objects = max_objects

    @property
    def source(self) -> str:
        return getattr(self._vlm, "model", "vlm")

    async def perceive(self, image_url: str, frame_id: str = "") -> SceneDescription:
        raw = ""
        try:
            resp = await self._vlm.ask_image(
                image_url, PERCEPTION_PROMPT, system_prompt=self._system_prompt,
            )
            raw = (resp.content or "").strip()
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001 - degrade, don't crash
            logger.warning("perceptor vlm call failed: {}", exc)
            return SceneDescription(frame_id=frame_id, source=self.source)

        obj = _extract_json_object(raw)
        if obj is None:
            # No JSON — keep the prose as a caption so the frame isn't wasted.
            return SceneDescription(
                frame_id=frame_id, caption=raw[:300], detailed_caption=raw[:300],
                source=self.source,
            )
        caption = str(obj.get("caption", "")).strip()
        objects = [
            Detection(label=lbl)
            for lbl in _strlist(obj.get("objects"), self._max_objects)
        ]
        ocr = _strlist(obj.get("text") or obj.get("ocr"), self._max_objects)
        return SceneDescription(
            frame_id=frame_id,
            caption=caption,
            detailed_caption=caption,
            objects=objects,
            ocr=ocr,
            source=self.source,
        )
