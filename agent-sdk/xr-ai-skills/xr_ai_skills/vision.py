# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable live-camera vision Q&A for agent brains.

``VisionModule`` is the "answer a question about what the camera sees" feature,
factored out of the individual samples (simple-vlm-example, xr-render-demo, …)
that each used to re-implement it. It owns the VLM call — fetch the freshest
frame (via :class:`~xr_ai_skills.frame_source.LiveFrameSource`), encode it,
and stream the answer. Callers that need a completed value use ``perceive()``,
which collects that same stream into a string.

Camera streaming is always-on (the client streams continuously); this module
never sends ``startCamera`` / ``stopCamera`` control messages.

A brain builds a ``VisionModule`` when it has a VLM service to back it. Voice
pipelines use ``stream`` so sentence-batched TTS can start before generation
finishes; tool loops use ``perceive`` when they need a completed string.

The module is framework-agnostic: it talks to the hub through a
``ProcessorEndpoint`` (subscribing to ``FrameSignal`` events and fetching frames)
and has no dependency on pipecat. A pipecat brain passes ``transport.endpoint``;
a non-pipecat agent passes its own endpoint.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from loguru import logger
from xr_ai_agent import ProcessorEndpoint
from xr_ai_models import VLMService

from .frame_source import LiveFrameSource, LiveFrameUnavailable
from .pixels import encode_image, frame_to_pil


class VisionUnavailable(Exception):
    """Raised by :meth:`VisionModule.perceive` when a live frame can't be turned
    into a VLM answer (no frame available, frame fetch failed, or VLM errored).
    The message is a short, user-facing sentence suitable to speak."""


class VisionModule:
    """Live-camera VLM question answering.

    Camera streaming is always-on — this module does not send camera control
    messages.  It waits up to ``frame_timeout_s`` for a fresh frame to arrive
    before raising :class:`VisionUnavailable`.

    Parameters
    ----------
    endpoint:
        The ``ProcessorEndpoint`` to talk to the hub through; the module
        subscribes to frame signals and fetches frames on demand.
        A pipecat brain passes ``transport.endpoint``.
    vlm:
        A ``VLMService`` (its token stream is used to answer).
    system_prompt:
        Default system prompt for the VLM (overridable per call).
    frame_max_age_s:
        Maximum age of a cached frame signal before it is considered stale.
    frame_timeout_s:
        How long to wait for a fresh frame before raising
        :class:`VisionUnavailable`.
    """

    def __init__(
        self,
        endpoint: ProcessorEndpoint,
        vlm: VLMService,
        *,
        system_prompt: str = "",
        frame_max_age_s: float = 2.0,
        frame_timeout_s: float = 5.0,
    ) -> None:
        self._frames = LiveFrameSource(
            endpoint, frame_max_age_s=frame_max_age_s, frame_timeout_s=frame_timeout_s,
        )
        self._vlm = vlm
        self._system_prompt = system_prompt

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def register(self) -> None:
        """Subscribe to the endpoint's frame signals. Call once at setup."""
        self._frames.register()

    def release(self, pid: str) -> None:
        """Drop all per-participant state (call from ``on_participant_left``)."""
        self._frames.release(pid)

    # ── frame acquisition ──────────────────────────────────────────────────────

    async def _acquire_image_url(self, pid: str) -> str:
        """Wait for a fresh frame, fetch and encode it to a JPEG data URL.
        Raises :class:`VisionUnavailable` if no usable frame arrives in time."""
        try:
            frame = await self._frames.get_frame(pid)
        except LiveFrameUnavailable as exc:
            raise VisionUnavailable(str(exc)) from exc
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: encode_image(frame_to_pil(frame)),
        )

    # ── the VLM call ────────────────────────────────────────────────────────────

    async def stream(
        self, pid: str, query: str, *, system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """Acquire a fresh frame and stream VLM answer chunks.

        Raises :class:`VisionUnavailable` (with a speakable message) on no
        frame, VLM error, or an empty answer.

        Does not touch the agent-status badge — the caller owns its own status
        (it is typically mid-turn doing other work), so this method stays out
        of it.
        """
        t0 = time.monotonic()
        image_url = await self._acquire_image_url(pid)   # raises VisionUnavailable
        has_content = False
        try:
            async for chunk in self._vlm.stream(
                image_url, query, system_prompt=system_prompt or self._system_prompt,
            ):
                if chunk.strip():
                    has_content = True
                yield chunk
        except Exception as exc:
            logger.error("vlm-server error: {}", exc)
            raise VisionUnavailable("VLM server unavailable — please retry.") from exc
        finally:
            logger.info("vision call pid={!r} elapsed={:.2f}s", pid, time.monotonic() - t0)
        if not has_content:
            raise VisionUnavailable("I couldn't make out anything in the view.")

    async def perceive(
        self, pid: str, query: str, *, system_prompt: str | None = None,
    ) -> str:
        """Collect the canonical VLM stream and return a completed string."""
        chunks = [
            chunk
            async for chunk in self.stream(
                pid, query, system_prompt=system_prompt,
            )
        ]
        return "".join(chunks).strip()
