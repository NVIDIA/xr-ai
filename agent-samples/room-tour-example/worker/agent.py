# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RoomTourBrain — text-space SLAM room tour for the unified pipecat pipeline.

A direct port of TextSLAM (github.com/nvddr/textslam) driven live from the XR
camera. The input is a monocular stream with no pose/depth, so instead of
geometry every frame is perceived into *text* — caption + objects + OCR/signage —
the pixels are thrown away, and a semantic-topological place graph
(``textslam.SemanticTopoMap``) is built and relocalized purely from that text.

The textslam package (``worker/textslam/``) is the upstream library carried over
intact (``types``/``scoring``/``relations``/``landmarks``/``topomap``); only the
two model backends are XR-native — ``VLMPerceptor`` (the shared Cosmos VLM via
xr-ai-models) and ``HashingEmbedder``. This brain is the thin glue: it parses
voice commands, feeds the live frames into the map's online SLAM step
(``ingest``), supervises place naming from speech, and answers via relocalization.

How the upstream pieces map to the room-tour UX (always-on voice, no wake word):
  "start room tour"                          → fresh SemanticTopoMap
  "this is the living room" / "...kitchen"   → name the place(s) being scanned;
       a background loop ``ingest``s the live view into the map as the user pans
       (each new node inherits the spoken room label — voice-supervised naming
       layered on textslam's otherwise-unsupervised data association)
  "stop tour"                                → finalize (consolidate + index)
  "where am I"                               → ``relocalize`` the current view
       (the queries below also work *during* the tour — the map is online, so
        the wearer needn't stop the tour to ask)
  "where is the sofa"                        → find the object/sign across nodes
                                               (+ a live left/center/right bearing)
  "take me to the monitor (from here)"       → GUIDED NAVIGATION: resolve the
                                               destination (object → its room, or
                                               a room) to a target node, BFS
                                               (``shortest_path``) from the
                                               relocalized current view, speak the
                                               room-by-room plan, then a live loop
                                               re-localizes each frame and narrates
                                               progress, corrects wrong turns, and
                                               announces arrival with a bearing
                                               (say "stop" to cancel)
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import AsyncIterator

import httpx
import numpy as np
from loguru import logger
from xr_ai_agent import DataMessage, FrameSignal
from xr_ai_models import VLMService
from xr_ai_pipecat import BrainProcessor, GatedQueryFrame
from xr_ai_pipecat.transport import XRMediaHubTransport

from pixels import encode_image, frame_to_pil
from textslam import HashingEmbedder, SceneDescription, SemanticTopoMap, VLMPerceptor

# OPTIONAL monocular pose backbone. The map is text-space (pose-free) by design;
# when this soft dependency and its model assets are present (MONO_SLAM_WORKSPACE
# set), the brain upgrades bearings from the VLM's coarse left/center/right to a
# true geometric bearing and adds a continuous tracking loop. Absent → unchanged
# pose-free behavior. See worker/pose_provider.py.
try:
    from pose_provider import MonoPoseProvider
except Exception:  # noqa: BLE001 - any import failure means "stay pose-free"
    MonoPoseProvider = None  # type: ignore[assignment]

DEFAULT_SYSTEM_PROMPT = (
    "You are a spatial memory assistant for smart glasses. You help the wearer "
    "remember the layout of their home after a guided room tour. Answer in one "
    "short, plain spoken sentence — never JSON, lists, or markdown. Speak in "
    "second person ('you are in…', 'the sofa is to your left'). Do not mention "
    "lighting, mood, or aesthetics."
)

# Command grammar (matched case-insensitively against the STT transcript).
_RE_START   = re.compile(r"\b(start|begin|do)\b.*\b(room\s*tour|tour)\b")
_RE_STOP    = re.compile(
    r"\b(?:stop|end|finish|that'?s it|done)\b.*\btour\b"        # 'stop/end/finish … tour'
    r"|\bstop\w*\s+(?:the\s+)?(?:tour\w*|touring|war|tor)\b"    # 'stop tour' / 'stopped war' (STT mishears)
)
_RE_ROOM    = re.compile(r"\bthis\s+is\s+(?:the\s+|a\s+|my\s+)?(.+)$")
_RE_WHEREAMI = re.compile(r"\bwhere\s+am\s+i\b|\bwhat\s+room\b|\bwhich\s+room\b")
_RE_ROUTE   = re.compile(
    r"\b(?:how\s+(?:do\s+i|to|can\s+i)\s+(?:get|go)|take\s+me|navigate|guide\s+me|"
    r"directions?|route)\b.*?\bto\b\s+(?:the\s+|my\s+|a\s+|an\s+)?(.+)$"
)
_RE_WHEREIS = re.compile(r"\bwhere(?:'?s| is| are| can i find)?\s+(?:the\s+|my\s+|a\s+|an\s+)?(.+)$")
# Cancel an in-progress guided navigation ("stop", "cancel", "we're here"…).
_RE_CANCEL  = re.compile(
    r"\b(?:stop|cancel|never\s*mind|forget\s+it|stop\s+(?:guiding|navigating|navigation)|"
    r"that'?s\s+(?:good|enough|fine)|i'?m\s+here|we'?re\s+here|arrived|here\s+already)\b"
)
# Trailing locational qualifier on a destination ("…from here", "…from where I am").
_RE_FROM_HERE = re.compile(
    r"\s*\b(?:from|starting\s+from)\s+(?:here|where\s+i\s+am|my\s+(?:current\s+)?"
    r"(?:location|spot|position|place))\b.*$|\s*\bright\s+now\b.*$"
)

# Accept the best relocalization candidate only if its raw place-similarity
# clears this floor — a guard against answering from a place that doesn't really
# match the current view (perceptual aliasing / unmapped spot).
_RELOCALIZE_MIN = 0.12


@dataclass
class _NavSession:
    """Live guided-navigation state. The brain holds at most one at a time."""

    pid: str
    target_node: int
    target_label: str          # what the wearer asked for, e.g. "monitor"
    dest_room: str | None      # the named room that target node lives in
    last_key: tuple | None = None   # debounce: last spoken (room, next, hint) state
    last_dist: int | None = None    # last route length (to detect heading away)
    nudge_ticks: int = 0            # ticks since we last said anything


def _now_us() -> int:
    return time.time_ns() // 1_000


def _bearing_phrase(bearing: str) -> str:
    return {
        "left":   "to your left",
        "right":  "to your right",
        "center": "right in front of you",
    }.get(bearing.strip().lower(), "")


def _scene_is_empty(desc: SceneDescription) -> bool:
    return not (desc.caption.strip() or desc.objects or desc.ocr)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{2,}", text.lower())}


class RoomTourBrain(BrainProcessor):
    """Thin orchestration over the VLM perceptor + an in-process textslam map."""

    def __init__(
        self,
        *,
        transport: XRMediaHubTransport,
        vlm: VLMService,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        frame_max_age_s: float = 5.0,
        frame_wait_s: float = 4.0,
        tour_capture_interval_s: float = 3.0,
        match_threshold: float = 0.62,
        nav_monitor_interval_s: float = 4.0,
        nav_max_secs: float = 120.0,
        camera_intrinsics: dict | None = None,
        track_interval_s: float = 0.1,
    ) -> None:
        super().__init__()
        self._transport = transport
        self._vlm = vlm
        self._system_prompt = system_prompt
        self._frame_max_age_us = int(frame_max_age_s * 1_000_000)
        self._frame_wait_s = frame_wait_s
        self._tour_capture_interval_s = tour_capture_interval_s
        self._match_threshold = match_threshold
        self._nav_monitor_interval_s = nav_monitor_interval_s
        self._nav_max_secs = nav_max_secs
        self._track_interval_s = track_interval_s

        # The text-space map (TextSLAM). Perception → text; embedding → vectors.
        self._embedder = HashingEmbedder()
        self._perceptor = VLMPerceptor(vlm, system_prompt)
        self._map: SemanticTopoMap | None = None
        # node_id → spoken room name (voice-supervised naming over the topo graph).
        self._place_names: dict[int, str] = {}

        # OPTIONAL pose backbone (additive geometric sensor over the text map).
        # Built only when the soft dependency imported AND glasses intrinsics were
        # supplied — wrong/absent intrinsics would bias VO + depth, so we refuse to
        # guess them and stay pose-free instead.
        self._pose = self._build_pose_provider(camera_intrinsics)
        self._track_task: asyncio.Task | None = None
        # node_id → room-frame position (mean of the camera positions held while the
        # node's observations were ingested). Populated only while a pose is being
        # tracked; this is what turns a textslam place node into a bearing target.
        self._node_positions: dict[int, np.ndarray] = {}
        self._node_pos_counts: dict[int, int] = {}

        self._touring = False
        self._tour_task: asyncio.Task | None = None
        self._query_active = False
        self._active_label: str | None = None   # room label applied to new nodes

        # Active guided-navigation session (live turn-by-turn). None when idle.
        self._nav: _NavSession | None = None
        self._nav_task: asyncio.Task | None = None

        # Latest FrameSignal per (pid, track_id); newest fresh one feeds the VLM.
        self._latest: dict[tuple[str, str], FrameSignal] = {}
        self._frame_events: dict[str, asyncio.Event] = {}
        self._active_pid: str | None = None

        ep = transport.endpoint
        ep.on_data(self._on_data)
        ep.on_frame(self._on_frame)

    # ── optional pose backbone ─────────────────────────────────────────────────

    @staticmethod
    def _build_pose_provider(camera_intrinsics: dict | None):
        """Construct the optional MonoPoseProvider, or return None to stay pose-free.

        Returns None (no error) when the backbone isn't importable or no glasses
        intrinsics were supplied — both are normal "pose-free" configurations.
        Model loading is deferred to ``start()`` so __init__ never blocks on GPU."""
        if MonoPoseProvider is None or not camera_intrinsics:
            return None
        try:
            from dataset import CameraParams  # backbone type, only present with the extra
            cam = CameraParams(
                fx=float(camera_intrinsics["fx"]), fy=float(camera_intrinsics["fy"]),
                cx=float(camera_intrinsics["cx"]), cy=float(camera_intrinsics["cy"]),
                width=int(camera_intrinsics["width"]), height=int(camera_intrinsics["height"]),
            )
            provider = MonoPoseProvider(camera_params=cam)
            logger.info("mono-slam pose provider configured (intrinsics={})", camera_intrinsics)
            return provider
        except Exception as exc:  # noqa: BLE001 - any failure → pose-free
            logger.warning("pose provider unavailable, staying pose-free: {}", exc)
            return None

    async def _start_pose_provider(self) -> None:
        """Load the backbone models off the event loop (GPU-bound)."""
        if self._pose is None:
            return
        try:
            await asyncio.to_thread(self._pose.start)
            logger.info("mono-slam pose provider started")
        except Exception as exc:  # noqa: BLE001 - degrade to pose-free
            logger.warning("pose provider start failed, staying pose-free: {}", exc)
            self._pose = None

    # ── BrainProcessor overrides ──────────────────────────────────────────────

    async def handle_query(
        self, pid: str, text: str, fresh_match: bool,
    ) -> AsyncIterator[str]:
        # Return (not yield) the async iterator — same contract as SimpleVlmBrain.
        return self._respond(pid, text)

    async def on_participant_left(self, pid: str) -> None:
        self._latest = {k: v for k, v in self._latest.items() if k[0] != pid}
        self._frame_events.pop(pid, None)
        if self._active_pid == pid:
            self._active_pid = None
        self._stop_nav()
        await self._stop_tour_task()

    async def shutdown(self) -> None:
        """Release the pose backbone's GPU models (no-op when pose-free)."""
        self._stop_nav()
        await self._stop_tour_task()
        if self._pose is not None:
            try:
                await asyncio.to_thread(self._pose.stop)
            except Exception:
                logger.opt(exception=True).warning("pose provider stop failed")

    # ── data-channel side path (typed queries) ────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        try:
            text = msg.data.decode(errors="replace").strip()
        except Exception:
            return
        if not text or text.lower() == "ping":
            return
        await self._spawn_query(GatedQueryFrame(
            participant_id=msg.participant_id, text=text,
            fresh_match=True, pts_us=msg.pts_us,
        ))

    # ── frame tracking (always-on camera) ─────────────────────────────────────

    async def _on_frame(self, sig: FrameSignal) -> None:
        self._latest[(sig.participant_id, sig.track_id)] = sig
        self._active_pid = sig.participant_id
        ev = self._frame_events.get(sig.participant_id)
        if ev is not None:
            ev.set()

    def _latest_signal(self, pid: str) -> FrameSignal | None:
        candidates = [v for k, v in self._latest.items() if k[0] == pid]
        return max(candidates, key=lambda s: s.pts_us) if candidates else None

    def _is_fresh(self, sig: FrameSignal) -> bool:
        return _now_us() - sig.pts_us < self._frame_max_age_us

    async def _wait_for_frame(self, pid: str, timeout: float) -> FrameSignal | None:
        ev = self._frame_events.setdefault(pid, asyncio.Event())
        deadline = asyncio.get_event_loop().time() + timeout
        sig = self._latest_signal(pid)
        if sig is not None and self._is_fresh(sig):
            return sig
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return self._latest_signal(pid)
            ev.clear()
            try:
                await asyncio.wait_for(ev.wait(), timeout=min(remaining, 2.0))
            except asyncio.TimeoutError:
                continue
            sig = self._latest_signal(pid)
            if sig is not None and self._is_fresh(sig):
                return sig

    async def _get_frame(self, pid: str):
        sig = self._latest_signal(pid)
        if not (sig and self._is_fresh(sig)):
            sig = await self._wait_for_frame(pid, self._frame_wait_s)
        if sig is None:
            return None
        return await self._transport.endpoint.request_frame(sig)

    async def _vlm_ask(self, frame, question: str) -> str:
        try:
            image_url = encode_image(frame_to_pil(frame))
            resp = await self._vlm.ask_image(
                image_url, question, system_prompt=self._system_prompt,
            )
            return (resp.content or "").strip()
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001 - degrade, don't crash
            logger.warning("vlm ask failed: {}", exc)
            return ""

    # ── perceptor: frame → SceneDescription (caption + objects + OCR) ──────────

    async def _perceive(self, pid: str) -> SceneDescription | None:
        """One TextSLAM perception step: current frame → structured text.
        Returns None only if no frame is available."""
        frame = await self._get_frame(pid)
        if frame is None:
            return None
        image_url = encode_image(frame_to_pil(frame))
        return await self._perceptor.perceive(image_url, frame_id=f"{pid}-{_now_us()}")

    # ── query routing ──────────────────────────────────────────────────────────

    async def _respond(self, pid: str, text: str) -> AsyncIterator[str]:
        self._query_active = True
        try:
            lower = text.lower().strip()
            logger.info("query  pid={!r}  {!r}", pid, text[:80])

            # A guided navigation is live: let the wearer call it off.
            if self._nav is not None and _RE_CANCEL.search(lower):
                self._stop_nav()
                yield "Okay, I'll stop guiding you."
                return

            if _RE_STOP.search(lower):
                yield self._stop_tour(pid)
                return
            if _RE_START.search(lower):
                yield self._start_tour(pid)
                return
            # While touring, "this is the <room>" labels the place being scanned
            # and takes priority. Every other query (where am I / navigate / where
            # is X) works *during* the tour too — the text-space map is queryable
            # online, so the wearer doesn't have to stop the tour to ask.
            if self._touring:
                m = _RE_ROOM.search(lower)
                if m:
                    yield await self._label_place(pid, m.group(1))
                    return
            if _RE_WHEREAMI.search(lower):
                yield await self._answer_where_am_i(pid)
                return
            m = _RE_ROUTE.search(lower)
            if m:
                yield await self._start_navigation(pid, m.group(1))
                return
            m = _RE_WHEREIS.search(lower)
            if m:
                yield await self._answer_where_is(pid, m.group(1))
                return

            yield self._help()
        finally:
            self._query_active = False

    def _help(self) -> str:
        if self._touring:
            return ("We're on a tour. Walk through each room and tell me "
                    "'this is the living room', then say 'stop tour' when done.")
        if not self._place_names:
            return ("Say 'start room tour' and walk me through your space, then "
                    "ask me where things are.")
        return ("Ask me 'where am I', 'where is the sofa', or 'how do I get to "
                "the kitchen'. Say 'start room tour' to map the space again.")

    # ── tour lifecycle ───────────────────────────────────────────────────────

    def _start_tour(self, pid: str) -> str:
        self._stop_nav()
        # A fresh text-space map. weights default to caption .5 / objects .3 /
        # ocr .2; relations stay off (the VLM emits no bounding boxes).
        self._map = SemanticTopoMap(self._embedder, match_threshold=self._match_threshold)
        self._place_names = {}
        self._active_label = None
        self._touring = True
        self._active_pid = pid
        self._node_positions = {}
        self._node_pos_counts = {}
        self._tour_task = asyncio.create_task(self._tour_capture_loop(pid))
        if self._pose is not None:
            self._track_task = asyncio.create_task(self._tracking_loop(pid))
        return ("Room tour started. Walk through your space and tell me each "
                "room — say 'this is the living room' as you enter it. Say "
                "'stop tour' when you're done.")

    def _stop_tour(self, pid: str) -> str:
        was = self._touring
        self._touring = False
        # the capture task is cancelled lazily on next participant_left / restart;
        # mark it for teardown without awaiting (we're in a sync responder branch)
        if self._tour_task is not None and not self._tour_task.done():
            self._tour_task.cancel()
            self._tour_task = None
        if self._track_task is not None and not self._track_task.done():
            self._track_task.cancel()
            self._track_task = None
        if self._map is not None:
            # consolidate over-fragmented places, then sharpen the landmark index
            # used by "where is X" — the upstream end-of-build steps.
            self._map.consolidate()
            self._reconcile_place_names()
            self._map.index_landmarks()
            logger.info("tour finalized: {}", self._map.stats())
        if not was and not self._place_names:
            return "There's no tour running. Say 'start room tour' to begin."
        if not self._place_names:
            return "Tour stopped, but I didn't catch any rooms."
        rooms = sorted(set(self._place_names.values()))
        return (f"Tour complete. I mapped: {', '.join(rooms)}. "
                "Ask me where things are.")

    async def _stop_tour_task(self) -> None:
        for attr in ("_tour_task", "_track_task"):
            task = getattr(self, attr)
            setattr(self, attr, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    def _reconcile_place_names(self) -> None:
        """After ``consolidate`` merges nodes, some named node_ids disappear.
        Drop names whose node no longer exists (its observations were folded into
        the surviving node, which already carries a name from the same room)."""
        if self._map is None:
            return
        self._place_names = {
            nid: name for nid, name in self._place_names.items()
            if nid in self._map.nodes
        }

    async def _label_place(self, pid: str, raw_name: str) -> str:
        name = raw_name.strip().rstrip(".?! ").strip()
        if not name or self._map is None:
            return "Which room is this?"
        self._active_label = name
        # Seed the place from the current view (the loop adds more as the user pans).
        desc = await self._perceive(pid)
        if desc is not None and not _scene_is_empty(desc):
            self._ingest_labeled(desc)
        # if the seed frame wasn't available, name whatever node we're currently in
        elif self._map.current_node_id is not None:
            self._place_names[self._map.current_node_id] = name
        return f"Got it — {name}."

    def _ingest_labeled(self, desc: SceneDescription) -> None:
        """Fold one description into the map (the online SLAM step) and tag the
        resulting node with the room label currently being spoken."""
        assert self._map is not None
        result = self._map.ingest(desc)
        if self._active_label is not None:
            self._place_names[result.node_id] = self._active_label
        self._record_node_position(result.node_id)
        logger.info(
            "ingest → node {} ({}) room={!r} caption={!r} objects={} text={}",
            result.node_id, result.reason, self._active_label,
            desc.caption[:60], sorted(desc.object_labels()), sorted(desc.ocr_tokens()),
        )

    async def _tour_capture_loop(self, pid: str) -> None:
        """While touring, periodically perceive the live frame and ingest it into
        the map (TextSLAM's incremental build). No-ops until a room is named."""
        logger.info("tour capture loop started  pid={!r}", pid)
        try:
            while self._touring:
                await asyncio.sleep(self._tour_capture_interval_s)
                if not self._touring or self._active_label is None:
                    continue
                if self._query_active:
                    continue  # yield VLM bandwidth to the user-facing query
                desc = await self._perceive(pid)
                if desc is not None and not _scene_is_empty(desc):
                    self._ingest_labeled(desc)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("tour capture loop error")

    # ── optional pose tracking (continuous, ~10 Hz) ────────────────────────────
    #
    # The VLM perceive loop is slow (~3 s) but monocular VO needs a continuous
    # stream to stay tracked, so this is a SEPARATE cheap loop. It only runs when
    # a pose provider is configured; on first frame it lazily loads the backbone
    # models, then feeds every fresh frame so a pose is always current for
    # bearings/relocalization. Tracking loss (pose None) is fine — the agent just
    # falls back to the VLM path whenever there's no current pose.

    async def _tracking_loop(self, pid: str) -> None:
        if self._pose is None:
            return
        await self._start_pose_provider()
        if self._pose is None:           # start() failed → degraded to pose-free
            return
        self._pose.reset_session()
        logger.info("pose tracking loop started pid={!r}", pid)
        try:
            while self._touring:
                await asyncio.sleep(self._track_interval_s)
                if not self._touring:
                    break
                sig = self._latest_signal(pid)
                if sig is None or not self._is_fresh(sig):
                    continue
                fd = await self._transport.endpoint.request_frame(sig)
                if fd is None:
                    continue
                try:
                    est = await asyncio.to_thread(
                        self._pose.ingest_frame, fd, pts_us=sig.pts_us, frame_id=sig.seq,
                    )
                except Exception:
                    logger.exception("pose ingest failed; continuing pose-free this tick")
                    continue
                if est.lost:
                    logger.debug("pose lost (frame seq={})", sig.seq)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("pose tracking loop error")

    def _record_node_position(self, node_id: int) -> None:
        """Fold the current camera position into a place node's running mean
        position. No-op without a current (non-lost) pose — the node simply has no
        geometric anchor and bearings to it fall back to the VLM."""
        if self._pose is None:
            return
        est = self._pose.current_pose
        if est is None or est.lost:
            return
        p = est.position()
        if p is None:
            return
        n = self._node_pos_counts.get(node_id, 0)
        if n == 0:
            self._node_positions[node_id] = p.astype(np.float64)
        else:
            self._node_positions[node_id] = (self._node_positions[node_id] * n + p) / (n + 1)
        self._node_pos_counts[node_id] = n + 1

    def _target_world_xyz(self, node_id: int) -> np.ndarray | None:
        """Room-frame position for a place node, if one was recorded during the
        tour. Falls back to a neighbor's position for an unlabeled transition node
        that itself was never pose-anchored."""
        pos = self._node_positions.get(node_id)
        if pos is not None:
            return pos
        if self._map is None:
            return None
        neighbors = [
            self._node_positions[nbr]
            for nbr in self._map.neighbors(node_id)
            if nbr in self._node_positions
        ]
        if neighbors:
            return np.mean(np.stack(neighbors), axis=0)
        return None

    def _pose_bearing_phrase(self, node_id: int) -> str | None:
        """Geometric bearing phrase from the current pose to a place node, or None
        when there is no current pose / no anchored position (→ VLM fallback)."""
        if self._pose is None:
            return None
        est = self._pose.current_pose
        if est is None or est.lost:
            return None
        tgt = self._target_world_xyz(node_id)
        if tgt is None:
            return None
        bearing = self._pose.bearing_to(tgt)
        return bearing.phrase() if bearing else None

    # ── queries over the finished map ──────────────────────────────────────────

    def _room_of(self, node_id: int) -> str | None:
        """Spoken room name for a node — its own label, else a graph neighbor's
        (transition nodes between rooms may be unlabeled)."""
        if self._map is None:
            return None
        name = self._place_names.get(node_id)
        if name:
            return name
        for nbr in self._map.neighbors(node_id):
            if nbr in self._place_names:
                return self._place_names[nbr]
        return None

    async def _relocalize_node(self, pid: str) -> int | None:
        """Text-space relocalization: perceive the current frame, score it
        against the map, return the best-matching place *node* id (or None).
        This is the shared primitive behind 'where am I' and route start."""
        if self._map is None or not self._map.nodes:
            return None
        desc = await self._perceive(pid)
        if desc is None or _scene_is_empty(desc):
            return None
        ranked = self._map.relocalize(desc, top_k=3)
        if not ranked or ranked[0].total < _RELOCALIZE_MIN:
            return None
        logger.info("relocalized → node {}  score={:.3f}", ranked[0].node_id, ranked[0].total)
        return ranked[0].node_id

    async def _relocalize(self, pid: str) -> str | None:
        """'Where am I' room name = the room of the best-matching node."""
        node_id = await self._relocalize_node(pid)
        return self._room_of(node_id) if node_id is not None else None

    async def _answer_where_am_i(self, pid: str) -> str:
        if self._map is None or not self._place_names:
            return "I don't have a map yet. Say 'start room tour' first."
        room = await self._relocalize(pid)
        if room is None:
            return "I'm not sure which room this is."
        # Optional fusion: the appearance relocalizer abstains by default, so it
        # never contradicts the text answer — but when a pose is tracked and it
        # abstains, soften the phrasing (the two sensors don't both agree).
        if self._pose is not None and self._pose.current_pose is not None:
            reloc = self._pose.relocalize()
            if not reloc.localized:
                return f"I think you're in the {room}, but I'm not fully sure."
        return f"You're in the {room}."

    def _find_object(self, obj_name: str) -> tuple[int, str] | None:
        """Locate an object/sign across the map by text. Returns the
        (node_id, matched_label) of the best place containing it, preferring a
        named room. Mirrors the landmark/entity lookup used for recall."""
        if self._map is None:
            return None
        query = _tokens(obj_name)
        if not query:
            return None
        named: tuple[int, str] | None = None
        any_match: tuple[int, str] | None = None
        for node_id, node in self._map.nodes.items():
            for label in sorted(node.label_set() | node.ocr_set()):
                ltok = _tokens(label)
                if query & ltok or obj_name in label or label in obj_name:
                    any_match = any_match or (node_id, label)
                    if node_id in self._place_names or self._room_of(node_id):
                        named = named or (node_id, label)
        return named or any_match

    async def _answer_where_is(self, pid: str, raw_obj: str) -> str:
        obj_name = raw_obj.strip().rstrip(".?! ").strip()
        if not obj_name:
            return "Where is what?"
        if self._map is None or not self._place_names:
            return "I don't have a map yet. Say 'start room tour' first."

        hit = self._find_object(obj_name)
        if hit is None:
            return f"I didn't see a {obj_name} during the tour."
        node_id, label = hit
        room = self._room_of(node_id)
        if room is None:
            return f"I saw a {label}, but I'm not sure which room it's in."

        current = await self._relocalize(pid)
        if current is not None and current == room:
            # Prefer the geometric pose bearing (true SE3 direction) when a pose is
            # tracked and the node is anchored; otherwise read a coarse L/C/R from
            # the live frame via the VLM.
            bearing = self._pose_bearing_phrase(node_id)
            if not bearing:
                frame = await self._get_frame(pid)
                if frame is not None:
                    raw = await self._vlm_ask(frame, (
                        f"Is there a {label} visible in this image? If yes, is it to "
                        "the left, in the center, or to the right? Reply with exactly "
                        "one word: left, center, right, or no."
                    ))
                    bearing = _bearing_phrase(raw)
            if bearing:
                return f"The {label} is in the {room}, {bearing}."
            return f"The {label} is here in the {room}."
        return f"The {label} is in the {room}."

    # ── node-level navigation ('how do I get to X') ────────────────────────────
    #
    # No new graph is needed: the map's place-node graph (built by ``ingest`` —
    # a temporal edge from each frame to the next, plus loop closures) is the
    # route network, and ``SemanticTopoMap.shortest_path`` is BFS over it. We
    # relocalize the current view to a start node, resolve the spoken
    # destination to a target node, BFS between them, and read each node on the
    # path as a spoken visual cue (its caption + any door/opening portal).

    def _describe_node(self, node_id: int) -> str:
        """Short spoken visual cue for one place node along a route — the VLM
        caption (which carries the distinctive 'red sofa', 'white walls' detail),
        falling back to the room label."""
        node = self._map.nodes[node_id]
        cue = node.summary().strip()
        if cue:
            return cue
        room = self._room_of(node_id)
        return f"the {room}" if room else "the next area"

    def _resolve_target_node(self, start: int, dest: str) -> int | None:
        """Resolve a spoken destination to a target node. A room name → the
        nearest node carrying that label (the first spot of that room you'd
        reach); otherwise an object/sign → the node containing it."""
        dtok = _tokens(dest)
        room_nodes = [
            nid for nid in self._map.nodes
            if (r := self._room_of(nid)) and (dest == r.lower() or bool(dtok & _tokens(r)))
        ]
        if room_nodes:
            reachable = [
                (len(p), nid) for nid in room_nodes
                if (p := self._map.shortest_path(start, nid)) is not None
            ]
            return min(reachable)[1] if reachable else room_nodes[0]
        hit = self._find_object(dest)
        return hit[0] if hit else None

    def _phrase_route(self, path: list[int]) -> str:
        """Turn a node path into spoken turn-by-turn directions. Each hop is a
        visual cue; a node with a door/opening portal is phrased as going
        through it. Consecutive identical cues are de-duplicated (the map
        fragments a room into several near-identical nodes)."""
        target = path[-1]
        steps: list[str] = []
        prev: str | None = None
        for nid in path[1:-1]:
            portals = self._map.place_portals(nid)
            cue = self._describe_node(nid)
            cue = f"through the {sorted(portals)[0]} to {cue}" if portals else cue
            if cue == prev:
                continue
            steps.append(cue)
            prev = cue
        dest = self._describe_node(target)
        if not steps:
            return f"From here, head straight to {dest}."
        return "From here, head to " + ", then ".join(steps) + f", and you'll reach {dest}."

    def _clean_dest(self, raw: str) -> str:
        """Strip a trailing locational qualifier ('… from here') so 'take me to
        the monitor from here' resolves the object 'monitor', not 'monitor from
        here'."""
        return _RE_FROM_HERE.sub("", raw).strip().rstrip(".?! ").strip().lower()

    def _rooms_along(self, path: list[int]) -> list[str]:
        """Ordered, de-duplicated named rooms a node path passes through."""
        out: list[str] = []
        for nid in path:
            r = self._room_of(nid)
            if r and (not out or out[-1] != r):
                out.append(r)
        return out

    def _next_named_room(self, path: list[int], cur_room: str | None) -> str | None:
        """The next named room to head for along ``path`` (BFS toward target)."""
        rooms = self._rooms_along(path)
        if cur_room and cur_room in rooms:
            i = rooms.index(cur_room)
            return rooms[i + 1] if i + 1 < len(rooms) else None
        return rooms[0] if rooms else None

    def _transit_hint(self, node_id: int) -> str | None:
        """A spoken movement cue for a transit spot, read from its perceived
        text: 'down the hallway' / 'across the open space' / 'through the door'."""
        node = self._map.nodes[node_id]
        text = (node.summary() + " " + " ".join(sorted(node.label_set()))).lower()
        if "hall" in text or "corridor" in text:
            return "down the hallway"
        if "open" in text:
            return "across the open space"
        portals = self._map.place_portals(node_id)
        if portals:
            return f"through the {sorted(portals)[0]}"
        return None

    # ── guided navigation: 'take me to the monitor (from here)' ─────────────────
    #
    # One-shot directions aren't enough for AR glasses — the wearer is walking.
    # So a route starts a live session: speak the initial plan, then a background
    # loop re-localizes each fresh frame, narrates progress space-by-space,
    # corrects wrong turns, and announces arrival with a live bearing to the
    # target object. Proactive speech is pushed straight to TTS via _push_text.

    async def _start_navigation(self, pid: str, raw_dest: str) -> str:
        dest = self._clean_dest(raw_dest)
        if not dest:
            return "Where would you like to go?"
        if self._map is None or not self._place_names:
            return "I don't have a map yet. Say 'start room tour' first."
        start = await self._relocalize_node(pid)
        if start is None:
            return "I'm not sure where you are right now — look around and ask again."
        target = self._resolve_target_node(start, dest)
        if target is None:
            return f"I didn't see {dest} during the tour."
        dest_room = self._room_of(target)
        if target == start or (dest_room and self._room_of(start) == dest_room):
            return f"You're already in the {dest_room or 'right area'} — look around for the {dest}."
        path = self._map.shortest_path(start, target)
        if not path or len(path) < 2:
            return (f"I can't find a route to {dest} — those spots weren't "
                    "connected during the tour.")

        self._stop_nav()
        start_room = self._room_of(start)
        nav = _NavSession(pid=pid, target_node=target, target_label=dest, dest_room=dest_room)
        nav.last_dist = len(path)
        nav.last_key = (start_room, self._next_named_room(path, start_room), None)
        self._nav = nav
        self._nav_task = asyncio.create_task(self._nav_monitor_loop(pid))
        logger.info("nav start pid={!r} node {} → {} (room={!r}) path={}",
                    pid, start, target, dest_room, path)
        return self._nav_intro(start_room, path, dest)

    def _nav_intro(self, start_room: str | None, path: list[int], target_label: str) -> str:
        """Initial spoken route plan as a sequence of named rooms to pass."""
        rooms = self._rooms_along(path)
        seq = [r for r in rooms if r != start_room] or rooms
        if not seq:                       # destination room unnamed → node-level
            return self._phrase_route(path)
        head = f"You're in the {start_room}. " if start_room else ""
        tail = seq[-1]
        # If the destination *is* a room (e.g. "take me to the office"), don't
        # tack on a redundant "where the office is".
        dest_is_room = target_label == tail or target_label in tail
        if len(seq) == 1:
            body = (f"the {tail} is just ahead" if dest_is_room
                    else f"the {target_label} is in the {tail}, just ahead")
        else:
            through = "go through the " + ", then the ".join(seq[:-1])
            body = (f"{through}, and you'll reach the {tail}" if dest_is_room
                    else f"{through}, and you'll reach the {tail} where the {target_label} is")
        return f"{head}To get to the {target_label}, {body}. I'll guide you as you walk."

    def _nav_progress_phrase(self, node: int, cur_room: str | None, next_room: str | None) -> str:
        hint = self._transit_hint(node)
        toward = f"toward the {next_room}" if next_room else "ahead"
        if hint:
            return f"Keep going {hint}, {toward}."
        if cur_room:
            return f"You're in the {cur_room} now. Keep heading {toward}."
        return f"Keep heading {toward}."

    async def _nav_say(self, pid: str, text: str) -> None:
        """Proactively speak a guidance sentence (pushed straight to TTS)."""
        if not text:
            return
        if not text.rstrip().endswith((".", "!", "?")):
            text = text.rstrip() + "."
        logger.info("nav say pid={!r}: {!r}", pid, text)
        try:
            await self._push_text(text, pid=pid)
        except Exception:
            logger.exception("nav say failed")

    async def _nav_arrival(self, pid: str, nav: _NavSession) -> None:
        room = nav.dest_room or "your destination"
        # Destination is a room itself → just announce arrival, no object bearing.
        if nav.dest_room and (nav.target_label == nav.dest_room or nav.target_label in nav.dest_room):
            await self._nav_say(pid, f"You've arrived at the {room}.")
            return
        bearing = self._pose_bearing_phrase(nav.target_node)
        if not bearing:
            frame = await self._get_frame(pid)
            if frame is not None:
                raw = await self._vlm_ask(frame, (
                    f"Is there a {nav.target_label} visible in this image? If yes, is "
                    "it to the left, in the center, or to the right? Reply with exactly "
                    "one word: left, center, right, or no."
                ))
                bearing = _bearing_phrase(raw)
        if bearing:
            await self._nav_say(pid, f"You've reached the {room}. The {nav.target_label} is {bearing}.")
        else:
            await self._nav_say(pid, f"You've reached the {room}. Look around — the "
                                     f"{nav.target_label} should be here.")

    async def _nav_monitor_loop(self, pid: str) -> None:
        """Watch the live frames and narrate the walk until arrival/cancel."""
        nav = self._nav
        if nav is None:
            return
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self._nav_max_secs
        logger.info("nav monitor loop started pid={!r} → {!r}", pid, nav.target_label)
        try:
            while self._nav is nav:
                await asyncio.sleep(self._nav_monitor_interval_s)
                if self._nav is not nav:
                    return
                if loop.time() > deadline:
                    await self._nav_say(pid, "I'll stop guiding now — ask again if you still need directions.")
                    break
                if self._query_active:
                    continue  # a user-facing turn is talking; don't speak over it
                node = await self._relocalize_node(pid)
                if node is None:
                    continue  # couldn't place this frame; wait for a clearer view
                cur_room = self._room_of(node)
                if node == nav.target_node or (nav.dest_room and cur_room == nav.dest_room):
                    await self._nav_arrival(pid, nav)
                    break
                path = self._map.shortest_path(node, nav.target_node)
                if not path or len(path) < 2:
                    await self._nav_arrival(pid, nav)
                    break
                next_room = self._next_named_room(path, cur_room)
                hint = self._transit_hint(node)
                dist = len(path)
                moved_away = nav.last_dist is not None and dist > nav.last_dist + 1
                nav.last_dist = dist
                key = (cur_room, next_room, hint)
                if moved_away:
                    await self._nav_say(pid, "That doesn't look right — turn around. "
                                        + self._nav_progress_phrase(node, cur_room, next_room))
                    nav.last_key, nav.nudge_ticks = key, 0
                elif key != nav.last_key:
                    await self._nav_say(pid, self._nav_progress_phrase(node, cur_room, next_room))
                    nav.last_key, nav.nudge_ticks = key, 0
                else:
                    nav.nudge_ticks += 1
                    if nav.nudge_ticks >= 3:
                        nav.nudge_ticks = 0
                        nr = next_room or nav.dest_room or nav.target_label
                        await self._nav_say(pid, f"Keep going toward the {nr}.")
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("nav monitor loop error")
        finally:
            if self._nav is nav:
                self._nav = None
                self._nav_task = None

    def _stop_nav(self) -> None:
        self._nav = None
        task, self._nav_task = self._nav_task, None
        if task is not None and not task.done():
            task.cancel()
