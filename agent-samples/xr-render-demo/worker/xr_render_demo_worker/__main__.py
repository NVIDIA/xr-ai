# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr-render-demo agent worker — voice-driven sphere.

Per AudioChunk: EMA-smoothed RMS → POST /sphere/radius (continuous, ~50 Hz).
Per utterance (energy VAD): STT → LLM → action list → render-mcp.

The LLM emits user-frame coordinates (+x right, -z forward); the worker
fetches head pose from oxr-mcp once per utterance and rotates by yaw +
translates by position before forwarding world-space to render-mcp. So
"to my left" stays correct after the user walks around.

Launched as a subprocess by ``uv run xr_render_demo``.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import math
import pathlib
import re
import signal
import time
import wave
from dataclasses import dataclass, field

import httpx
import numpy as np
import yaml
from fastmcp import Client as McpClient

from xr_ai_agent import (
    AudioChunk, DataMessage, ParticipantEvent, ProcessorEndpoint, Subscribe,
)


def _tool_payload(result) -> dict | list | None:
    """Extract the dict/list a FastMCP tool returned. Tool results expose
    either ``.data`` (newer fastmcp) or ``.structured_content`` (older)."""
    if hasattr(result, "data") and result.data is not None:
        return result.data
    return getattr(result, "structured_content", None)

log = logging.getLogger("xr_render_demo")

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"

_XR_SESSION_STARTED_TOPIC = "xr.session.started"
_RENDER_READY_TOPIC       = "render.ready"



# System prompt for the action LLM. The calibration examples are deliberate
# — Minitron-8B-Instruct tends to produce 0.1 m default magnitudes for bare
# directional words ("left", "down") without them, and confuses sphere.move
# vs sphere.position when both look applicable. Trim with care.
_ACTIONS_SYSTEM_PROMPT = (
    "You output a JSON action list to control a sphere in a 3D scene.\n"
    "\n"
    "Output ONLY one JSON object on one line: {\"actions\": [...]}.\n"
    "No prose, no markdown, no preamble. Empty list if the user said nothing actionable.\n"
    "\n"
    "Actions (pick zero or more):\n"
    '  {"op":"sphere.color",    "value":[r,g,b]}            RGB floats 0..1.\n'
    '  {"op":"sphere.position", "value":[x,y,z]}            Absolute target, metres.\n'
    '  {"op":"sphere.move",     "value":[dx,dy,dz]}         Relative delta, metres.\n'
    '  {"op":"sphere.gaze",     "value":{"distance":F}}     Place along the user\'s gaze.\n'
    '  {"op":"sphere.reset"}                                Restore defaults.\n'
    "\n"
    "Coordinates are USER-FRAME, relative to the user's current head pose:\n"
    "  +x / -x  -> user's right / left\n"
    "  +y / -y  -> up / down\n"
    "  -z / +z  -> in front of / behind the user\n"
    "The user's head is at the origin. Distances are metres.\n"
    "\n"
    "Use sphere.move when the request is relative to where the sphere currently is\n"
    "(\"left\", \"up\", \"a bit closer\"). Use sphere.position when it's absolute relative\n"
    "to the user (\"in front of me\", \"on my left\"). Use sphere.gaze for\n"
    "\"where I'm looking\".\n"
    "\n"
    "DEFAULT MAGNITUDE for sphere.move is at least 1.0 metre. Use a smaller\n"
    "magnitude ONLY when the user explicitly says so — \"a tiny bit / a hair /\n"
    "slightly / a centimeter / an inch\" or a numeric quantity. Plain bare\n"
    "directional words (\"left\", \"up\", \"forward\") get the 1m default.\n"
    "\n"
    "Calibration examples — every axis sign appears at least once. \"Right\"\n"
    "alone is the directional (+x), NOT an acknowledgement; treat all six\n"
    "primary directions symmetrically:\n"
    '  "left"             -> {"actions":[{"op":"sphere.move","value":[-1.0, 0.0, 0.0]}]}\n'
    '  "right"            -> {"actions":[{"op":"sphere.move","value":[ 1.0, 0.0, 0.0]}]}\n'
    '  "up"               -> {"actions":[{"op":"sphere.move","value":[ 0.0, 1.0, 0.0]}]}\n'
    '  "down"             -> {"actions":[{"op":"sphere.move","value":[ 0.0,-1.0, 0.0]}]}\n'
    '  "forward"          -> {"actions":[{"op":"sphere.move","value":[ 0.0, 0.0,-1.0]}]}\n'
    '  "back"             -> {"actions":[{"op":"sphere.move","value":[ 0.0, 0.0, 1.0]}]}\n'
    '  "down one meter"   -> {"actions":[{"op":"sphere.move","value":[ 0.0,-1.0, 0.0]}]}\n'
    '  "two meters right" -> {"actions":[{"op":"sphere.move","value":[ 2.0, 0.0, 0.0]}]}\n'
    '  "a tiny bit left"  -> {"actions":[{"op":"sphere.move","value":[-0.1, 0.0, 0.0]}]}\n'
    '  "a hair down"      -> {"actions":[{"op":"sphere.move","value":[ 0.0,-0.1, 0.0]}]}\n'
    '  "way over there"   -> {"actions":[{"op":"sphere.move","value":[ 0.0, 0.0,-3.0]}]}\n'
)


def _extract_json_object(text: str) -> str | None:
    """Outermost balanced ``{...}`` from *text*, skipping braces inside
    double-quoted strings. Returns None if nothing balances."""
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    return None


def _now_us() -> int:
    return time.time_ns() // 1_000


def _chunks_to_wav(chunks: list[AudioChunk]) -> bytes:
    """Float32 chunks → 16-bit PCM WAV. Assumes all chunks share format
    (true today; revisit if mid-utterance codec changes ever become real)."""
    raw = b"".join(c.data for c in chunks)
    arr = np.frombuffer(raw, dtype=np.float32)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(chunks[0].channels)
        wf.setsampwidth(2)
        wf.setframerate(chunks[0].sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# World-frame clamps applied post-transform so a hallucinated LLM number
# can't yeet the sphere to (1e9, 1e9, 1e9). Generous on purpose — y goes
# negative because user-frame y=-1.5 ("at my feet") plus a seated head at
# world y≈1.2 yields a valid world y≈-0.3.
_POS_BOUNDS = {
    "x": (-10.0, 10.0),
    "y": ( -2.0,  5.0),
    "z": (-10.0, 10.0),
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _clamp_pos(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        _clamp(xyz[0], *_POS_BOUNDS["x"]),
        _clamp(xyz[1], *_POS_BOUNDS["y"]),
        _clamp(xyz[2], *_POS_BOUNDS["z"]),
    )


# ── Head-pose math ────────────────────────────────────────────────────────────
# OpenXR Y-up; quaternion is (qx, qy, qz, qw). User-frame is world translated
# to head position and rotated by head YAW only (so head tilt doesn't make
# "left" diagonal). ``sphere.gaze`` uses the full orientation to honour pitch.


@dataclass
class HeadPose:
    is_valid:    bool
    position:    tuple[float, float, float]
    orientation: tuple[float, float, float, float]   # (qx, qy, qz, qw)
    ts:          int                                  # ms since epoch

    @classmethod
    def from_response(cls, body: dict) -> HeadPose:
        return cls(
            is_valid    = bool(body.get("is_valid", False)),
            position    = tuple(float(v) for v in body.get("position",    (0.0, 0.0, 0.0))),
            orientation = tuple(float(v) for v in body.get("orientation", (0.0, 0.0, 0.0, 1.0))),
            ts          = int(body.get("ts", 0)),
        )


def _yaw_from_quat(q: tuple[float, float, float, float]) -> float:
    """Yaw (Y-axis rotation, radians) from a unit quaternion (x,y,z,w)."""
    qx, qy, qz, qw = q
    return math.atan2(2.0 * (qw * qy + qx * qz),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _yaw_rotate(local: tuple[float, float, float], yaw: float) -> tuple[float, float, float]:
    """Rotate a vector around world +Y by *yaw* radians."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    lx, ly, lz = local
    return (cy * lx + sy * lz,
            ly,
            -sy * lx + cy * lz)


def _user_to_world(local: tuple[float, float, float], head: HeadPose) -> tuple[float, float, float]:
    yaw = _yaw_from_quat(head.orientation)
    rx, ry, rz = _yaw_rotate(local, yaw)
    return (head.position[0] + rx,
            head.position[1] + ry,
            head.position[2] + rz)


def _world_to_user(world: tuple[float, float, float], head: HeadPose) -> tuple[float, float, float]:
    """Inverse of _user_to_world; used to give the LLM the sphere's current
    user-frame position so it can scale relative magnitudes."""
    yaw = _yaw_from_quat(head.orientation)
    rel = (world[0] - head.position[0],
           world[1] - head.position[1],
           world[2] - head.position[2])
    return _yaw_rotate(rel, -yaw)


def _forward_from_quat(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """OpenXR local-forward (0,0,-1) rotated by *q*. Used by sphere.gaze
    so it honours pitch as well as yaw."""
    qx, qy, qz, qw = q
    return (
        -2.0 * (qx * qz + qw * qy),
         2.0 * (qw * qx - qy * qz),
        -(1.0 - 2.0 * (qx * qx + qy * qy)),
    )


def _parse_xyz(val: object) -> tuple[float, float, float] | None:
    """Accept ``[x, y, z]`` as a list/tuple of three numerics."""
    if not isinstance(val, (list, tuple)) or len(val) != 3:
        return None
    try:
        return (float(val[0]), float(val[1]), float(val[2]))
    except (TypeError, ValueError):
        return None


def _parse_actions_response(text: str) -> list[dict]:
    """Validate and normalise the LLM's action list. Invalid actions are
    dropped silently. User-frame coords are NOT clamped here — that happens
    after the user→world transform at dispatch time. Output shape:

      {"op": "sphere.color",    "rgb":    [r, g, b]}
      {"op": "sphere.position", "uxyz":   [x, y, z]}     # user-frame
      {"op": "sphere.move",     "udelta": [dx,dy,dz]}    # user-frame
      {"op": "sphere.gaze",     "distance": F}
      {"op": "sphere.reset"}
    """
    if not text:
        return []
    obj_text = _extract_json_object(text)
    if obj_text is None:
        return []
    try:
        obj = json.loads(obj_text)
    except json.JSONDecodeError:
        return []

    raw_actions = obj.get("actions")
    if not isinstance(raw_actions, list):
        return []

    out: list[dict] = []
    for a in raw_actions:
        if not isinstance(a, dict):
            continue
        op = a.get("op")
        val = a.get("value")
        if op == "sphere.color":
            rgb = _parse_xyz(val)
            if rgb is None:
                continue
            # Clamp in case the LLM emits 0..255 or out-of-range.
            r, g, b = (_clamp(rgb[0], 0.0, 1.0),
                       _clamp(rgb[1], 0.0, 1.0),
                       _clamp(rgb[2], 0.0, 1.0))
            out.append({"op": op, "rgb": [r, g, b]})
        elif op == "sphere.position":
            xyz = _parse_xyz(val)
            if xyz is None:
                continue
            out.append({"op": op, "uxyz": list(xyz)})
        elif op == "sphere.move":
            delta = _parse_xyz(val)
            if delta is None:
                continue
            out.append({"op": op, "udelta": list(delta)})
        elif op == "sphere.gaze":
            # Accept scalar distance, {"distance": F}, or {} (default 1.5).
            distance = 1.5
            if isinstance(val, (int, float)):
                distance = float(val)
            elif isinstance(val, dict):
                d = val.get("distance")
                if isinstance(d, (int, float)):
                    distance = float(d)
            elif val is None:
                pass
            else:
                continue
            out.append({"op": op, "distance": _clamp(distance, 0.1, 6.0)})
        elif op == "sphere.reset":
            out.append({"op": op})
        # Unknown ops fall through silently so a stale worker tolerates new ops.
    return out


@dataclass
class _VoiceState:
    ema:         float = 0.0
    chunks:      list[AudioChunk] = field(default_factory=list)
    speech_s:    float = 0.0
    silent_s:    float = 0.0
    sample_rate: int   = 16000
    channels:    int   = 1
    stt_busy:    bool  = False     # one in-flight STT call per participant
    radius_in_flight: bool = False  # one in-flight /sphere/radius POST
    # VAD diagnostics
    in_speech:        bool  = False
    rms_window_max:   float = 0.0
    last_stats_time:  float = field(default_factory=time.monotonic)


class XRRenderDemoAgent:
    """Audio→sphere bridge: continuous radius + STT colour + LLM-driven position."""

    def __init__(self, cfg: dict) -> None:
        # AUDIO + DATA only — we don't process video frames.
        self._ep = ProcessorEndpoint(
            sub_addr=_HUB_PUB, push_addr=_HUB_PUSH,
            filter=Subscribe.AUDIO | Subscribe.DATA,
        )
        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_participant(self._on_participant)

        # Sphere mapping
        self._radius_base    = float(cfg.get("radius_base", 0.05))
        self._radius_gain    = float(cfg.get("radius_gain", 5.0))
        self._radius_min     = float(cfg.get("radius_min",  0.03))
        self._radius_max     = float(cfg.get("radius_max",  1.5))
        self._ema_alpha      = float(cfg.get("ema_alpha",   0.30))

        # VAD
        self._vad_threshold  = float(cfg.get("silence_threshold", 0.005))
        self._vad_silence_s  = float(cfg.get("silence_duration",  0.8))
        self._vad_min_s      = float(cfg.get("min_speech",        0.15))

        # Service endpoints. STT + LLM speak OpenAI REST; render-mcp speaks
        # both REST (/sphere/radius) and MCP (/mcp); oxr-mcp speaks MCP.
        # McpClients are opened long-lived in run().
        render_base          = cfg.get("render_mcp_url", "http://localhost:8220").rstrip("/")
        self._stt_url        = cfg.get("stt_server",     "http://localhost:8103").rstrip("/") + "/v1/audio/transcriptions"
        self._llm_url        = cfg.get("llm_server",     "http://localhost:8101").rstrip("/") + "/v1/chat/completions"
        self._radius_url     = render_base + "/sphere/radius"
        self._render_mcp_url = render_base + "/mcp"
        self._oxr_mcp_url    = cfg.get("oxr_mcp_url",    "http://localhost:8230").rstrip("/") + "/mcp"

        self._http: httpx.AsyncClient = httpx.AsyncClient(timeout=10.0)
        self._render: McpClient | None = None
        self._oxr:    McpClient | None = None

        self._voice: dict[str, _VoiceState] = {}
        self._xr_started = False

        # Tracked locally so sphere.move (a user-frame delta) can be applied
        # to the sphere's current world position. Initialised to LOVR's
        # default anchor; reset on sphere.reset.
        self._sphere_world_pos: tuple[float, float, float] = (0.0, 1.6, -1.5)

    # ── audio path ────────────────────────────────────────────────────────────

    async def _on_audio(self, chunk: AudioChunk) -> None:
        if chunk.samples <= 0 or not chunk.data:
            return
        samples = np.frombuffer(chunk.data, dtype=np.float32)
        if samples.size == 0:
            return

        pid = chunk.participant_id
        vs  = self._voice.setdefault(
            pid,
            _VoiceState(sample_rate=chunk.sample_rate, channels=chunk.channels),
        )

        # Continuous: EMA → radius. sqrt() expands quiet speech across the
        # range so the user doesn't have to yell to drive against radius_max.
        rms = float(np.sqrt(np.mean(np.square(samples))))
        a = self._ema_alpha
        vs.ema = (1.0 - a) * vs.ema + a * rms
        loudness = float(np.sqrt(vs.ema))
        target = self._radius_base + self._radius_gain * loudness
        target = max(self._radius_min, min(self._radius_max, target))
        # Skip until LOVR is up (render-mcp would drop anyway), and drop
        # new POSTs while one is in flight so a stall can't queue stale
        # radii on the worker side.
        if self._xr_started and not vs.radius_in_flight:
            vs.radius_in_flight = True
            asyncio.create_task(
                self._post_radius(vs, target),
                name="render-radius",
            )

        # Event-driven: VAD slices utterances.
        chunk_s = chunk.samples / max(chunk.sample_rate, 1)
        is_speech = rms >= self._vad_threshold
        if is_speech:
            if not vs.in_speech:
                log.info("vad pid=%r  speech START  rms=%.4f thr=%.4f",
                         pid, rms, self._vad_threshold)
                vs.in_speech = True
            vs.chunks.append(chunk)
            vs.speech_s += chunk_s
            vs.silent_s  = 0.0
        else:
            if vs.chunks:
                vs.chunks.append(chunk)  # keep trailing silence for natural endpoint
            vs.silent_s += chunk_s

        vs.rms_window_max = max(vs.rms_window_max, rms)
        now = time.monotonic()
        if now - vs.last_stats_time >= 1.0:
            log.info("vad pid=%r  max_rms=%.4f  thr=%.4f  speech=%.2fs  silent=%.2fs",
                     pid, vs.rms_window_max, self._vad_threshold,
                     vs.speech_s, vs.silent_s)
            vs.rms_window_max = 0.0
            vs.last_stats_time = now

        if (vs.silent_s >= self._vad_silence_s
                and vs.speech_s >= self._vad_min_s
                and not vs.stt_busy):
            log.info("vad pid=%r  speech END  spoken=%.2fs  silent=%.2fs  chunks=%d",
                     pid, vs.speech_s, vs.silent_s, len(vs.chunks))
            utterance      = vs.chunks[:]
            vs.chunks      = []
            vs.speech_s    = vs.silent_s = 0.0
            vs.in_speech   = False
            vs.stt_busy    = True
            asyncio.create_task(
                self._handle_utterance(pid, utterance, vs),
                name=f"xr_render_demo-stt-{pid}",
            )
        elif (vs.silent_s >= self._vad_silence_s
                and 0.0 < vs.speech_s < self._vad_min_s):
            log.info("vad pid=%r  drop short blip  spoken=%.2fs (< %.2fs min) — "
                     "lower min_speech in xr_render_demo_worker.yaml if this is your speech",
                     pid, vs.speech_s, self._vad_min_s)
            vs.chunks    = []
            vs.speech_s  = 0.0
            vs.silent_s  = 0.0
            vs.in_speech = False

    async def _handle_utterance(
        self, pid: str, chunks: list[AudioChunk], vs: _VoiceState,
    ) -> None:
        try:
            wav = _chunks_to_wav(chunks)
            transcript = await self._transcribe(wav)
            if not transcript.strip():
                return

            log.info("stt pid=%r  transcript=%s", pid, transcript)

            # Fetch pose once per utterance, fall back to identity if oxr-mcp
            # isn't up yet so the demo still works in desktop testing.
            pose = await self._fetch_pose()
            if pose is None or not pose.is_valid:
                log.info("pose pid=%r  unavailable — using identity transform", pid)
                pose = HeadPose(
                    is_valid    = True,
                    position    = (0.0, 1.6, 0.0),
                    orientation = (0.0, 0.0, 0.0, 1.0),
                    ts          = 0,
                )
            else:
                log.info("pose pid=%r  pos=(%.2f,%.2f,%.2f)  yaw=%.1f deg",
                         pid, pose.position[0], pose.position[1], pose.position[2],
                         math.degrees(_yaw_from_quat(pose.orientation)))

            try:
                actions = await self._classify_actions(transcript, pose)
            except httpx.HTTPError as exc:
                log.error("llm error pid=%r: %s", pid, exc)
                return

            if not actions:
                log.info("actions pid=%r  none", pid)
                return

            # Sequential dispatch: sphere.move reads + writes _sphere_world_pos
            # so concurrent moves would race and only one delta would land.
            log.info("actions pid=%r  count=%d  %s",
                     pid, len(actions),
                     "  ".join(self._format_action(a) for a in actions))
            for a in actions:
                await self._dispatch_action(pid, a, pose)
        except httpx.HTTPError as exc:
            log.error("stt error pid=%r: %s", pid, exc)
        finally:
            vs.stt_busy = False

    async def _dispatch_action(self, pid: str, action: dict, pose: HeadPose) -> None:
        op = action["op"]
        if op == "sphere.color":
            r, g, b = action["rgb"]
            await self._call_render("set_sphere_color", {"r": r, "g": g, "b": b})
            return

        if op == "sphere.position":
            world = _clamp_pos(_user_to_world(tuple(action["uxyz"]), pose))
            await self._call_render("set_sphere_position",
                                    {"x": world[0], "y": world[1], "z": world[2]})
            self._sphere_world_pos = world
            return

        if op == "sphere.move":
            yaw = _yaw_from_quat(pose.orientation)
            dx, dy, dz = _yaw_rotate(tuple(action["udelta"]), yaw)
            sx, sy, sz = self._sphere_world_pos
            world = _clamp_pos((sx + dx, sy + dy, sz + dz))
            await self._call_render("set_sphere_position",
                                    {"x": world[0], "y": world[1], "z": world[2]})
            self._sphere_world_pos = world
            return

        if op == "sphere.gaze":
            distance = float(action["distance"])
            fx, fy, fz = _forward_from_quat(pose.orientation)
            world = _clamp_pos((
                pose.position[0] + fx * distance,
                pose.position[1] + fy * distance,
                pose.position[2] + fz * distance,
            ))
            await self._call_render("set_sphere_position",
                                    {"x": world[0], "y": world[1], "z": world[2]})
            self._sphere_world_pos = world
            return

        if op == "sphere.reset":
            await self._call_render("reset_sphere", {})
            self._sphere_world_pos = (0.0, 1.6, -1.5)   # LOVR's default
            return

        log.debug("dispatch pid=%r  unknown op=%r", pid, op)

    @staticmethod
    def _format_action(action: dict) -> str:
        op = action["op"]
        if op == "sphere.color":
            r, g, b = action["rgb"]
            return f"color=({r:.2f},{g:.2f},{b:.2f})"
        if op == "sphere.position":
            x, y, z = action["uxyz"]
            return f"pos_u=({x:+.2f},{y:+.2f},{z:+.2f})"
        if op == "sphere.move":
            dx, dy, dz = action["udelta"]
            return f"move_u=({dx:+.2f},{dy:+.2f},{dz:+.2f})"
        if op == "sphere.gaze":
            return f"gaze({action['distance']:.2f}m)"
        if op == "sphere.reset":
            return "reset"
        return op

    # ── pose ──────────────────────────────────────────────────────────────────

    async def _fetch_pose(self) -> HeadPose | None:
        """Call oxr-mcp's get_head_pose tool; None on transport failure."""
        if self._oxr is None:
            return None
        try:
            res  = await self._oxr.call_tool("get_head_pose", {})
            data = _tool_payload(res)
            if not isinstance(data, dict):
                log.warning("oxr-mcp get_head_pose returned non-dict: %r", data)
                return None
            return HeadPose.from_response(data)
        except Exception as exc:
            log.warning("oxr-mcp get_head_pose failed: %s", exc)
            return None

    # ── data path ─────────────────────────────────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        if msg.topic != _XR_SESSION_STARTED_TOPIC:
            return
        # Already up — likely a reconnect / second tab; just re-ack.
        if self._xr_started:
            await self._ep.send_return_data(DataMessage(
                participant_id=msg.participant_id,
                topic=_RENDER_READY_TOPIC,
                pts_us=_now_us(),
                data=b"",
            ))
            return

        log.info("%s received from %s — start_xr → render-mcp",
                 msg.topic, msg.participant_id)
        start_res = await self._call_render("start_xr", {})
        if start_res is None:
            log.warning("start_xr failed — render-mcp will not spawn LOVR")
            return
        if start_res.get("status") == "error":
            log.error("start_xr reported error: %s", start_res.get("error"))
            return

        log.info("start_xr accepted (status=%s) — polling get_health.lovr_started …",
                 start_res.get("status"))
        if not await self._wait_lovr_started():
            return
        self._xr_started = True
        log.info("render.ready — LOVR spawned, sending ack to %s",
                 msg.participant_id)
        await self._ep.send_return_data(DataMessage(
            participant_id=msg.participant_id,
            topic=_RENDER_READY_TOPIC,
            pts_us=_now_us(),
            data=b"",
        ))

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if not event.joined:
            self._voice.pop(event.participant_id, None)

    # ── render-mcp tool helpers ───────────────────────────────────────────────

    async def _call_render(self, tool: str, args: dict, *,
                           log_errors: bool = True) -> dict | None:
        """Invoke a render-mcp MCP tool. Returns the tool's payload dict
        on success, None on transport failure."""
        if self._render is None:
            if log_errors:
                log.error("render-mcp client not connected (tool=%s)", tool)
            return None
        try:
            res  = await self._render.call_tool(tool, args)
            data = _tool_payload(res)
            if not isinstance(data, dict):
                if log_errors:
                    log.error("render-mcp %s returned non-dict: %r", tool, data)
                return None
            return data
        except Exception as exc:
            if log_errors:
                log.error("render-mcp %s failed: %s", tool, exc)
            return None

    async def _post_radius(self, vs: _VoiceState, value: float) -> None:
        """Single-flight POST /sphere/radius."""
        try:
            try:
                resp = await self._http.post(self._radius_url, json={"value": value})
                if resp.is_error:
                    log.debug("render-mcp /sphere/radius → %d", resp.status_code)
            except httpx.HTTPError as exc:
                log.debug("render-mcp /sphere/radius failed: %s", exc)
        finally:
            vs.radius_in_flight = False

    async def _wait_lovr_started(self, *, timeout_s: float = 120.0) -> bool:
        """Poll render-mcp's get_health until lovr_started flips (the spawn
        runs as a background task on render-mcp; see start_xr)."""
        deadline = asyncio.get_running_loop().time() + timeout_s
        interval = 0.5
        while asyncio.get_running_loop().time() < deadline:
            health = await self._call_render("get_health", {}, log_errors=False)
            if health is not None:
                if health.get("lovr_started"):
                    return True
                if health.get("spawn_error"):
                    log.error("render-mcp get_health reports spawn_error: %s",
                              health["spawn_error"])
                    return False
            await asyncio.sleep(interval)
        log.warning("render-mcp start_xr: lovr_started never reported true within %.0fs",
                    timeout_s)
        return False

    async def _transcribe(self, wav_bytes: bytes) -> str:
        resp = await self._http.post(
            self._stt_url,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"response_format": "json"},
            timeout=30.0,
        )
        if resp.is_error:
            log.error("stt %s: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()
        return resp.json().get("text", "")

    async def _classify_actions(self, transcript: str, pose: HeadPose) -> list[dict]:
        """Ask llm-server for a JSON-only ``{actions: [...]}`` response. We
        pass the sphere's current user-frame position as context so qualifiers
        like "even further" scale against current distance."""
        ux, uy, uz = _world_to_user(self._sphere_world_pos, pose)
        distance = math.sqrt(ux * ux + uy * uy + uz * uz)
        context = (
            f"Sphere is currently at user-frame ({ux:+.2f}, {uy:+.2f}, {uz:+.2f}) "
            f"({distance:.2f} m from you)."
        )
        body = {
            "model": "llm",
            "messages": [
                {"role": "system", "content": _ACTIONS_SYSTEM_PROMPT},
                {"role": "system", "content": context},
                {"role": "user",   "content": transcript},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        resp = await self._http.post(self._llm_url, json=body, timeout=30.0)
        if resp.is_error:
            log.error("llm %s: %s", resp.status_code, resp.text[:300])
            resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        log.debug("llm raw response: %r", text)
        return _parse_actions_response(text)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        # render-mcp + oxr-mcp clients stay open for the worker's lifetime so
        # the per-chunk set_sphere_radius hot path doesn't pay session setup
        # each call.
        async with McpClient(self._render_mcp_url) as render, \
                   McpClient(self._oxr_mcp_url) as oxr:
            self._render = render
            self._oxr    = oxr
            try:
                await self._ep.run()
            finally:
                self._render = None
                self._oxr    = None

    async def aclose(self) -> None:
        await self._http.aclose()

    def shutdown(self) -> None:
        # Sync (signal-safe); schedule the async http close best-effort.
        self._ep.stop()
        self._ep.close()
        try:
            asyncio.get_running_loop().create_task(self.aclose())
        except RuntimeError:
            pass


# ── entry point ───────────────────────────────────────────────────────────────

async def _wait_for_health(name: str, url: str, *, blocking: bool) -> None:
    """Poll *url* until it returns 200. Loops forever; when *blocking* is
    False, log nothing while waiting (intended to run as a bg task)."""
    interval = 5.0
    while True:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(url)
                if r.is_success:
                    log.info("%s ready", name)
                    return
        except httpx.HTTPError:
            pass
        if blocking:
            log.info("waiting for %s at %s …", name, url)
        await asyncio.sleep(interval)


async def _wait_for_services(cfg: dict) -> None:
    """Block on STT (the speech path is dead without it). Warm LLM in the
    background so the worker can serve the radius/xr-start paths immediately
    while ~16 GB of weights download."""
    stt_health = cfg.get("stt_server", "http://localhost:8103").rstrip("/") + "/health"
    llm_health = cfg.get("llm_server", "http://localhost:8101").rstrip("/") + "/health"
    asyncio.create_task(
        _wait_for_health("LLM", llm_health, blocking=False),
        name="llm-warmup",
    )
    await _wait_for_health("STT", stt_health, blocking=True)


async def main(cfg: dict) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # The per-chunk /sphere/radius POSTs (~50/s) would drown the log.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    await _wait_for_services(cfg)

    agent = XRRenderDemoAgent(cfg)
    loop  = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    log.info("xr_render_demo connecting sub=%s push=%s", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        agent.shutdown()
    log.info("xr_render_demo stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    asyncio.run(main(cfg))


if __name__ == "__main__":
    run()
