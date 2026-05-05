# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pipecat FrameProcessors for xr-render-demo.

Pipeline:
  InputTransport → SttProcessor → RenderSceneProcessor → TtsProcessor → OutputTransport

SttProcessor
  Silero VAD (falls back to adaptive energy) → per-utterance STT → TranscriptionFrame.

RenderSceneProcessor
  TranscriptionFrame → parallel quick-ack + multi-step agentic loop.

  Agentic loop (max _MAX_LOOP iterations):
    - LLM outputs {"think": "...", "tool": "<name>", "args": {...}}  → execute tool, continue
    - LLM outputs {"done": true, "response": "..."}                  → finish
  Tools route to render-mcp (scene ops) or oxr-mcp (spatial helpers).
  Each tool call sends a brief progress message so the user isn't left waiting.

TtsProcessor
  TextFrame → sentence-batched synthesis → hub return audio.
"""
from __future__ import annotations

import asyncio
import json
import logging
import string
import time
from pathlib import Path

import httpx
import numpy as np
from fastmcp import Client as McpClient
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None           # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from xr_ai_agent import DataMessage
from xr_ai_pipecat.audio import stream_sentences_to_audio
from xr_ai_pipecat.services import SttClient, TtsClient
from xr_ai_pipecat.transport import XRMediaHubTransport, SAMPLE_RATE

from config import WorkerConfig

log = logging.getLogger("xr_render_demo.processors")

# Dedicated trace logger — writes clean session transcripts to a file.
# Key events: user speech, pre-fetched context, think flag, tool calls +
# results, agent response, validation.  Tail or paste the file to debug.
_trace_log = logging.getLogger("xr_render_demo.trace")

def _setup_trace_log(path: str = "/tmp/xr-agent-trace.log") -> None:
    """Call once at worker startup to attach the trace file handler."""
    h = logging.FileHandler(path, mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    _trace_log.addHandler(h)
    _trace_log.setLevel(logging.DEBUG)
    _trace_log.propagate = False
    _trace_log.info("=== trace started ===")

_SILERO_WINDOW  = 512    # 32 ms at 16 kHz
_MAX_UTT_S      = 30.0
_PRE_ROLL_CHUNKS = 10   # ~320 ms pre-roll; prepended when speech onset is detected
_MAX_LOOP       = 10     # visual queries need up to 5 steps; give headroom
_FILLER_PHRASES = frozenset({
    "mm-hmm", "mm hmm", "uh huh", "uh-huh", "uh", "um", "ah", "oh", "eh",
    "huh", "hmm", "yeah", "yep", "yup", "okay", "ok", "right",
    "sure", "thanks", "thank you",
})


# Tools managed by the worker directly (control-plane); excluded from the
# LLM tool list. get_scene_state is intentionally NOT here — the model must
# call it to discover object ids before any manipulation.
_WORKER_MANAGED_TOOLS = frozenset({"start_xr", "get_health"})

# Tools served by oxr-mcp (routed there, not to render-mcp).
_OXR_TOOLS = frozenset({"get_head_pose", "position_ahead", "position_relative"})

# Tools served by vlm-mcp and video-mcp.
_VLM_TOOLS   = frozenset({"ask_image"})
_VIDEO_TOOLS = frozenset({
    "get_latest_frame", "get_frame_from_time",
    "list_live_participants", "list_recorded_participants",
    "get_video_stats", "query_video",
})

# Brief human-readable progress message shown while a tool runs.
_TOOL_PROGRESS: dict[str, str] = {
    "get_head_pose":    "Checking your position...",
    "position_ahead":   "Computing gaze position...",
    "position_relative":"Computing relative position...",
    "get_scene_state":  "Scanning the scene...",
    "add_primitive":    "Creating object...",
    "update_primitive": "Updating object...",
    "remove_primitive": "Removing object...",
}

_AGENT_RESPONSE_TOPIC  = "agent.response"
_AGENT_PROGRESS_TOPIC  = "agent.progress"


class _SceneNotReadyError(Exception):
    """Raised when render-mcp returns not_started — LOVR hasn't launched yet."""


def _now_us() -> int:
    return time.time_ns() // 1_000


# ── SttProcessor ──────────────────────────────────────────────────────────────

class SttProcessor(FrameProcessor):
    """Silero VAD (with adaptive energy fallback) + utterance-level STT."""

    def __init__(
        self,
        stt: SttClient,
        transport: XRMediaHubTransport,
        cfg: WorkerConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._stt       = stt
        self._transport = transport
        self._cfg       = cfg

        self._buffer:        list[bytes] = []
        self._buffer_samples = 0
        self._speech_s  = 0.0
        self._silent_s  = 0.0
        self._speaking  = False
        self._stt_busy  = False
        # Circular pre-roll: always keep the last N chunks before speech onset.
        # Prepended to the utterance buffer so the first word's attack is captured.
        self._pre_roll:  list[bytes] = []

        self._silero = None
        try:
            from silero_vad import load_silero_vad
            self._silero = load_silero_vad(onnx=True)
            log.info("Silero VAD loaded")
        except Exception as exc:
            log.warning("Silero VAD unavailable (%s) — using adaptive energy VAD", exc)

        self._silero_buf:  np.ndarray = np.zeros(0, np.float32)
        self._noise_floor: float = 0.001

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self._feed(frame)
        else:
            await self.push_frame(frame, direction)

    async def _feed(self, frame: InputAudioRawFrame) -> None:
        pcm     = frame.audio
        n_s     = len(pcm) // 2
        chunk_s = n_s / max(SAMPLE_RATE, 1)

        if self._silero is not None and _TORCH_AVAILABLE:
            f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            self._silero_buf = np.concatenate([self._silero_buf, f32])
            speech_prob = 0.0
            while len(self._silero_buf) >= _SILERO_WINDOW:
                window = self._silero_buf[:_SILERO_WINDOW]
                self._silero_buf = self._silero_buf[_SILERO_WINDOW:]
                tensor = _torch.from_numpy(np.ascontiguousarray(window))
                speech_prob = max(speech_prob,
                                  float(self._silero(tensor, SAMPLE_RATE)))
            is_speech = speech_prob > self._cfg.silero_threshold
        else:
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0
            if not self._speaking and not self._buffer:
                self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
            eff_thr   = max(self._cfg.silence_threshold,
                            self._noise_floor * self._cfg.vad_noise_mult)
            is_speech = rms >= eff_thr

        if is_speech:
            if not self._speaking:
                log.info("speech START")
                self._speaking = True
                # Prepend pre-roll so the first word's attack isn't clipped.
                if self._pre_roll:
                    pre_bytes = b"".join(self._pre_roll)
                    self._buffer.insert(0, pre_bytes)
                    self._buffer_samples += len(pre_bytes) // 2
                    self._pre_roll.clear()
            self._buffer.append(pcm)
            self._buffer_samples += n_s
            self._speech_s += chunk_s
            self._silent_s  = 0.0
        else:
            if self._speaking:
                self._buffer.append(pcm)
                self._buffer_samples += n_s
                self._silent_s += chunk_s
            else:
                # Not speaking — maintain a rolling pre-roll window.
                self._pre_roll.append(pcm)
                if len(self._pre_roll) > _PRE_ROLL_CHUNKS:
                    self._pre_roll.pop(0)

        utt_s = self._buffer_samples / max(SAMPLE_RATE, 1)
        if self._speaking and utt_s > _MAX_UTT_S:
            log.info("max utterance length — finalising")
            await self._finalize()
            return

        if (self._speaking
                and self._speech_s >= self._cfg.min_speech
                and self._silent_s >= self._cfg.silence_duration
                and not self._stt_busy):
            await self._finalize()

    async def _finalize(self) -> None:
        if not self._buffer:
            self._speaking = False
            return
        audio_bytes = b"".join(self._buffer)
        self._buffer.clear()
        self._buffer_samples = 0
        self._speaking  = False
        self._silent_s  = 0.0
        self._speech_s  = 0.0
        self._stt_busy  = True
        dur_s = len(audio_bytes) // 2 / max(SAMPLE_RATE, 1)
        log.info("transcribing %.1fs", dur_s)
        try:
            text = await self._stt.transcribe(audio_bytes, SAMPLE_RATE)
        except Exception:
            log.exception("STT failed")
            return
        finally:
            self._stt_busy = False
        if not text:
            log.debug("STT returned empty (%.1fs audio) — VAD may have caught noise", dur_s)
            return
        log.info("transcript: %r", text)
        _trace_log.info("USER  %s", text)
        lower = text.lower().strip().rstrip(string.punctuation).strip()
        if _is_filler(lower):
            log.debug("filler — skipping")
            return
        await self.push_frame(
            TranscriptionFrame(text=text, user_id="", timestamp=str(_now_us())),
            FrameDirection.DOWNSTREAM,
        )


def _is_filler(lower: str) -> bool:
    if not lower:
        return True
    tokens = lower.split()
    # Single-word utterance: only skip if it's a known filler, not a command.
    if len(tokens) == 1:
        return lower.rstrip(string.punctuation) in _FILLER_PHRASES
    if lower in _FILLER_PHRASES:
        return True
    return all(t.rstrip(string.punctuation) in _FILLER_PHRASES for t in tokens)


# ── RenderSceneProcessor ──────────────────────────────────────────────────────

class RenderSceneProcessor(FrameProcessor):
    """
    Multi-step agentic loop over render-mcp and oxr-mcp tools.

    Uses Llama-Nemotron (port 8106) with OpenAI tool calling + LMFE for the
    reasoning loop — guaranteed syntactically valid tool calls every iteration.
    Uses Minitron (port 8101) for the parallel quick-ack (fast, cheap).

    On each utterance:
      1. Quick-ack fires immediately (parallel, max 25 tokens) → agent.progress
      2. Agentic loop: model calls tools via OpenAI tool_calls protocol until
         it returns a text response (finish_reason != "tool_calls")
      3. Progress messages sent before each tool execution → agent.progress
      4. Final response → agent.response + TextFrame downstream for TTS
    """

    def __init__(
        self,
        transport:   XRMediaHubTransport,
        cfg:         WorkerConfig,
        render:      McpClient,
        oxr:         McpClient,
        vlm:         McpClient,
        video:       McpClient,
        prompt_path: Path,
        tools_openai: list,   # OpenAI tool definitions built from MCP discovery
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._transport      = transport
        self._cfg            = cfg
        self._render         = render
        self._oxr            = oxr
        self._vlm            = vlm
        self._video          = video
        self._prompt_path      = prompt_path
        self._prompt_cache     = prompt_path.read_text(encoding="utf-8").strip()
        _prompts               = prompt_path.parent
        self._quick_ack_path   = _prompts / "quick_ack.txt"
        self._still_work_path  = _prompts / "still_working.txt"
        self._validate_path    = _prompts / "validate.txt"
        self._quick_ack_cache  = self._quick_ack_path.read_text(encoding="utf-8").strip()
        self._still_work_cache = self._still_work_path.read_text(encoding="utf-8").strip()
        self._validate_cache   = self._validate_path.read_text(encoding="utf-8").strip()
        self._tools_openai     = tools_openai
        self._http           = httpx.AsyncClient(timeout=180.0)

        self._pending:    tuple[str, str, int] | None = None  # (text, pid, ref_us)
        self._lock        = asyncio.Lock()
        self._drain_task:   asyncio.Task | None = None
        self._agentic_task: asyncio.Task | None = None
        # Rolling conversation buffer — last N turns of (user_text, agent_response).
        # Injected as context so the agent understands "fix that", "undo", "the one I just added".
        self._history:    list[tuple[str, str]] = []
        self._history_max = 4


    def _read_prompt(self, path: Path, cache_attr: str) -> str:
        try:
            text = path.read_text(encoding="utf-8").strip()
            setattr(self, cache_attr, text)
            return text
        except OSError:
            log.warning("prompt file unreadable: %s — using cache", path.name)
            return getattr(self, cache_attr)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            await self._enqueue(frame)
        else:
            await self.push_frame(frame, direction)

    async def _enqueue(self, frame: TranscriptionFrame) -> None:
        text = frame.text.strip()
        if not text:
            return
        pid = self._transport.target_participant
        # Capture the moment the user finished speaking so visual tool calls
        # can be anchored to that timestamp (not to when the tool fires).
        try:
            ref_us = int(frame.timestamp) if frame.timestamp else _now_us()
        except (TypeError, ValueError):
            ref_us = _now_us()
        async with self._lock:
            self._pending = (text, pid, ref_us)
            # Cancel any in-flight agentic loop so the new utterance is handled
            # immediately rather than waiting for the old one to finish.
            if self._agentic_task and not self._agentic_task.done():
                self._agentic_task.cancel()
            if self._drain_task is None or self._drain_task.done():
                self._drain_task = asyncio.create_task(
                    self._drain(), name="render-scene-drain",
                )

    async def _drain(self) -> None:
        while True:
            async with self._lock:
                if self._pending is None:
                    return
                text, pid, ref_us = self._pending
                self._pending = None

            # Quick-ack: fast Minitron call that (a) speaks an immediate
            # acknowledgment and (b) classifies whether Nemotron needs
            # reasoning enabled.  Await it first so the think flag is ready
            # before the main loop starts — it takes ~1s so the delay is small.
            try:
                ack, needs_thinking = await self._quick_ack(text)
            except Exception:
                log.exception("quick ack failed")
                ack, needs_thinking = "", False

            ack_pid = pid or self._transport.target_participant
            if ack and ack_pid:
                await self._send(ack_pid, ack, topic=_AGENT_PROGRESS_TOPIC)
                if needs_thinking:
                    await self.push_frame(TextFrame(text=ack), FrameDirection.DOWNSTREAM)

            # Start a "still working" timer — fires if reasoning takes >5s.
            # Cancelled as soon as the loop returns.
            # thinking_ctx is a one-element list shared with the agentic loop so
            # the still-working messages can reflect what the 30B is reasoning about.
            thinking_ctx: list[str] = [""]
            still_pid = pid or self._transport.target_participant
            still_task = asyncio.create_task(
                self._still_working_loop(text, still_pid, thinking_ctx,
                                         enabled=needs_thinking),
                name="still-working",
            )

            # Run the agentic loop as a tracked task so a new utterance can
            # cancel it mid-flight without tearing down the drain loop.
            self._agentic_task = asyncio.create_task(
                self._agentic_loop(
                    text, pid, ref_us=ref_us, needs_thinking=needs_thinking,
                    thinking_ctx=thinking_ctx,
                ),
                name="agentic-loop",
            )
            response = None
            try:
                response = await self._agentic_task
            except asyncio.CancelledError:
                log.info("agentic loop interrupted by new utterance")
                send_pid = pid or self._transport.target_participant
                if send_pid:
                    try:
                        await self._transport.endpoint.flush_return_audio(send_pid)
                    except Exception:
                        log.debug("flush_return_audio failed during cancellation", exc_info=True)
            except Exception:
                log.exception("agentic loop failed")
                response = "Something went wrong — please try again."
            finally:
                self._agentic_task = None
                still_task.cancel()
                try:
                    await still_task
                except asyncio.CancelledError:
                    pass  # expected — we just cancelled it above

            send_pid = pid or self._transport.target_participant

            # Record the turn in the rolling history buffer.
            if response:
                self._history.append((text, response))
                if len(self._history) > self._history_max:
                    self._history.pop(0)

            if response and send_pid:
                await self._send(send_pid, response, topic=_AGENT_RESPONSE_TOPIC)
                await self.push_frame(
                    TextFrame(text=response), FrameDirection.DOWNSTREAM,
                )

    # ── quick ack ─────────────────────────────────────────────────────────────

    async def _quick_ack(self, transcript: str) -> tuple[str, bool]:
        """Fast call: returns (ack_text, needs_thinking).

        Passes the last conversation turn as context so corrections like
        "try it again" or "that was wrong" produce sensible acks.
        """
        # Include the most recent agent action as context.
        context = ""
        if self._history:
            last_user, last_agent = self._history[-1]
            context = f"[Previous turn] User: {last_user} / Agent: {last_agent}\n"

        body = {
            "model": "llm",
            "messages": [
                {"role": "system", "content": self._read_prompt(
                    self._quick_ack_path, "_quick_ack_cache")},
                {"role": "user", "content": context + transcript},
            ],
            "max_tokens": 40,
            "temperature": 0.0,
        }
        try:
            resp = await asyncio.wait_for(
                self._http.post(
                    self._cfg.llm_server.rstrip("/") + "/v1/chat/completions",
                    json=body,
                ),
                timeout=8.0,
            )
            if not resp.is_error:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                obj_text = _extract_json(raw)
                if obj_text:
                    try:
                        obj = json.loads(obj_text)
                        ack   = str(obj.get("ack", "")).strip()
                        think = bool(obj.get("think", False))
                        log.info("quick-ack: %r  think=%s", ack, think)
                        _trace_log.info("ACK   %s  [think=%s]", ack, think)
                        return ack, think
                    except json.JSONDecodeError:
                        pass
                # Fallback: treat raw text as ack, no thinking
                return raw, False
        except Exception as exc:
            log.warning("quick-ack failed: %s", exc)
        return "", False

    # ── agentic loop (OpenAI tool calling + LMFE) ────────────────────────────

    async def _still_working_msg(self, transcript: str, sent: list[str],
                                  thinking_ctx: list[str]) -> str:
        """Ask Minitron for a short contextual 'still working' sentence.

        `sent` is the list of messages already shown this turn.
        `thinking_ctx` is a one-element list holding the latest reasoning_content
        from the 30B model — used to make progress updates reflect what the model
        is actually working on rather than guessing from the transcript alone.
        """
        avoid = ""
        if sent:
            avoid = (
                " Do NOT repeat or paraphrase any of these already-sent messages: "
                + ", ".join(f'"{m}"' for m in sent[-3:])
                + "."
            )
        thinking = thinking_ctx[0].strip() if thinking_ctx[0] else ""
        # Truncate — we only need the last few lines of thinking to get the gist.
        if thinking:
            lines = [l.strip() for l in thinking.splitlines() if l.strip()]
            thinking = " ".join(lines[-6:])[-400:]

        user_content = f"User request: {transcript}"
        if thinking:
            user_content += f"\n\nWhat the AI is currently reasoning through:\n{thinking}"

        base = self._read_prompt(self._still_work_path, "_still_work_cache")
        body = {
            "model": "llm",
            "messages": [
                {"role": "system", "content": base + avoid},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 24,
            "temperature": 0.9,
        }
        try:
            resp = await asyncio.wait_for(
                self._http.post(
                    self._cfg.llm_server.rstrip("/") + "/v1/chat/completions",
                    json=body,
                ),
                timeout=6.0,
            )
            if not resp.is_error:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            log.debug("still-working message failed: %s", exc)
        return ""

    async def _still_working_loop(
        self, transcript: str, pid: str, thinking_ctx: list[str],
        *, first_after: float = 5.0, repeat_every: float = 10.0,
        enabled: bool = True,
    ) -> None:
        """Speak periodic contextual updates while the agentic loop runs.

        Only fires when enabled=True (i.e. needs_thinking).  Non-thinking turns
        resolve in under a second — a "still working" message would just be noise.
        """
        if not enabled:
            return
        sent: list[str] = []
        await asyncio.sleep(first_after)
        while True:
            msg = await self._still_working_msg(transcript, sent, thinking_ctx)
            if msg and pid:
                # Data channel only — spoken updates stack up in the TTS queue
                # and play after the real response, confusing the user.
                await self._send(pid, msg, topic=_AGENT_PROGRESS_TOPIC)
                sent.append(msg)
            await asyncio.sleep(repeat_every)

    async def _validate(self, transcript: str, post_scene: dict) -> tuple[bool, str]:
        """Ask Minitron whether the task was completed as requested.

        Returns (ok, issue). Defaults to ok=True on any failure so a broken
        validator never blocks the response.
        """
        body = {
            "model": "llm",
            "messages": [
                {"role": "system", "content": self._read_prompt(
                    self._validate_path, "_validate_cache")},
                {"role": "user", "content": (
                    f"Request: {transcript}\n"
                    f"Current scene: {json.dumps(post_scene or {})}"
                )},
            ],
            "max_tokens": 60,
            "temperature": 0.0,
        }
        try:
            resp = await asyncio.wait_for(
                self._http.post(
                    self._cfg.llm_server.rstrip("/") + "/v1/chat/completions",
                    json=body,
                ),
                timeout=8.0,
            )
            if not resp.is_error:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                obj_text = _extract_json(raw)
                if obj_text:
                    obj = json.loads(obj_text)
                    ok    = bool(obj.get("ok", True))
                    issue = str(obj.get("issue", ""))
                    log.info("validation: ok=%s  issue=%r", ok, issue)
                    return ok, issue
        except Exception:
            log.debug("validation call failed — defaulting to ok", exc_info=True)
        return True, ""

    async def _agentic_loop(self, transcript: str, pid: str, *,
                            ref_us: int = 0,
                            needs_thinking: bool = False,
                            thinking_ctx: list[str] | None = None) -> str:
        """
        Multi-turn tool-calling loop using the OpenAI tool calling protocol.

        Scene state and head pose are pre-fetched concurrently before the
        loop starts and included in the user message, so the model skips
        those tool calls and goes straight to the operation.  Tools remain
        available for refresh queries or follow-up calls.
        """
        # Pre-fetch scene state, head pose, and the most common spatial position
        # (1.5 m ahead) concurrently.  Saves 1-3 tool-call iterations per turn.
        scene, pose, ahead = await asyncio.gather(
            self._call_mcp(self._render, "get_scene_state",  {}, silent=True),
            self._call_mcp(self._oxr,    "get_head_pose",    {}, silent=True),
            self._call_mcp(self._oxr,    "position_ahead",   {"distance": 1.5}, silent=True),
        )

        ctx_parts: list[str] = []

        # ── Scene ──────────────────────────────────────────────────────────────
        if isinstance(scene, dict) and scene.get("objects"):
            objs = scene["objects"]
            lines = ["SCENE OBJECTS:"]
            for o in objs:
                pos = o.get("position", {})
                col = o.get("color", {})
                lines.append(
                    f"  {o['id']} ({o['type']})  "
                    f"pos=({pos.get('x',0):.2f}, {pos.get('y',0):.2f}, {pos.get('z',0):.2f})  "
                    f"color=(r={col.get('r',0):.2f} g={col.get('g',0):.2f} b={col.get('b',0):.2f})  "
                    f"size={o.get('size',0.1):.3f}m"
                )
            ctx_parts.append("\n".join(lines))
        else:
            ctx_parts.append("SCENE OBJECTS: (empty)")

        # ── Head pose + derived spatial shortcuts ─────────────────────────────
        if isinstance(pose, dict) and pose.get("is_valid"):
            p  = pose["position"]
            fv = pose["forward"]
            rv = pose["right"]
            uv = pose.get("up", {"x": 0, "y": 1, "z": 0})

            # Compute common offsets directly — no extra tool calls needed.
            def _off(vec: dict, d: float) -> str:
                return (f"({p['x']+vec['x']*d:.2f}, "
                        f"{p['y']+vec['y']*d:.2f}, "
                        f"{p['z']+vec['z']*d:.2f})")

            ahead_str = (
                f"({ahead['x']:.2f}, {ahead['y']:.2f}, {ahead['z']:.2f})"
                if isinstance(ahead, dict) and "x" in ahead
                else _off(fv, 1.5)
            )

            ctx_parts.append(
                "HEAD POSE:\n"
                f"  position : ({p['x']:.2f}, {p['y']:.2f}, {p['z']:.2f})\n"
                f"  forward  : ({fv['x']:.3f}, {fv['y']:.3f}, {fv['z']:.3f})  ← 'ahead/forward'\n"
                f"  right    : ({rv['x']:.3f}, {rv['y']:.3f}, {rv['z']:.3f})  ← 'right'\n"
                f"  up       : ({uv['x']:.3f}, {uv['y']:.3f}, {uv['z']:.3f})  ← 'up'\n"
                f"  yaw={pose.get('yaw_deg',0):.1f}°  pitch={pose.get('pitch_deg',0):.1f}°\n"
                "SPATIAL SHORTCUTS (pre-computed — use directly, no tool call needed):\n"
                f"  1.5m ahead of you     : {ahead_str}\n"
                f"  1m to your right      : {_off(rv,  1.0)}\n"
                f"  1m to your left       : {_off(rv, -1.0)}\n"
                f"  0.5m above eye level  : {_off(uv,  0.5)}\n"
                f"  1m behind you         : {_off(fv, -1.0)}\n"
                "  For other distances: new_pos = obj.pos + direction_vec × distance (per component)"
            )
        else:
            ctx_parts.append("HEAD POSE: unavailable")

        if pid:
            ctx_parts.append(f"Participant: {pid}")
        if ref_us:
            ctx_parts.append(f"Reference time (when user spoke): {ref_us} µs")

        # Recent conversation history — lets the agent understand "fix that",
        # "undo", "the sphere I just added", etc.
        if self._history:
            hist_lines = []
            for u, a in self._history:
                hist_lines.append(f"  User: {u}")
                hist_lines.append(f"  Agent: {a}")
            ctx_parts.append("[Recent conversation]\n" + "\n".join(hist_lines))

        context = "\n".join(ctx_parts)
        log.info("pre-fetched context for turn")
        _trace_log.info("CTX   %s", context.replace("\n", " | "))

        try:
            system_content = self._prompt_path.read_text(encoding="utf-8").strip()
            self._prompt_cache = system_content
        except OSError:
            log.warning("prompt file unreadable — using cached version")
            system_content = self._prompt_cache
        if needs_thinking:
            system_content = (
                "Use your private <think> block to work through these steps. "
                "NEVER output these steps as your response — your only text output "
                "to the user is ONE SHORT sentence AFTER all tool calls are done.\n"
                "\n"
                "THINK STEP 1 — RESOLVE: Which object? "
                "Pronouns ('it', 'that') = most recently added/modified object. "
                "Named ('the blue sphere') = match by color/type in scene.\n"
                "\n"
                "THINK STEP 2 — LOCATE: Copy the exact x, y, z of the target object "
                "and the head pose right/forward/up vectors from the context.\n"
                "\n"
                "THINK STEP 3 — COMPUTE: Calculate new coordinates with explicit arithmetic. "
                "User-relative move: new = old + head_vec × distance (per component). "
                "Near object: new = obj.pos ± world_offset. "
                "Midpoint: new = (A + B) / 2 per component. "
                "Write out each component: x=…, y=…, z=…\n"
                "\n"
                "THINK STEP 4 — EXECUTE: call the tool with the computed values, "
                "then reply with ONE short sentence to the user.\n\n"
                + system_content
            )

        messages: list[dict] = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": (
                    f"[Pre-fetched context — do not call get_scene_state or "
                    f"get_head_pose unless you need to refresh after changes]\n"
                    f"{context}\n\n"
                    f"[Request]\n{transcript}"
                ),
            },
        ]

        for iteration in range(_MAX_LOOP):
            body = {
                "model": "llm",
                "messages": messages,
                "tools": self._tools_openai,
                # thinking off: 1024 covers any tool-call JSON.
                # thinking on: budget=1024 gives room for step-by-step arithmetic;
                # 2048 total leaves 1024 for the tool-call response.
                "max_tokens": 2048 if needs_thinking else 1024,
                "temperature": 0.0,
                "chat_template_kwargs": {
                    "enable_thinking": needs_thinking,
                    **({"thinking_budget": 1024} if needs_thinking else {}),
                },
            }
            try:
                resp = await self._http.post(
                    self._cfg.agent_llm_server.rstrip("/") + "/v1/chat/completions",
                    json=body,
                )
                if resp.is_error:
                    log.error("agent-llm %s: %s", resp.status_code, resp.text[:300])
                    break
            except Exception:
                log.exception("agent-llm call failed on iteration %d", iteration)
                break

            choice  = resp.json()["choices"][0]
            message = choice["message"]
            finish  = choice.get("finish_reason", "")

            # Capture the 30B's reasoning content and share it with the
            # still-working loop so progress updates reflect actual reasoning steps.
            reasoning = (message.get("reasoning_content") or "").strip()
            if reasoning and thinking_ctx is not None:
                thinking_ctx[0] = reasoning

            tool_calls = message.get("tool_calls") or []
            content    = (message.get("content") or "").strip()

            log.info("agent-llm iter=%d  finish=%s  tool_calls=%d  content=%r",
                     iteration, finish, len(tool_calls), content[:200])

            if not tool_calls:
                # Thinking filled the token budget without emitting a tool call.
                # Turn off thinking and retry the same iteration so messages is
                # unchanged and the model gets another chance.
                if finish == "length" and needs_thinking:
                    log.warning(
                        "agent-llm iter=%d hit length limit during thinking — "
                        "retrying without thinking", iteration,
                    )
                    needs_thinking = False
                    continue

                # Recover from off-script tool call output.
                # The model sometimes emits tool calls as plain-text JSON instead
                # of using the tool_calls field.  Two shapes seen in practice:
                #   (a) bare name:  "get_head_pose"
                #   (b) JSON obj:   {"name": "update_primitive", "arguments": {...}}
                import uuid as _uuid
                all_names = {t["function"]["name"] for t in self._tools_openai}
                recovered: dict | None = None

                if content in all_names:
                    # Shape (a): bare tool name, no args.
                    recovered = {"name": content, "arguments": {}}
                else:
                    obj_text = _extract_json(content)
                    if obj_text:
                        try:
                            obj = json.loads(obj_text)
                            name = obj.get("name") or obj.get("tool") or obj.get("function")
                            if isinstance(name, str) and name in all_names:
                                args = obj.get("arguments") or obj.get("args") or {}
                                recovered = {"name": name, "arguments": args if isinstance(args, dict) else {}}
                        except json.JSONDecodeError:
                            # Best-effort recovery: the plain-text fragment is not valid JSON —
                            # leave recovered=None and fall through to the no-tool-call path.
                            log.debug("failed to decode recovered tool-call JSON: %r", obj_text)

                if recovered:
                    log.warning("text-format tool call %r — recovering", recovered["name"])
                    tool_calls = [{
                        "id":       f"call_{_uuid.uuid4().hex[:12]}",
                        "type":     "function",
                        "function": {
                            "name":      recovered["name"],
                            "arguments": json.dumps(recovered["arguments"]),
                        },
                    }]
                else:
                    # Genuine final response.
                    _trace_log.info("RESP  %s", content or "Done.")
                    return content or "Done."

            # Add the assistant's tool-call message to the conversation.
            messages.append({
                "role":       "assistant",
                "content":    content or None,
                "tool_calls": tool_calls,
            })

            # Execute each tool call and append results.
            for tc in tool_calls:
                name     = tc["function"]["name"]
                args_str = tc["function"].get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}

                progress = _TOOL_PROGRESS.get(name)
                if progress and pid:
                    # Data channel only — TTS is too spammy for per-tool updates.
                    await self._send(pid, progress, topic=_AGENT_PROGRESS_TOPIC)

                log.info("tool call  iter=%d  tool=%s  args=%s", iteration, name, args)
                _trace_log.info("TOOL  [%d] %s(%s)", iteration, name,
                                ", ".join(f"{k}={v}" for k, v in args.items()))
                try:
                    result = await self._execute_tool(name, args)
                except _SceneNotReadyError:
                    return (
                        "The XR scene isn't ready yet. "
                        "Please click 'Launch XR' to start the headset session first."
                    )
                result_str = json.dumps(result, default=str)
                log.info("tool result  tool=%s  %s", name, result_str[:200])
                _trace_log.info("RES   [%d] %s → %s", iteration, name, result_str[:300])

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      result_str,
                })

        return "Done."

    # ── tool routing ──────────────────────────────────────────────────────────

    async def _execute_tool(self, tool: str, args: dict) -> dict | str | None:
        """Route a tool call to render-mcp or oxr-mcp."""
        # Normalize nested dicts that the LLM sometimes generates instead of
        # flat scalar args — e.g. {"position": {x,y,z}} → x=, y=, z=.
        args = _normalize_tool_args(args)

        if tool in _OXR_TOOLS:
            return await self._call_mcp(self._oxr, tool, args)

        # Intercept not_started before it reaches the model — it means LOVR
        # hasn't spawned yet and no render op will succeed.  Return a clear
        # explanation so the model can respond to the user instead of retrying.
        if tool in _VLM_TOOLS:
            # Guard against fabricated paths: the model must call
            # get_frame_from_time first and use the returned path.
            if tool == "ask_image":
                import os
                path = args.get("image_path", "")
                if path and not os.path.isfile(path):
                    return {
                        "error": (
                            f"File not found: {path!r}. "
                            "You must call get_frame_from_time first to get the "
                            "real image path, then pass that path to ask_image."
                        )
                    }
            return await self._call_mcp(self._vlm, tool, args)
        if tool in _VIDEO_TOOLS:
            return await self._call_mcp(self._video, tool, args)
        result = await self._call_mcp(self._render, tool, args)
        if isinstance(result, dict) and result.get("reason") == "not_started":
            raise _SceneNotReadyError()
        return result

    async def _call_mcp(
        self, client: McpClient, tool: str, args: dict, *, silent: bool = False
    ) -> dict | str | None:
        try:
            res  = await client.call_tool(tool, args)
            data = _tool_payload(res)
            return data
        except Exception as exc:
            if not silent:
                log.error("mcp %s failed: %s", tool, exc)
            return {"error": str(exc)}

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _send(self, pid: str, text: str, *, topic: str) -> None:
        try:
            await self._transport.send_return_data(DataMessage(
                participant_id=pid,
                topic=topic,
                pts_us=_now_us(),
                data=text.encode(),
            ))
        except Exception:
            log.exception("send failed  topic=%s", topic)

    async def close(self) -> None:
        await self._http.aclose()


# ── TtsProcessor ─────────────────────────────────────────────────────────────

class TtsProcessor(FrameProcessor):
    """TextFrame → sentence-batched TTS → hub return audio."""

    def __init__(
        self,
        tts: TtsClient,
        transport: XRMediaHubTransport,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tts       = tts
        self._transport = transport

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            return
        text = frame.text.strip()
        if not text:
            await self.push_frame(frame, direction)
            return
        pid = self._transport.target_participant
        if not pid:
            await self.push_frame(frame, direction)
            return
        try:
            await stream_sentences_to_audio(
                self._transport.endpoint, self._tts.synthesize, text, pid,
            )
        except Exception:
            log.exception("TTS failed  pid=%r", pid)
        await self.push_frame(frame, direction)


# ── pipeline factory ──────────────────────────────────────────────────────────

def build_pipeline(
    transport:   XRMediaHubTransport,
    stt:         SttClient,
    tts:         TtsClient,
    scene:       RenderSceneProcessor,
) -> tuple[Pipeline, PipelineTask]:
    stt_proc = SttProcessor(stt, transport, scene._cfg)
    tts_proc = TtsProcessor(tts, transport)

    pipeline = Pipeline([
        transport.input(),
        stt_proc,
        scene,
        tts_proc,
        transport.output(),
    ])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
        idle_timeout_secs=None,
    )
    return pipeline, task


# ── helpers ───────────────────────────────────────────────────────────────────


def _tool_payload(result) -> dict | list | None:
    if hasattr(result, "data") and result.data is not None:
        return result.data
    return getattr(result, "structured_content", None)


def _extract_json(text: str) -> str | None:
    depth, start, in_string, escape = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escape:        escape = False
            elif ch == "\\": escape = True
            elif ch == '"':  in_string = False
            continue
        if ch == '"':   in_string = True; continue
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            if depth == 0: continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    return None


def _parse_agent_turn(text: str) -> dict:
    """Parse one LLM output turn.

    Expected shapes:
      {"tool": "<name>", "args": {...}}
      {"done": true, "response": "..."}

    If no JSON is found but the text looks like a prose response, returns
    {"done": true, "response": <first sentence>} as a graceful fallback so
    the agent doesn't silently give up.
    """
    obj_text = _extract_json(text)
    if obj_text:
        try:
            obj = json.loads(obj_text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            # Coerce numeric argument values to float.
            args = obj.get("args") or obj.get("arguments") or {}
            if isinstance(args, dict):
                obj["args"] = {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in args.items()
                }
            return obj

    # No JSON — model went off-script.  If it looks like a completion
    # statement, extract the first sentence as a response.
    stripped = text.strip()
    if stripped:
        first_sentence = stripped.split(".")[0].strip()
        if first_sentence:
            return {"done": True, "response": first_sentence + "."}
    return {}


def _normalize_tool_args(args: dict) -> dict:
    """Flatten nested position/color dicts that the LLM sometimes generates.

    The LLM may produce {"position": {"x":0,"y":1.6,"z":-1.5}} because it
    pattern-matches the get_scene_state output format. Flatten to scalar kwargs
    so FastMCP validation passes.
    """
    args = dict(args)

    if "position" in args and isinstance(args["position"], dict):
        pos = args.pop("position")
        for k in ("x", "y", "z"):
            if k in pos and k not in args:
                args[k] = float(pos[k])

    if "color" in args and isinstance(args["color"], dict):
        col = args.pop("color")
        for k in ("r", "g", "b"):
            if k in col and k not in args:
                args[k] = float(col[k])

    # Strip None and empty-string values — the model sometimes emits r=''
    # when thinking is enabled and the value wasn't filled in.
    return {k: v for k, v in args.items() if v is not None and v != ""}


def _tool_param_sig(input_schema: dict) -> str:
    props = (input_schema or {}).get("properties", {})
    if not props:
        return ""
    return ", ".join(
        f"{k}: {v.get('type', 'any')}" for k, v in props.items()
    )
