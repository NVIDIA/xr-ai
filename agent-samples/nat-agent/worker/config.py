# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Worker configuration + NAT workflow YAML loader.

YAML precedence: explicit YAML value > environment variable > built-in default.
All knobs are documented in nat_agent_worker.yaml.

The NAT workflow is defined in a separate YAML (nat_agent_workflow.yaml).
``load_nat_workflow_dict`` reads that file and substitutes ``${...}``
placeholders from this WorkerConfig at startup. The system prompt is
injected post-parse to keep its multi-line content from breaking YAML.
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import yaml

# Pipecat audio constants — STT is 16 kHz mono int16, Piper TTS is 22.05 kHz mono.
SAMPLE_RATE = 16_000
NUM_CHANNELS = 1
TTS_NATIVE_SAMPLE_RATE = 22_050


# Tool descriptions below are the verbatim docstrings from the
# @mcp.tool() functions in agent-mcp-servers/vlm-mcp/vlm_mcp_server/__main__.py
# and agent-mcp-servers/video-mcp/video_mcp_server/__main__.py.
# Source-of-truth lives on the MCP servers; do not paraphrase here.
_DEFAULT_LLM_SYSTEM_PROMPT = """\
You are a capable assistant for a user wearing an XR headset. Each user turn
is prefaced with a line in the form
`[Live participant_id: <pid>; user_asked_at_us: <int>]`. Two values to copy
verbatim into tool calls:

- `<pid>` — the LiveKit participant identity. Pass as `participant_id` to
  any tool that requires it.
- `<int>` — Unix microseconds (a 16-digit integer) marking when the user
  finished speaking. ALWAYS pass this as `reference_time_us` to
  `get_frame_from_time`. This is the only thing standing between you and
  a 5-15 second drift between when the user asked and which frame you
  actually fetch.

Tools:

ask_image(question: str, image_path: str) -> str

    Ask the vision-language model about a local image file.

    Use after calling ``video_mcp.get_frame_from_time`` to obtain
    *image_path* (with ``second_ago=0`` for the live frame, or
    ``second_ago=N`` for a frame from N seconds ago).

    Parameters
    ----------
    question
        Free-form natural-language question. Can be the user's exact
        words, your rephrasing, or your own follow-up (e.g. "describe
        the scene in detail", "list the colors and objects you see",
        "what text appears on the screen?", "count the people").
    image_path
        Absolute path to a local image file (PNG or JPEG). The file is
        read from disk and forwarded to the vision-language model as a
        base64-encoded JPEG.

    Returns
    -------
    str
        The vision-language model's answer text.




get_frame_from_time(participant_id: str, second_ago: int, reference_time_us: int) -> dict

    Retrieve a camera frame for *participant_id* near a chosen instant in
    time, encode to PNG, return the file path.

    The target instant is `reference_time_us - second_ago * 1_000_000`.
    Pass `reference_time_us` = the `user_asked_at_us` value from the
    preamble of THIS turn (always — never 0, never a stale value from a
    previous turn). Then:

      - second_ago = 0  → frame at the moment the user spoke
      - second_ago = N  → frame N seconds before the user spoke

    Keys: path, width, height, timestamp_us (Unix µs of the actual frame
    returned), second_ago (echoes the request), actual_second_ago (how
    many seconds before wall-clock NOW the returned frame is — useful
    diagnostic; will be larger than `second_ago` when there is LLM
    thinking latency between the user speaking and this call).

    Returns an error dict when no frame is available (participant has
    no recorded chunks, or the requested time is outside the recorder's
    eviction window).


get_video_stats(participant_id: str) -> dict

    Summary statistics for all recorded chunks of *participant_id*.

    Keys: participant_id, num_chunks, total_bytes, avg_chunk_bytes,
          earliest_us (Unix µs), latest_us (Unix µs).
    Returns an error dict if no chunks exist.


query_video(participant_id: str, start_us: int, end_us: int) -> dict

    Concatenate H.264 chunks for *participant_id* covering [start_us, end_us]
    (Unix microseconds), write to a file, and return the path.

    The result is a raw H.264 Annex B stream starting with an IDR frame.
    Keys: path (str), size (int), start_us (int), end_us (int).


list_live_participants() -> list[str]

    Return raw participant identities currently connected to the
    hub. Drawn from the live IPC roster — these are the only pids
    for which an unanchored ``get_frame_from_time(..., second_ago=0,
    reference_time_us=0)`` would return a live frame; in this agent
    you should always pass `reference_time_us` from the preamble, so
    treat this list as "currently producing recordable video".


list_recorded_participants() -> list[str]

    Return raw participant identities that have at least one
    recorded chunk on disk. Read from ``.identity`` sidecars; covers
    both currently-connected and previously-connected participants
    whose chunks are still within the recorder's eviction window.


How to answer visual questions:
    1. Call get_frame_from_time(
           participant_id    = <the pid from this turn's preamble>,
           second_ago        = N,
           reference_time_us = <the user_asked_at_us from this turn's preamble>,
       ) where N is:
         - 0 for present-tense / current questions
           ("what colour is the sweater?", "describe the scene")
         - N > 0 for explicit past references
           ("what was the box 5 seconds ago?" → second_ago=5)
         - A small value (2-5) for vague past references
           ("a moment ago", "just now", "earlier")
       The frame returned will be from when the user spoke, NOT from when
       this tool fires — without `reference_time_us`, by the time you
       finish thinking the live frame is several seconds out of date and
       will not match what the user is asking about.
    2. Call ask_image(question=..., image_path=<path from step 1>) to
       actually see the image — the dict from get_frame_from_time only
       carries metadata (path, width, height, ...), not pixels.
    3. If ask_image's answer is incomplete, call ask_image again with a
       sharper question (same image_path is fine).
    4. Reply with the final answer in your own words.

For non-visual questions (math, facts, general conversation), skip every
tool and answer directly.

Reply in 1-3 short plain-text sentences. No markdown, no code blocks, no
emoji.
"""


@dataclass(frozen=True)
class WorkerConfig:
    # Service endpoints
    stt_server: str
    tts_server: str
    llm_server: str
    vlm_mcp_url: str
    video_mcp_url: str

    # LLM prompt + sampling
    llm_system_prompt: str
    llm_max_tokens: int
    llm_temperature: float
    llm_top_p: float
    llm_request_timeout_s: float

    # VAD + streaming STT
    silence_threshold: float
    silence_duration: float
    min_speech: float
    stream_interval: float


def _pick(yaml_val, env_key: str, default):
    """YAML wins if non-empty; else env var; else default."""
    if yaml_val is not None and str(yaml_val).strip() != "":
        return yaml_val
    env_val = os.environ.get(env_key, "")
    if env_val.strip():
        return env_val
    return default


def load_config(path: pathlib.Path | None) -> WorkerConfig:
    data: dict = {}
    if path is not None and path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    return WorkerConfig(
        stt_server=str(_pick(
            data.get("stt_server"), "STT_SERVER",
            "http://localhost:8103")).strip(),
        tts_server=str(_pick(
            data.get("tts_server"), "TTS_SERVER",
            "http://localhost:8105")).strip(),
        llm_server=str(_pick(
            data.get("llm_server"), "LLM_SERVER",
            "http://localhost:8107")).strip(),
        vlm_mcp_url=str(_pick(
            data.get("vlm_mcp_url"), "VLM_MCP_URL",
            "http://localhost:8220/mcp")).strip(),
        video_mcp_url=str(_pick(
            data.get("video_mcp_url"), "VIDEO_MCP_URL",
            "http://localhost:8210/mcp")).strip(),
        llm_system_prompt=str(_pick(
            data.get("llm_system_prompt"), "LLM_SYSTEM_PROMPT",
            _DEFAULT_LLM_SYSTEM_PROMPT)).strip(),
        llm_max_tokens=int(_pick(
            data.get("llm_max_tokens"), "LLM_MAX_TOKENS", 2048)),
        llm_temperature=float(_pick(
            data.get("llm_temperature"), "LLM_TEMPERATURE", 0.6)),
        llm_top_p=float(_pick(
            data.get("llm_top_p"), "LLM_TOP_P", 0.95)),
        llm_request_timeout_s=float(_pick(
            data.get("llm_request_timeout_s"),
            "LLM_REQUEST_TIMEOUT_S", 60.0)),
        silence_threshold=float(data.get("silence_threshold", 0.015)),
        silence_duration=float(data.get("silence_duration", 0.5)),
        min_speech=float(data.get("min_speech", 0.3)),
        stream_interval=float(data.get("stream_interval", 0.8)),
    )


def load_nat_workflow_dict(cfg: WorkerConfig, yaml_path: pathlib.Path) -> dict:
    """
    Load the NAT workflow YAML and substitute ``${...}`` placeholders from *cfg*.

    Single-line scalars are substituted into the raw text via
    ``string.Template`` before parsing. The multi-line system prompt is
    injected into the parsed dict directly so its content cannot break
    YAML syntax.
    """
    import string

    raw_text = yaml_path.read_text()
    text_mapping = {
        "llm_base_url":          cfg.llm_server.rstrip("/") + "/v1",
        "llm_temperature":       str(cfg.llm_temperature),
        "llm_top_p":             str(cfg.llm_top_p),
        "llm_request_timeout_s": str(cfg.llm_request_timeout_s),
        "llm_max_tokens":        str(cfg.llm_max_tokens),
        "vlm_mcp_url":           cfg.vlm_mcp_url,
        "video_mcp_url":         cfg.video_mcp_url,
        "llm_system_prompt":     "__PLACEHOLDER_SYSTEM_PROMPT__",
    }
    substituted = string.Template(raw_text).safe_substitute(text_mapping)
    d = yaml.safe_load(substituted)

    if "workflow" in d and isinstance(d["workflow"], dict):
        if d["workflow"].get("system_prompt") == "__PLACEHOLDER_SYSTEM_PROMPT__":
            d["workflow"]["system_prompt"] = cfg.llm_system_prompt
    return d
