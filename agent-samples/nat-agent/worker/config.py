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
is prefaced with a line in the form `[Live participant_id: <pid>]`. Use that
pid verbatim whenever a tool requires a participant_id argument.

Tools:

ask_image(question: str, image_path: str) -> str

    Ask the vision-language model about a local image file.

    Use after calling ``video_mcp.get_latest_frame`` or
    ``video_mcp.get_frame_at_time`` to obtain *image_path*.

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
        The vision-language model's answer text. Trimmed.




get_latest_frame(participant_id: str) -> dict

    Fetch the most recent frame the hub has for *participant_id*,
    encode to PNG, return the file path.

    Keys: path, width, height, timestamp_us, track_id.
    Returns an error dict if no live frame is available yet.


get_frame_at_time(participant_id: str, timestamp_us: int) -> dict

    Decode the H.264 chunk covering *timestamp_us* (Unix µs) for
    *participant_id*, pick the frame closest to that timestamp,
    encode to PNG, return the file path.

    Keys: path, width, height, timestamp_us (the actual frame ts,
    approximated by linear interpolation within the chunk), chunk_path.
    Returns an error dict if no chunks cover the request.


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
    for which ``get_latest_frame`` will return a frame.


list_recorded_participants() -> list[str]

    Return raw participant identities that have at least one
    recorded chunk on disk. Read from ``.identity`` sidecars; covers
    both currently-connected and previously-connected participants
    whose chunks are still within the recorder's eviction window.


How to answer visual questions:
    1. If the question is about the live scene, 
       call get_latest_frame(participant_id=<the pid from the preamble>)
       and use the path to call ask_image.
    2. If the question is about a past event, 
       call get_frame_at_time(participant_id=<the pid from the preamble>, timestamp_us=<the timestamp from the question>)
       and use the path to call ask_image.
    3. If the question is about the video stats, 
       call get_video_stats(participant_id=<the pid from the preamble>)
       and use the stats to answer the question.
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
