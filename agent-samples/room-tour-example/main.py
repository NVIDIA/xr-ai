# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
room-tour-example orchestrator — voice-driven semantic room tour.

The wearer turns on the camera, says "start room tour", and walks through the
space naming each room ("this is the living room", "this is the kitchen"). The
agent harvests the objects it sees per room into an in-memory semantic map.
Afterwards it answers spatial questions:

    "where am I"          → "You're in the kitchen."
    "where is the sofa"   → "The sofa is in the living room, to your left."

Pipeline
--------
Audio in (mic) → STT → text → RoomTourBrain → TTS reply
Live video frames → VLM (object/room perception, on demand)

Reuses the shared services: STT (NeMo Parakeet), VAD (Silero via xr-ai-vad),
TTS (Piper), VLM (Cosmos) — all via xr-ai-models / xr-ai-pipecat, the same way
simple-vlm-example does. No SLAM/pose backend is required; direction is
re-acquired from the live frame at query time.

How to run (from agent-samples/room-tour-example/):
    uv sync && uv run room_tour_example
"""
from pathlib import Path

from xr_ai_launcher import Process, run_stack, warn_if_missing
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_PROCESSES: list[Process] = [
    Process("hub", "../../server-runtime", "xr_media_hub",
            config="yaml/xr_media_hub.yaml"),
    Process("vlm", "../../ai-services/vlm-server", "vlm_server",
            config="yaml/vlm_server.yaml"),
    Process("stt", "../../ai-services/stt-server", "stt_server",
            config="yaml/stt_server.yaml"),
    Process("tts", "../../ai-services/tts/piper", "piper_tts_server",
            config="yaml/piper_tts_server.yaml"),
    Process("worker", "worker", "room_tour_example_worker",
            config="yaml/room_tour_example_worker.yaml"),
]


def run() -> None:
    setup_logging("orchestrator", namespace="room-tour-example")
    # HF_TOKEN only raises HF rate limits / is needed for gated models — warn,
    # don't block (the default STT/VLM/TTS models are public). See
    # docs/credentials.md.
    warn_if_missing("HF_TOKEN")
    run_stack(_PROCESSES, _BASE)


if __name__ == "__main__":
    run()
