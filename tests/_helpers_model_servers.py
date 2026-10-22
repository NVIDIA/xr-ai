# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize the real sample configs for service configuration tests."""
from pathlib import Path
from tempfile import TemporaryDirectory

from xr_ai_model_server_stack._stack import _configs, _processes

_ROOT = Path(__file__).resolve().parents[1]
_DIRECTORY = TemporaryDirectory(prefix="test-model-configs-")


def profile_config(basename: str, profile: str) -> Path:
    if basename == "stt_server":
        # Batch STT remains an explicit alternative, not a default sample service.
        return _ROOT / "services/stt-server/stt_server.yaml"
    sample = "tea-making-sample" if basename == "embedding_server" else "lab-instrument-monitoring"
    services = ("speaker-stt", "tts", "omni", "embedding") if sample == "tea-making-sample" else (
        "speaker-stt", "tts", "omni", "vlm",
    )
    base = _ROOT / "agent-samples" / sample / "model-servers"
    directory = Path(_DIRECTORY.name) / sample / profile
    directory.mkdir(parents=True, exist_ok=True)
    _processes(_configs(base, services, profile), directory, _ROOT)
    return directory / f"{basename}.yaml"
