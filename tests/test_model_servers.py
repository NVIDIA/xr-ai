# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-specific startup ordering for the shared model stack."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

_MAIN = Path(__file__).parents[1] / "agent-samples" / "model-servers" / "main.py"


def _module():
    spec = importlib.util.spec_from_file_location("model_servers_main", _MAIN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dual_ada_starts_smaller_models_before_shared_services() -> None:
    module = _module()

    with patch.object(module, "detect_gpu_config", return_value="dual_48G_ada"):
        processes = module._build_processes()

    assert [process.name for process in processes] == [
        "vlm",
        "llm",
        "stt",
        "agent-llm",
    ]


def test_other_profiles_keep_large_model_first() -> None:
    module = _module()

    with patch.object(module, "detect_gpu_config", return_value="96G_blackwell"):
        processes = module._build_processes()

    assert [process.name for process in processes] == [
        "stt",
        "agent-llm",
        "vlm",
        "llm",
    ]
