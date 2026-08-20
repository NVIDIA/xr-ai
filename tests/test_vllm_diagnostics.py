# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for actionable vLLM startup-failure classifications."""
from __future__ import annotations

from pathlib import Path

from xr_ai_vllm._diagnostics import classify_vllm_failure


def test_negative_kv_cache_is_classified_as_insufficient_gpu_memory(
    tmp_path: Path,
) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(
        "Available KV cache memory: -0.17 GiB\n"
        "ValueError: No available memory for the cache blocks.\n"
    )

    diagnosis = classify_vllm_failure(
        log, ["vllm", "--gpu-memory-utilization", "0.78"],
    )

    assert diagnosis is not None
    assert diagnosis.startswith("INSUFFICIENT GPU MEMORY")
    assert "-0.17 GiB" in diagnosis
    assert "0.78" in diagnosis


def test_cuda_oom_is_classified(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text("torch.OutOfMemoryError: CUDA out of memory")

    diagnosis = classify_vllm_failure(log, ["vllm", "serve", "model"])

    assert diagnosis is not None
    assert diagnosis.startswith("INSUFFICIENT GPU MEMORY")


def test_conflicting_process_memory_is_classified(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(
        "Free memory on device (10.0/45.0 GiB) at startup is less than desired "
        "GPU memory utilization (0.78, 35.1 GiB)."
    )

    diagnosis = classify_vllm_failure(
        log, ["vllm", "serve", "model", "--gpu-memory-utilization", "0.78"],
    )

    assert diagnosis is not None
    assert diagnosis.startswith("INSUFFICIENT FREE GPU MEMORY")
    assert "0.78" in diagnosis


def test_unrelated_failure_has_no_gpu_memory_diagnosis(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text("ValueError: unsupported architecture")

    assert classify_vllm_failure(log, ["vllm", "serve", "model"]) is None
