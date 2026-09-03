# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for actionable vLLM startup-failure classifications."""
from __future__ import annotations

from pathlib import Path

from xr_ai_vllm._diagnostics import (
    classify_vllm_failure,
    is_cuda_memory_allocation_failure,
)

_EARLY_CUDA_ALLOCATION_FAILURE = (
    'File "/opt/vllm/vllm/v1/worker/gpu_worker.py", line 282, in init_device\n'
    "  self.init_snapshot = MemorySnapshot(device=self.device)\n"
    'File "/opt/vllm/vllm/utils/mem_utils.py", line 108, in measure\n'
    "  self.free_memory, self.total_memory = current_platform.mem_get_info(device)\n"
    "torch.AcceleratorError: CUDA error: out of memory\n"
    "Search for `cudaErrorMemoryAllocation' in the CUDA runtime API\n"
)


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


def test_spark_driver_allocation_failure_has_uma_diagnosis(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(_EARLY_CUDA_ALLOCATION_FAILURE)

    assert is_cuda_memory_allocation_failure(log)
    diagnosis = classify_vllm_failure(
        log,
        ["vllm", "serve", "model"],
        spark_uma=True,
    )

    assert diagnosis is not None
    assert diagnosis.startswith("DGX SPARK UMA ALLOCATION FAILURE")
    assert "retry" not in diagnosis.lower()

    exhausted = classify_vllm_failure(
        log,
        ["vllm", "serve", "model"],
        spark_uma=True,
        retry_exhausted=True,
    )
    assert exhausted is not None
    assert "retry was exhausted" in exhausted.lower()


def test_late_driver_allocation_failure_is_not_retryable(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(
        "model_executor.load_model()\n"
        "torch.AcceleratorError: CUDA error: out of memory\n"
        "cudaErrorMemoryAllocation\n"
    )

    assert not is_cuda_memory_allocation_failure(log)
    diagnosis = classify_vllm_failure(
        log,
        ["vllm", "serve", "model"],
        spark_uma=True,
    )
    assert diagnosis is not None
    assert diagnosis.startswith("INSUFFICIENT GPU MEMORY")


def test_torch_cuda_oom_is_not_driver_allocation_signature(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text("torch.OutOfMemoryError: CUDA out of memory")

    assert not is_cuda_memory_allocation_failure(log)


def test_attempt_boundary_excludes_an_old_driver_allocation_failure(
    tmp_path: Path,
) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(_EARLY_CUDA_ALLOCATION_FAILURE)
    current_attempt = log.stat().st_size
    with log.open("a", encoding="utf-8") as stream:
        stream.write("HfHubHTTPError: Invalid credentials in Authorization header\n")

    assert not is_cuda_memory_allocation_failure(
        log,
        start_offset=current_attempt,
    )
    assert classify_vllm_failure(
        log,
        ["vllm", "serve", "model"],
        spark_uma=True,
        start_offset=current_attempt,
    ) is None


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
