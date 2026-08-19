# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Absolute VRAM reservation, preflight, and measurement-profile tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from xr_ai_launcher import (
    GPUDevice,
    GPUProcess,
    VRAMProfileError,
    derive_gpu_memory_utilization,
    format_vram_preflight,
    load_deployment_profile,
    load_vram_profile,
    preflight_vram,
    require_vram_preflight,
    utilization_overrides,
    validate_vram_certification,
)
from xr_ai_launcher.vram_measure import main as measure_main
from xr_ai_vllm._config import gpu_memory_utilization
from xr_ai_vllm._diagnostics import classify_vllm_failure

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _gpu(index: int, *, total: float = 45.0, free: float = 44.0) -> GPUDevice:
    return GPUDevice(
        index=index,
        uuid=f"GPU-{index}",
        pci_bus_id=f"0000:{index:02x}:00.0",
        name="NVIDIA L40S",
        compute_capability=8.9,
        total_memory_gib=total,
        free_memory_gib=free,
        used_memory_gib=total - free,
        processes=(GPUProcess(f"GPU-{index}", 123, "python", total - free),),
    )


def _profile(tmp_path: Path, *, reservation: float = 37.0) -> Path:
    path = tmp_path / "vram.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "hardware_profile": "dual_48G_ada",
        "stack": "test",
        "status": "certified",
        "device_safety_reserve_gib": 2.0,
        "services": {
            "omni": {"gpu": 1, "reservation_gib": reservation, "vllm": True},
            "stt": {"gpu": 1, "reservation_gib": 3.0, "vllm": False},
        },
    }))
    return path


def test_preflight_accounts_for_complete_per_gpu_stack(tmp_path: Path) -> None:
    profile = load_vram_profile(_profile(tmp_path))
    results = preflight_vram(profile, (_gpu(0), _gpu(1)))

    assert results[0].incremental_required_gib == 40.0
    assert results[0].remaining_gib == 2.0
    require_vram_preflight(results)
    rendered = format_vram_preflight(profile, results)
    assert "omni" in rendered
    assert "Existing PID 123" in rendered


def test_preflight_reports_shortfall_and_active_service_is_not_double_counted(
    tmp_path: Path,
) -> None:
    profile = load_vram_profile(_profile(tmp_path, reservation=42.0))
    failed = preflight_vram(profile, (_gpu(0), _gpu(1)))
    with pytest.raises(VRAMProfileError, match=r"shortfall 3\.0 GiB"):
        require_vram_preflight(failed)

    active = preflight_vram(
        profile, (_gpu(0), _gpu(1)), active_services=frozenset({"omni"}),
    )
    require_vram_preflight(active)


def test_vllm_utilization_is_derived_upward_from_absolute_gib(tmp_path: Path) -> None:
    profile = load_vram_profile(_profile(tmp_path))
    inventory = (_gpu(0), _gpu(1, total=45.7))

    assert derive_gpu_memory_utilization(37.0, 45.7) == 0.8097
    assert utilization_overrides(profile, inventory) == {"omni": "0.8097"}


def test_service_prefers_launcher_derived_utilization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XR_AI_GPU_MEMORY_UTILIZATION", "0.8123")
    assert gpu_memory_utilization({"gpu_memory_utilization": 0.5}, 0.85) == 0.8123


def test_certification_rejects_changed_service_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "omni.yaml"
    config.write_text("max_model_len: 32768\n")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    profile_path = tmp_path / "vram.json"
    profile_path.write_text(json.dumps({
        "schema_version": 1,
        "hardware_profile": "dual_48G_ada",
        "stack": "test",
        "status": "certified",
        "device_safety_reserve_gib": 2,
        "services": {"omni": {
            "gpu": 1,
            "reservation_gib": 40,
            "vllm": True,
            "measurement_signature": {
                "git_commit": "abc123",
                "driver_version": "580.0",
                "config_sha256": {"measured.yaml": digest},
            },
        }},
    }))
    profile = load_vram_profile(profile_path)
    monkeypatch.setattr(
        "xr_ai_launcher._vram.subprocess.check_output",
        lambda _command, **_kwargs: "580.0",
    )

    validate_vram_certification(profile, {"omni": config})
    config.write_text("max_model_len: 65536\n")
    with pytest.raises(VRAMProfileError, match="config changed"):
        validate_vram_certification(profile, {"omni": config})


def test_vllm_kv_cache_failure_is_classified_as_insufficient_vram(
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
    assert diagnosis.startswith("INSUFFICIENT VRAM")
    assert "-0.17 GiB" in diagnosis
    assert "0.78" in diagnosis


def test_certify_builds_profile_from_measurement(tmp_path: Path) -> None:
    measurement = tmp_path / "omni.measurement.json"
    measurement.write_text(json.dumps({
        "schema_version": 1,
        "kind": "xr-ai-vram-measurement",
        "measurement_signature": {"runtime": "test"},
        "summary": {"1": {"recommended_reservation_gib": 40.5}},
    }))
    output = tmp_path / "vram.default.json"

    assert measure_main([
        "certify",
        "--hardware-profile", "dual_48G_ada",
        "--stack", "default",
        "--output", str(output),
        "--minimum-runs", "1",
        "--service", f"omni:1:vllm:{measurement}",
    ]) == 0

    profile = load_vram_profile(output)
    assert profile.status == "certified"
    assert profile.services[0].reservation_gib == 40.5


@pytest.mark.parametrize("hardware", ["dual_48G_ada", "96G_blackwell", "spark"])
@pytest.mark.parametrize("stack", ["default", "vlm_llm_nim", "vlm_speech_nim"])
def test_every_bundled_model_stack_has_a_valid_vram_profile(
    hardware: str, stack: str,
) -> None:
    profile = load_vram_profile(
        _REPO_ROOT / "agent-samples/model-servers/yaml" / hardware
        / f"vram.{stack}.json"
    )
    assert profile.hardware_profile == hardware
    assert profile.stack == stack
    deployment = load_deployment_profile(
        _REPO_ROOT / "agent-samples/model-servers/yaml" / f"models.{stack}.json"
    )
    assert {item.service for item in profile.services} == {
        service for service, mode in deployment.services.items() if mode == "own"
    }
