# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-YAML VRAM reservation, preflight, and measurement tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from xr_ai_launcher import (
    GPUDevice,
    GPUHardwareProfile,
    GPUProcess,
    VRAMProfileError,
    derive_gpu_memory_utilization,
    format_vram_preflight,
    load_gpu_hardware_profile,
    load_service_reservation,
    preflight_vram,
    require_vram_preflight,
    resolve_vram_profile,
    utilization_overrides,
)
from xr_ai_launcher.vram_measure import main as measure_main
from xr_ai_vllm._config import gpu_memory_utilization
from xr_ai_vllm._diagnostics import classify_vllm_failure

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _gpu(index: int, *, total: float = 45.0, free: float = 44.0) -> GPUDevice:
    return GPUDevice(
        index=index, uuid=f"GPU-{index}", pci_bus_id=f"0000:{index:02x}:00.0",
        name="NVIDIA L40S", compute_capability=8.9, total_memory_gib=total,
        free_memory_gib=free, used_memory_gib=total - free,
        processes=(GPUProcess(f"GPU-{index}", 123, "python", total - free),),
    )


def _hardware(tmp_path: Path) -> GPUHardwareProfile:
    path = tmp_path / "gpu_profile.yaml"
    path.write_text(
        "profile: test\nrequired_gpu_count: 2\nmin_compute_capability: 8.9\n"
        "min_memory_gib_per_gpu: 44\ndevice_safety_reserve_gib: 2\n"
    )
    return load_gpu_hardware_profile(path)


def _service(tmp_path: Path, name: str, gpu: int, reservation: float) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"port: 81{gpu:02d}\ncuda_visible_devices: \"{gpu}\"\n"
        f"gpu_memory_reservation_gib: {reservation}\n"
    )
    return path


def _profile(tmp_path: Path, reservation: float = 37.0):
    inventory = (_gpu(0), _gpu(1))
    return resolve_vram_profile(
        stack="test", hardware=_hardware(tmp_path), inventory=inventory,
        service_configs={
            "omni": (_service(tmp_path, "omni", 1, reservation), True),
            "stt": (_service(tmp_path, "stt", 1, 3.0), False),
        },
    )


def test_preflight_accounts_for_complete_per_gpu_stack(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    results = preflight_vram(profile, (_gpu(0), _gpu(1)))
    assert results[0].incremental_required_gib == 40.0
    assert results[0].remaining_gib == 2.0
    require_vram_preflight(results)
    assert "Existing PID 123" in format_vram_preflight(profile, results)


def test_preflight_reports_shortfall_and_active_service_is_not_double_counted(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path, 42.0)
    with pytest.raises(VRAMProfileError, match=r"shortfall 3\.0 GiB"):
        require_vram_preflight(preflight_vram(profile, (_gpu(0), _gpu(1))))
    require_vram_preflight(preflight_vram(
        profile, (_gpu(0), _gpu(1)), active_services=frozenset({"omni"}),
    ))


def test_vllm_utilization_is_derived_upward_from_absolute_gib(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    inventory = (_gpu(0), _gpu(1, total=45.7))
    assert derive_gpu_memory_utilization(37.0, 45.7) == 0.8097
    assert utilization_overrides(profile, inventory) == {"omni": "0.8097"}


def test_legacy_gpu_memory_utilization_remains_supported(tmp_path: Path) -> None:
    config = tmp_path / "legacy.yaml"
    config.write_text("cuda_visible_devices: 0\ngpu_memory_utilization: 0.20\n")
    reservation = load_service_reservation("vlm", config, (_gpu(0),), vllm=True)
    assert reservation is not None
    assert reservation.reservation_gib == 9.0


def test_service_prefers_launcher_derived_utilization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XR_AI_GPU_MEMORY_UTILIZATION", "0.8123")
    assert gpu_memory_utilization({"gpu_memory_utilization": 0.5}, 0.85) == 0.8123


def test_certify_updates_service_yaml_and_detects_later_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "omni.yaml"
    config.write_text("cuda_visible_devices: 1\nmax_model_len: 32768\n")
    signature = {"driver_version": "580.0", "git_commit": "abc123"}
    measurement = tmp_path / "omni.measurement.json"
    measurement.write_text(json.dumps({
        "kind": "xr-ai-vram-measurement", "measurement_signature": signature,
        "summary": {"1": {"recommended_reservation_gib": 40.5}},
    }))
    assert measure_main([
        "certify", "--config", str(config), "--gpu", "1",
        "--minimum-runs", "1", "--measurement", str(measurement),
    ]) == 0
    monkeypatch.setattr(
        "xr_ai_launcher._vram.subprocess.check_output",
        lambda _command, **_kwargs: "580.0",
    )
    reservation = load_service_reservation("omni", config, (_gpu(1),), vllm=True)
    assert reservation is not None
    assert reservation.reservation_gib == 40.5
    config.write_text(config.read_text() + "max_num_seqs: 9\n")
    with pytest.raises(VRAMProfileError, match="changed since"):
        load_service_reservation("omni", config, (_gpu(1),), vllm=True)


def test_vllm_kv_cache_failure_is_classified_as_insufficient_vram(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(
        "Available KV cache memory: -0.17 GiB\n"
        "ValueError: No available memory for the cache blocks.\n"
    )
    diagnosis = classify_vllm_failure(log, ["vllm", "--gpu-memory-utilization", "0.78"])
    assert diagnosis is not None and diagnosis.startswith("INSUFFICIENT VRAM")
    assert "-0.17 GiB" in diagnosis


@pytest.mark.parametrize("hardware", ["dual_48G_ada", "96G_blackwell", "spark"])
def test_every_bundled_hardware_profile_and_service_yaml_has_reservation(
    hardware: str,
) -> None:
    directory = _REPO_ROOT / "agent-samples/model-servers/yaml" / hardware
    assert load_gpu_hardware_profile(directory / "gpu_profile.yaml").name == hardware
    for config in directory.glob("*_server*.yaml"):
        assert "gpu_memory_reservation_gib:" in config.read_text(), config
