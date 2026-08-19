# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-YAML GPU-memory requirement and preflight tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from xr_ai_launcher import (
    GPUDevice,
    GPUMemoryError,
    derive_gpu_memory_utilization,
    format_gpu_memory_preflight,
    load_service_gpu_requirement,
    preflight_gpu_memory,
    require_gpu_memory_preflight,
    resolve_gpu_memory_plan,
    utilization_overrides,
)
from xr_ai_vllm import gpu_memory_utilization

_ROOT = Path(__file__).resolve().parents[1]


def _gpu(
    index: int, *, total: float = 45.0, free: float = 44.0, capability: float = 8.9,
) -> GPUDevice:
    return GPUDevice(
        index=index,
        uuid=f"GPU-{index}",
        pci_bus_id=f"0000:{index:02x}:00.0",
        name="NVIDIA L40S",
        compute_capability=capability,
        total_memory_gib=total,
        free_memory_gib=free,
        used_memory_gib=total - free,
    )


def _service(tmp_path: Path, name: str, gpu: int, memory: float) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"port: 8100\ncuda_visible_devices: {gpu}\n"
        f"gpu_memory_reservation_gib: {memory}\n",
        encoding="utf-8",
    )
    return path


def _plan(tmp_path: Path, omni_memory: float = 37.0):
    inventory = (_gpu(0), _gpu(1))
    return resolve_gpu_memory_plan(
        stack="test",
        inventory=inventory,
        service_configs={
            "omni": (_service(tmp_path, "omni", 1, omni_memory), True),
            "stt": (_service(tmp_path, "stt", 1, 3.0), False),
        },
    )


def test_preflight_accounts_for_complete_per_gpu_stack(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    results = preflight_gpu_memory(plan, (_gpu(0), _gpu(1)))

    assert results[0].incremental_required_gib == 40.0
    assert results[0].remaining_gib == 2.0
    require_gpu_memory_preflight(results)
    assert "Result: PASS" in format_gpu_memory_preflight(plan, results)


def test_preflight_fails_cleanly_when_free_memory_telemetry_is_unavailable(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    unavailable = GPUDevice(
        index=1,
        uuid="GPU-1",
        pci_bus_id="0000:01:00.0",
        name="NVIDIA GB10",
        compute_capability=10.0,
        total_memory_gib=None,
        free_memory_gib=None,
        used_memory_gib=None,
    )

    with pytest.raises(GPUMemoryError, match="free GPU memory telemetry is unavailable"):
        preflight_gpu_memory(plan, (_gpu(0), unavailable))


def test_preflight_reports_shortfall_without_double_counting_active_service(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, 42.0)
    inventory = (_gpu(0), _gpu(1))

    with pytest.raises(GPUMemoryError, match=r"shortfall 3\.0 GiB"):
        require_gpu_memory_preflight(preflight_gpu_memory(plan, inventory))
    require_gpu_memory_preflight(preflight_gpu_memory(
        plan, inventory, active_services=frozenset({"omni"}),
    ))


def test_legacy_gpu_memory_utilization_remains_supported(tmp_path: Path) -> None:
    config = tmp_path / "legacy.yaml"
    config.write_text(
        "cuda_visible_devices: 0\ngpu_memory_utilization: 0.20\n",
        encoding="utf-8",
    )

    requirement = load_service_gpu_requirement(
        "vlm", config, (_gpu(0),), vllm=True,
    )

    assert requirement.memory_gib == 9.0


def test_vllm_utilization_is_derived_from_absolute_gib(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    inventory = (_gpu(0), _gpu(1, total=45.7))

    assert derive_gpu_memory_utilization(37.0, 45.7) == 0.8097
    assert utilization_overrides(plan, inventory) == {"omni": "0.8097"}


def test_service_prefers_launcher_derived_utilization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XR_AI_GPU_MEMORY_UTILIZATION", "0.8123")

    assert gpu_memory_utilization({"gpu_memory_utilization": 0.5}, 0.85) == 0.8123


def test_simple_vlm_contract_fits_one_45_gib_l40s() -> None:
    sample = _ROOT / "agent-samples/simple-vlm-example/yaml"
    inventory = (_gpu(0, total=45.0, free=45.0),)
    plan = resolve_gpu_memory_plan(
        stack="simple-vlm-example/local",
        inventory=inventory,
        service_configs={
            "vlm": (sample / "vlm_server.yaml", True),
            "stt": (sample / "stt_server.yaml", False),
        },
    )

    results = preflight_gpu_memory(plan, inventory)
    require_gpu_memory_preflight(results)
    assert results[0].incremental_required_gib == 27.0
    assert results[0].remaining_gib == 16.0


@pytest.mark.parametrize("directory", ["dual_48G_ada", "96G_blackwell", "spark"])
def test_bundled_model_server_yaml_declares_gpu_memory(directory: str) -> None:
    root = _ROOT / "agent-samples/model-servers/yaml" / directory
    for config in root.glob("*_server*.yaml"):
        body = config.read_text(encoding="utf-8")
        assert "gpu_memory_reservation_gib:" in body, config


def test_spark_absolute_requirements_preserve_previous_utilization_budgets() -> None:
    root = _ROOT / "agent-samples/model-servers/yaml/spark"
    expected = {
        "vlm_server.yaml": 24.0,
        "nemotron_omni_llm_server.yaml": 30.0,
        "embedding_server.yaml": 9.6,
    }
    for filename, memory_gib in expected.items():
        requirement = load_service_gpu_requirement(
            filename,
            root / filename,
            (_gpu(0, total=120.0, free=120.0, capability=10.0),),
            vllm=True,
        )
        assert requirement.memory_gib == memory_gib
