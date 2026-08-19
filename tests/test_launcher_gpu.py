# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for strict per-device GPU inventory and profile matching."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from xr_ai_launcher import GPUInventoryError, detect_gpu_config, query_gpu_inventory

_PROFILES = Path(__file__).resolve().parents[1] / "agent-samples/model-servers/yaml"


def _row(
    index: int, name: str, cap: float, total_mib: float,
    *, free_mib: float | None = None, used_mib: float = 0,
) -> str:
    free = total_mib - used_mib if free_mib is None else free_mib
    return (
        f"{index}, GPU-{index}, 00000000:{index:02x}:00.0, {name}, {cap}, "
        f"{total_mib}, {free}, {used_mib}"
    )


def _mock_smi(lines: list[str]):
    return patch(
        "xr_ai_launcher._gpu.subprocess.check_output",
        return_value="\n".join(lines),
    )


def test_inventory_records_each_gpu_capacity_independently() -> None:
    with _mock_smi([
        _row(0, "NVIDIA L40S", 8.9, 46068, free_mib=45000, used_mib=1068),
        _row(1, "NVIDIA L40S", 8.9, 46068, free_mib=44000, used_mib=2068),
    ]):
        inventory = query_gpu_inventory()

    assert [gpu.index for gpu in inventory] == [0, 1]
    assert inventory[0].total_memory_gib == pytest.approx(44.988, abs=0.001)
    assert inventory[1].free_memory_gib == pytest.approx(42.969, abs=0.001)


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError(), subprocess.CalledProcessError(1, "nvidia-smi")],
)
def test_detection_failure_does_not_select_an_unsafe_default(error: Exception) -> None:
    with patch("xr_ai_launcher._gpu.subprocess.check_output", side_effect=error):
        with pytest.raises(GPUInventoryError, match="nvidia-smi"):
            detect_gpu_config(_PROFILES)


def test_empty_inventory_is_rejected() -> None:
    with _mock_smi([]), pytest.raises(GPUInventoryError, match="no GPUs"):
        detect_gpu_config(_PROFILES)


def test_unparseable_inventory_is_rejected() -> None:
    with _mock_smi(["not a valid row"]), pytest.raises(
        GPUInventoryError, match="unparseable",
    ):
        detect_gpu_config(_PROFILES)


def test_single_ada_does_not_match_dual_profile() -> None:
    with _mock_smi([_row(0, "RTX 6000 Ada", 8.9, 49140)]), pytest.raises(
        GPUInventoryError, match="no bundled",
    ):
        detect_gpu_config(_PROFILES)


def test_dual_ada_requires_minimum_memory_on_each_gpu() -> None:
    with _mock_smi([
        _row(0, "NVIDIA L40S", 8.9, 46068),
        _row(1, "NVIDIA L40S", 8.9, 46068),
    ]):
        assert detect_gpu_config(_PROFILES).name == "dual_48G_ada"

    with _mock_smi([
        _row(0, "NVIDIA L40S", 8.9, 46068),
        _row(1, "RTX 4090", 8.9, 24564),
    ]), pytest.raises(GPUInventoryError, match="GPU 1"):
        detect_gpu_config(_PROFILES)


def test_single_large_blackwell_matches_workstation_profile() -> None:
    with _mock_smi([_row(0, "RTX PRO 6000 Blackwell", 12.0, 98304)]):
        assert detect_gpu_config(_PROFILES).name == "96G_blackwell"


@pytest.mark.parametrize("name", ["NVIDIA GB10", "NVIDIA B10"])
def test_spark_name_matches_spark(name: str) -> None:
    with _mock_smi([_row(0, name, 10.0, 131072)]):
        assert detect_gpu_config(_PROFILES).name == "spark"
