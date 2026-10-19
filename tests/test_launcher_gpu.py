# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for strict per-device GPU inventory."""
from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest
from xr_ai_launcher import GPUInventoryError, detect_gpu_config
from xr_ai_launcher._gpu import _query_gpu_inventory


def _row(
    index: int,
    name: str,
    cap: float,
    total_mib: float | str,
    *,
    free_mib: float | str | None = None,
    used_mib: float | str = 0,
) -> str:
    if free_mib is not None:
        free = free_mib
    elif isinstance(total_mib, (float, int)) and isinstance(used_mib, (float, int)):
        free = total_mib - used_mib
    else:
        free = "[N/A]"
    return (
        f"{index}, GPU-{index}, 00000000:{index:02x}:00.0, {name}, {cap}, "
        f"{total_mib}, {free}, {used_mib}"
    )


def _mock_smi(
    lines: list[str],
    process_lines: list[str] | None = None,
    process_error: Exception | None = None,
):
    def output(command: list[str], **_kwargs) -> str:
        if any("query-compute-apps" in value for value in command):
            if process_error is not None:
                raise process_error
            return "\n".join(process_lines or [])
        return "\n".join(lines)

    return patch("xr_ai_launcher._gpu.subprocess.check_output", side_effect=output)


def test_inventory_records_each_gpu_capacity() -> None:
    with _mock_smi(
        [
            _row(0, "NVIDIA L40S", 8.9, 46068, free_mib=45000, used_mib=1068),
            _row(1, "NVIDIA L40S", 8.9, 46068, free_mib=44000, used_mib=2068),
        ],
    ):
        inventory = _query_gpu_inventory()

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
            detect_gpu_config()


def test_empty_inventory_is_rejected() -> None:
    with _mock_smi([]), pytest.raises(GPUInventoryError, match="no GPUs"):
        detect_gpu_config()


def test_unparseable_inventory_is_rejected() -> None:
    with _mock_smi(["not a valid row"]), pytest.raises(
        GPUInventoryError, match="unparseable",
    ):
        detect_gpu_config()


def test_failed_process_inventory_does_not_block_supported_topology() -> None:
    error = subprocess.CalledProcessError(1, "nvidia-smi", stderr="not supported")
    with _mock_smi([_row(0, "NVIDIA GB10", 10.0, 131072)], process_error=error):
        assert detect_gpu_config() == "spark"


def test_failed_process_inventory_is_context_not_a_replacement_error() -> None:
    error = subprocess.CalledProcessError(1, "nvidia-smi", stderr="not supported")
    with _mock_smi(
        [_row(0, "NVIDIA L40S", 8.9, 46068)], process_error=error,
    ), pytest.raises(
        GPUInventoryError,
        match="no existing.*process inventory unavailable.*not supported",
    ):
        detect_gpu_config()


def test_unparseable_process_inventory_is_nonfatal_context() -> None:
    with _mock_smi(
        [_row(0, "NVIDIA L40S", 8.9, 46068)],
        ["not a valid process row"],
    ), pytest.raises(
        GPUInventoryError,
        match="no existing.*process inventory unavailable.*unparseable",
    ):
        detect_gpu_config()


def test_unsupported_topology_reports_active_compute_processes() -> None:
    with _mock_smi(
        [_row(0, "NVIDIA L40S", 8.9, 46068)],
        ["GPU-0, 4321, python, 2048"],
    ), pytest.raises(
        GPUInventoryError, match=r"active compute processes: GPU 0 PID 4321 python \(2\.0 GiB\)",
    ):
        detect_gpu_config()


def test_empty_memory_fields_are_rejected_as_inventory_error() -> None:
    row = "0, GPU-0, 00000000:01:00.0, NVIDIA L40S, 8.9, , , "
    with _mock_smi([row]), pytest.raises(GPUInventoryError, match="unparseable"):
        detect_gpu_config()


@pytest.mark.parametrize("sentinel", ["Not Supported", "[Not Supported]"])
def test_not_supported_memory_fields_are_nullable(sentinel: str) -> None:
    with _mock_smi([_row(
        0, "NVIDIA GB10", 12.1, sentinel, free_mib=sentinel, used_mib=sentinel,
    )]):
        inventory = _query_gpu_inventory()
        assert detect_gpu_config() == "spark"

    assert inventory[0].total_memory_gib is None
    assert inventory[0].free_memory_gib is None
    assert inventory[0].used_memory_gib is None


def test_single_ada_is_not_misclassified_as_dual() -> None:
    with _mock_smi([_row(0, "NVIDIA L40S", 8.9, 46068)]), pytest.raises(
        GPUInventoryError, match="no existing.*GPU 0.*L40S",
    ):
        detect_gpu_config()


def test_dual_ada_requires_capacity_on_each_device() -> None:
    with _mock_smi([
        _row(0, "NVIDIA L40S", 8.9, 46068),
        _row(1, "NVIDIA L40S", 8.9, 46068),
    ]):
        assert detect_gpu_config() == "dual_48G_ada"

    with _mock_smi([
        _row(0, "NVIDIA L40S", 8.9, 46068),
        _row(1, "RTX 4090", 8.9, 24564),
    ]), pytest.raises(GPUInventoryError, match="GPU 1.*24.0 GiB"):
        detect_gpu_config()


def test_single_large_blackwell_matches_workstation_config() -> None:
    with _mock_smi([_row(0, "RTX PRO 6000 Blackwell", 12.0, 98304)]):
        assert detect_gpu_config() == "96G_blackwell"


def test_spark_name_matches_when_memory_fields_are_unavailable() -> None:
    with _mock_smi([_row(
        0, "NVIDIA GB10", 10.0, "[N/A]", free_mib="[N/A]", used_mib="[N/A]",
    )]):
        inventory = _query_gpu_inventory()
        assert inventory[0].total_memory_gib is None
        assert inventory[0].free_memory_gib is None
        assert inventory[0].used_memory_gib is None
        assert detect_gpu_config() == "spark"


def test_capacity_match_rejects_unavailable_total_memory_cleanly() -> None:
    with _mock_smi([_row(0, "Unknown Blackwell", 12.0, "[N/A]")]), pytest.raises(
        GPUInventoryError, match=r"no existing.*N/A total",
    ):
        detect_gpu_config()


def test_small_blackwell_does_not_match_96_gib_config() -> None:
    with _mock_smi([_row(0, "NVIDIA GeForce RTX 5090", 12.0, 32768)]), pytest.raises(
        GPUInventoryError, match=r"no existing.*32\.0 GiB",
    ):
        detect_gpu_config()


@pytest.mark.parametrize("name", ["NVIDIA GB10", "NVIDIA B10"])
def test_spark_name_matches_spark_config(name: str) -> None:
    with _mock_smi([_row(0, name, 10.0, 131072)]):
        assert detect_gpu_config() == "spark"


def test_large_unnamed_blackwell_matches_spark_by_capacity() -> None:
    with _mock_smi([_row(0, "NVIDIA Future Blackwell", 12.0, 131072)]):
        assert detect_gpu_config() == "spark"
