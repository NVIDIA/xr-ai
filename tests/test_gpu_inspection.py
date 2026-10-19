# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared hardware queries preserve deadlines and caller failure policy."""
import subprocess
from unittest.mock import patch

import pytest
from test_cloudxr_runtime_process import _import_cloudxr_runtime
from xr_ai_gpu import (
    GPUInventoryError,
    query_gpu_inventory,
    query_gpu_pci_bus_ids,
    query_gpu_processes,
)
from xr_ai_launcher import detect_gpu_config


@pytest.mark.parametrize("query", [query_gpu_inventory, query_gpu_pci_bus_ids, query_gpu_processes])
@pytest.mark.parametrize("error", [
    FileNotFoundError("missing"), PermissionError("not executable"),
    subprocess.CalledProcessError(1, "nvidia-smi", stderr="driver unavailable"),
    subprocess.TimeoutExpired("nvidia-smi", 5),
])
def test_public_queries_normalize_failures(query, error):
    with patch("xr_ai_gpu._query.subprocess.check_output", side_effect=error):
        with pytest.raises(GPUInventoryError) as raised:
            query(timeout=5)
    assert raised.value.__cause__ is error


def test_pci_query_ignores_cuda_remapping_and_tolerates_bad_rows(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    with patch("xr_ai_gpu._query.subprocess.check_output", return_value=(
        "broken\nno-index, ignored\n0, 00000000:01:00.0\n1, 00000000:02:00.0\n"
    )) as command:
        assert query_gpu_pci_bus_ids(timeout=5) == {
            0: "00000000:01:00.0", 1: "00000000:02:00.0",
        }
    assert command.call_args.kwargs["timeout"] == 5
    assert command.call_args.args[0] == [
        "nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits",
    ]


def test_launcher_inventory_and_diagnostic_query_keep_no_deadline():
    with patch("xr_ai_gpu._query.subprocess.check_output", side_effect=[
        "0, GPU-0, 00000000:01:00.0, L40S, 8.9, 46068, 46068, 0", "",
    ]) as command:
        with pytest.raises(GPUInventoryError, match="no existing"):
            detect_gpu_config()
    assert command.call_count == 2
    assert all("timeout" not in call.kwargs for call in command.call_args_list)


def test_cloudxr_uses_shared_pci_query_with_existing_deadline(monkeypatch):
    module = _import_cloudxr_runtime(monkeypatch)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    with patch("xr_ai_gpu._query.subprocess.check_output", return_value="1, 00000000:2a:00.0") as command:
        assert module._xr_gpu_env(1) == {
            "CUDA_VISIBLE_DEVICES": "1", "VK_LOADER_DEVICE_SELECT": "PCI:2a:00:0",
            "DRI_PRIME": "pci-0000_2a_00_0",
        }
    assert command.call_args.kwargs["timeout"] == 5.0


@pytest.mark.parametrize("result", ["", "malformed", "0, invalid-bus", "2, 00000000:02:00.0"])
def test_cloudxr_skips_pinning_for_unusable_inventory(monkeypatch, result):
    module = _import_cloudxr_runtime(monkeypatch)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    with patch("xr_ai_gpu._query.subprocess.check_output", return_value=result):
        assert module._xr_gpu_env(0) is None


@pytest.mark.parametrize("error", [PermissionError(), subprocess.TimeoutExpired("nvidia-smi", 5)])
def test_cloudxr_continues_without_pinning_on_driver_failure(monkeypatch, error):
    module = _import_cloudxr_runtime(monkeypatch)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    with patch("xr_ai_gpu._query.subprocess.check_output", side_effect=error):
        assert module._xr_gpu_env(0) is None
