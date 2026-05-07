# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_launcher._gpu.detect_gpu_config."""
from __future__ import annotations

from unittest.mock import patch

from xr_ai_launcher._gpu import detect_gpu_config


def _mock_smi(lines: list[str]):
    """Return a context manager that patches check_output to return *lines*."""
    output = "\n".join(lines)
    return patch(
        "xr_ai_launcher._gpu.subprocess.check_output",
        return_value=output,
    )


class TestDetectGpuConfig:
    def test_nvidia_smi_unavailable_returns_default(self):
        with patch(
            "xr_ai_launcher._gpu.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            assert detect_gpu_config() == "dual_48G_ada"

    def test_empty_output_returns_default(self):
        with _mock_smi([]):
            assert detect_gpu_config() == "dual_48G_ada"

    def test_unparseable_line_skipped(self):
        # Only one valid line (an Ada GPU), the other is garbage.
        with _mock_smi(["RTX 4090, 8.9, 24564 MiB", "not a valid line"]):
            # one GPU, Ada (cap<10), <2, falls back to dual_48G_ada
            result = detect_gpu_config()
            assert result == "dual_48G_ada"

    # ── Ada / dual-GPU scenarios ───────────────────────────────────────────────

    def test_single_ada_gpu_returns_dual_48G_ada(self):
        with _mock_smi(["RTX 6000 Ada, 8.9, 49140 MiB"]):
            assert detect_gpu_config() == "dual_48G_ada"

    def test_dual_ada_gpus_returns_dual_48G_ada(self):
        with _mock_smi([
            "RTX 6000 Ada, 8.9, 49140 MiB",
            "RTX 6000 Ada, 8.9, 49140 MiB",
        ]):
            assert detect_gpu_config() == "dual_48G_ada"

    # ── Blackwell scenarios ────────────────────────────────────────────────────

    def test_blackwell_96gb_returns_96G_blackwell(self):
        with _mock_smi(["RTX PRO 6000 Blackwell, 12.0, 98304 MiB"]):
            assert detect_gpu_config() == "96G_blackwell"

    def test_blackwell_large_vram_returns_spark(self):
        # >=120 GiB total → spark profile
        with _mock_smi(["GB200, 10.0, 131072 MiB"]):
            assert detect_gpu_config() == "spark"

    def test_spark_name_gb10_returns_spark(self):
        with _mock_smi(["NVIDIA GB10, 10.0, 0 MiB"]):
            assert detect_gpu_config() == "spark"

    def test_spark_name_b10_returns_spark(self):
        with _mock_smi(["NVIDIA B10, 10.0, 0 MiB"]):
            assert detect_gpu_config() == "spark"

    def test_blackwell_no_mem_data_returns_spark(self):
        # compute_cap >= 10 and no parseable mem → spark
        with _mock_smi(["Blackwell GPU, 10.0, N/A"]):
            assert detect_gpu_config() == "spark"

    # ── Robustness ─────────────────────────────────────────────────────────────

    def test_extra_whitespace_tolerated(self):
        with _mock_smi(["  RTX 6000 Ada  ,  8.9  ,  49140 MiB  "]):
            assert detect_gpu_config() == "dual_48G_ada"

    def test_subprocess_error_returns_default(self):
        import subprocess
        with patch(
            "xr_ai_launcher._gpu.subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
        ):
            assert detect_gpu_config() == "dual_48G_ada"
