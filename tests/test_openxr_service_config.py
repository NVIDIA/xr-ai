# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""openxr-service config parsing: allow_sim_pose arms a global tracking
override on an open port, so only a real YAML boolean may enable it."""
from __future__ import annotations

from pathlib import Path

import pytest
from openxr_service.__main__ import _load_config


def _write(tmp_path: Path, allow_sim_pose: str) -> Path:
    config = tmp_path / "openxr_service.yaml"
    config.write_text(f"endpoint: tcp://127.0.0.1:0\nallow_sim_pose: {allow_sim_pose}\n")
    return config


def test_real_booleans_parse(tmp_path: Path) -> None:
    assert _load_config(_write(tmp_path, "true")).allow_sim_pose is True
    assert _load_config(_write(tmp_path, "false")).allow_sim_pose is False


def test_omitted_defaults_to_disabled(tmp_path: Path) -> None:
    config = tmp_path / "openxr_service.yaml"
    config.write_text("endpoint: tcp://127.0.0.1:0\n")
    assert _load_config(config).allow_sim_pose is False


@pytest.mark.parametrize("value", ['"false"', '"true"', '"yes"', "1", "enabled"])
def test_non_boolean_values_are_rejected(tmp_path: Path, value: str) -> None:
    with pytest.raises(SystemExit):
        _load_config(_write(tmp_path, value))
