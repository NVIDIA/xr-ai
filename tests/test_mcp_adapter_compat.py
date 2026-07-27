# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Back-compat tests for the deprecated ``xr_ai_nat.adapters.mcp`` alias.

The MCP publisher moved from ``xr_ai_nat.adapters.mcp`` to ``xr_ai_nat.mcp``.
``xr_ai_nat.adapters.mcp`` was a documented public import, so it must remain as
a deprecated forwarding alias rather than a silent breaking change. These tests
pin that contract: the legacy path still resolves to the same
``create_mcp_server`` object and emits a ``DeprecationWarning`` on import.
"""
from __future__ import annotations

import importlib

import pytest


def test_legacy_import_resolves_to_same_object() -> None:
    from xr_ai_nat.adapters.mcp import create_mcp_server as legacy
    from xr_ai_nat.mcp import create_mcp_server as canonical

    assert legacy is canonical


def test_legacy_import_emits_deprecation_warning() -> None:
    import xr_ai_nat.adapters.mcp as legacy_module

    with pytest.warns(DeprecationWarning, match="xr_ai_nat.mcp"):
        importlib.reload(legacy_module)
