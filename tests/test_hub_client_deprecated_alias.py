# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The renamed hub client keeps a deprecated ``xr_ai_agent`` forwarding alias.

``xr_ai_agent`` was a public top-level import before the package became
``xr_ai_hub``. The ``xr-ai-hub-client`` distribution therefore also ships an
``xr_ai_agent`` package that re-exports the canonical objects and warns on
import, so out-of-tree code keeps importing while it migrates. (The *dependency*
rename is breaking; only the import path is aliased — see the package README.)
"""

import importlib
import warnings

import pytest
import xr_ai_hub


def test_canonical_import_is_warning_free() -> None:
    # The alias must not fire transitively: importing the canonical package
    # should never emit the deprecation warning.
    module = importlib.import_module("xr_ai_hub")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.reload(module)


def test_deprecated_alias_warns_once() -> None:
    module = importlib.import_module("xr_ai_agent")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(module)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert "xr_ai_hub" in str(deprecations[0].message)


def test_every_public_name_forwards_to_the_same_object() -> None:
    """The load-bearing assertion.

    The alias re-exports via ``from xr_ai_hub import *``, which only copies names
    listed in ``__all__`` — so a public name missing from that list would
    silently fail to forward. Check every one resolves to the identical object.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        alias = importlib.import_module("xr_ai_agent")

    missing = [name for name in xr_ai_hub.__all__ if not hasattr(alias, name)]
    assert missing == [], f"names absent from the alias: {missing}"
    mismatched = [
        name
        for name in xr_ai_hub.__all__
        if getattr(alias, name) is not getattr(xr_ai_hub, name)
    ]
    assert mismatched == [], f"names not forwarding to the canonical object: {mismatched}"


def test_alias_all_matches_canonical_all() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        alias = importlib.import_module("xr_ai_agent")

    assert sorted(alias.__all__) == sorted(xr_ai_hub.__all__)


@pytest.mark.parametrize("name", ["ProcessorEndpoint", "DataMessage", "LiveFrameSource"])
def test_representative_names_are_importable_from_the_alias(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        alias = importlib.import_module("xr_ai_agent")

    assert getattr(alias, name) is getattr(xr_ai_hub, name)
