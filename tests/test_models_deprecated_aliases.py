# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The privatized xr-ai-models modules keep deprecated forwarding aliases.

``config``/``factory``/``openai_compat``/``protocols`` moved under underscore
names; the old public module paths remain as thin, warning-emitting shims that
re-export the canonical objects, so external imports keep working.
"""

import importlib

import pytest
import xr_ai_models
from xr_ai_models import _config, _factory, _openai_compat, _protocols

# old public module path -> [(attribute, canonical object)]
_ALIASES = {
    "xr_ai_models.protocols": [
        ("VLMService", _protocols.VLMService),
        ("ChatMessage", _protocols.ChatMessage),
    ],
    "xr_ai_models.openai_compat": [
        ("OpenAICompatVLM", _openai_compat.OpenAICompatVLM),
    ],
    "xr_ai_models.config": [
        ("load_models_config", _config.load_models_config),
        ("KIND_OPENAI_COMPAT", _config.KIND_OPENAI_COMPAT),
        # Config-vocabulary names not re-exported at the package root must still
        # import from the deprecated module path.
        ("Category", _config.Category),
        ("Spec", _config.Spec),
        ("ModelKind", _config.ModelKind),
    ],
    "xr_ai_models.factory": [
        ("make_vlm", _factory.make_vlm),
    ],
}


@pytest.mark.parametrize("module_name", list(_ALIASES))
def test_deprecated_module_alias_forwards_and_warns(module_name: str) -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        module = importlib.import_module(module_name)
        importlib.reload(module)
    for attr, canonical in _ALIASES[module_name]:
        assert getattr(module, attr) == canonical


def test_public_names_reachable_from_package_root() -> None:
    # Privatization must not change the package's public API surface.
    assert xr_ai_models.VLMService is _protocols.VLMService
    assert xr_ai_models.OpenAICompatVLM is _openai_compat.OpenAICompatVLM
    assert xr_ai_models.load_models_config is _config.load_models_config
    assert xr_ai_models.make_vlm is _factory.make_vlm
