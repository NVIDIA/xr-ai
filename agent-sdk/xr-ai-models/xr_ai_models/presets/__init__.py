# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in presets for in-tree AI services.

Each preset is a dict with the same keys a ``models.yaml`` entry would carry
(except ``base_url``, which is the caller's responsibility).  A YAML entry
that says ``kind: preset:<name>`` merges its keys on top of the preset's;
explicit keys win.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .cosmos_vlm      import COSMOS_VLM
from .llama_nemotron  import LLAMA_NEMOTRON
from .magpie_tts      import MAGPIE_TTS
from .nemotron3_nano  import NEMOTRON3_NANO
from .nemotron_omni   import NEMOTRON_OMNI
from .parakeet_stt    import PARAKEET_STT
from .piper_tts       import PIPER_TTS


_PRESETS: dict[str, dict[str, Any]] = {
    "cosmos_vlm":     COSMOS_VLM,
    "llama_nemotron": LLAMA_NEMOTRON,
    "magpie_tts":     MAGPIE_TTS,
    "nemotron3_nano": NEMOTRON3_NANO,
    "nemotron_omni":  NEMOTRON_OMNI,
    "parakeet_stt":   PARAKEET_STT,
    "piper_tts":      PIPER_TTS,
}


def get_preset(name: str) -> dict[str, Any]:
    try:
        return deepcopy(_PRESETS[name])
    except KeyError as exc:
        raise KeyError(
            f"unknown preset {name!r}; known: {sorted(_PRESETS)}"
        ) from exc


def available_presets() -> list[str]:
    return sorted(_PRESETS)


__all__ = ["get_preset", "available_presets"]
