# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional transport and framework adapters for native XR functions.

The voice adapters are the public way to drive a native function from a voice
session:

    from xr_ai_nat.adapters import as_voice_handler, record_voice_transcripts

They are re-exported lazily because they need the optional ``[voice]`` extra
(``xr-ai-voice``). Importing this package without that extra stays cheap and
succeeds; only touching an attribute raises, and the error names the extra to
install. Lazy access also keeps the deprecated ``adapters.mcp`` alias from
emitting its warning on an unrelated import of this package.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from .voice import as_voice_handler, record_voice_transcripts

_VOICE_EXPORTS = frozenset({"as_voice_handler", "record_voice_transcripts"})

__all__ = ["as_voice_handler", "record_voice_transcripts"]


def __getattr__(name: str):
    """Resolve the voice adapters on first access (PEP 562)."""
    if name in _VOICE_EXPORTS:
        try:
            from . import voice
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                f"xr_ai_nat.adapters.{name} requires the optional 'voice' extra; "
                "install xr-ai-nat[voice]."
            ) from exc
        return getattr(voice, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
