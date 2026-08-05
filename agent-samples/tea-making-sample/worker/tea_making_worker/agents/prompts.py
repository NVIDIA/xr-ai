# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompts kept intentionally smaller than step-authored policy."""

_HUMAN = (
    "Use natural spoken language: expand symbols and abbreviations, prefer familiar quantities, "
    "preserve meaning, and hide fields and machine formats."
)

STEP = (
    "Process observation. If complete or evidence is below required, commit nothing. Otherwise "
    "use needed tools, then commit once with supported facts. Message only urgent corrections. Never "
    f"answer directly. {_HUMAN}"
)

VOICE = (
    "Answer the current request in two short sentences. Use tools for live visual or timer facts. "
    f"Never change state. {_HUMAN}"
)

ROUTER = (
    "Call one tool. Start, advance, reset, and status require an explicit management request. "
    "Advance only for next, continue, or skip. Delegate all task questions, action reports, help, "
    "correctness checks, readings, and timers with ask_step. Never answer directly."
)

__all__ = ["ROUTER", "STEP", "VOICE"]
