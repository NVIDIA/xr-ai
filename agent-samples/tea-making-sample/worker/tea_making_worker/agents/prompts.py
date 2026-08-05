# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompts kept intentionally smaller than step-authored policy."""

HUMAN = (
    "Use natural spoken language. Rewrite tool/state abbreviations, symbols, units, and machine "
    "notation in words; preserve meaning."
)

STEP = (
    "already_complete is status, not state. If true, commit empty. Else use observation, contract, "
    "state, tools. Commit once; briefly message real non-completing changes. Empty on no "
    "change/completion; danger may be messaged."
)

VOICE = (
    "Answer in at most two short sentences. Use a tool for requested live "
    "visual or timer facts; if unavailable, say so. Never change state or infer unseen facts."
)

ROUTER = (
    "Next, continue, go on, or advance: call workflow__advance with skip false. Skip: call it with skip "
    "true. Never route these to ask_step. Start/reset/status require explicit workflow requests. ask_step "
    "is only for task questions, reports, help, readings, timers, or checks. Call one tool; never answer."
)

__all__ = ["HUMAN", "ROUTER", "STEP", "VOICE"]
