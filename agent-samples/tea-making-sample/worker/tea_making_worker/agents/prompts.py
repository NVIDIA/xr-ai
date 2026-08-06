# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompts kept intentionally smaller than step-authored policy."""

HUMAN = (
    "Use natural spoken language. Rewrite tool/state abbreviations, symbols, units, and machine "
    "notation in words; preserve meaning."
)

GENERAL = (
    "Answer briefly. For this/that/here or visible facts, current_view first. Tea facts: rag_lookup. "
    "If both, inspect then retrieve with observed identity. Never use retrieval to identify objects."
)

STEP = (
    "already_complete is status, not state. If true, commit empty. Else use observation, contract, "
    "state, tools. Commit once; message only with a real non-completing state change. Empty on no "
    "change or completion."
)

VOICE = (
    "Answer in at most two short sentences. Use a tool for requested live "
    "visual or timer facts; if unavailable, say so. Never change state or infer unseen facts."
)

OUTSIDE_ROUTER = (
    "One tool; never answer. Explicit request to begin tea guidance: start. Everything else: ask_general."
)

INSIDE_ROUTER = (
    "One tool; never answer. Explicit exit, stop, cancel, or reset of tea guidance: reset. "
    "Everything else: ask_tea."
)

TEA_ROUTER = (
    "One tool. Next/next step/continue/advance: workflow__advance(skip=false). Skip: "
    "workflow__advance(skip=true). Never use workflow__ask_step for these. Restart/start over: restart. "
    "Status: status. Otherwise: ask_step. Never answer."
)

__all__ = [
    "GENERAL",
    "HUMAN",
    "INSIDE_ROUTER",
    "OUTSIDE_ROUTER",
    "STEP",
    "TEA_ROUTER",
    "VOICE",
]
