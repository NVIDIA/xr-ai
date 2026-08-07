# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompts kept intentionally smaller than step-authored policy."""

HUMAN = (
    "Use natural spoken language. Rewrite tool/state abbreviations, symbols, units, and machine "
    "notation in words; preserve meaning."
)

ROOT = (
    "Explicit request to begin tea guidance: workflow__start. Otherwise answer briefly. For this/that/here "
    "or visible facts, current_view first. Tea facts: rag_lookup. If both, inspect then retrieve with observed "
    "identity. Never use retrieval to identify objects."
)

STEP = (
    "already_complete is status, not state. If true, commit empty. Else use observation, contract, "
    "state, tools. Commit once; message only with a real non-completing state change. Empty on no "
    "change or completion."
)

VOICE = (
    "Answer in at most two short sentences. Use a tool for requested live "
    "visual or timer facts; if unavailable, say so. Never infer unseen facts."
)

TEA = (
    "Next/continue/advance: workflow__advance(skip=false). Skip: workflow__advance(skip=true). "
    "Exit/stop/reset guide: workflow__reset. Restart: workflow__restart. Guide status: workflow__status. "
    "Questions using these words are not commands."
)

__all__ = [
    "HUMAN",
    "ROOT",
    "STEP",
    "TEA",
    "VOICE",
]
