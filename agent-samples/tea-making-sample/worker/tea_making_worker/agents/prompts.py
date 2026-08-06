# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompts kept intentionally smaller than step-authored policy."""

HUMAN = (
    "Use natural spoken language. Rewrite tool/state abbreviations, symbols, units, and machine "
    "notation in words; preserve meaning."
)

GENERAL = (
    "Answer in at most two short sentences. Use current_view for visible facts and rag_lookup for tea "
    "or brewing knowledge; use both only when needed. Never let retrieval identify visible objects."
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

ROUTER = (
    "Explicit next/continue/advance calls workflow__advance; skip sets skip true. Explicit "
    "start/reset/status calls its tool. Mentions are not commands. Current-step/item questions or reports "
    "call ask_step. General knowledge, including tea, or visual requests call ask_general. One tool; "
    "never answer."
)

__all__ = ["GENERAL", "HUMAN", "ROUTER", "STEP", "VOICE"]
