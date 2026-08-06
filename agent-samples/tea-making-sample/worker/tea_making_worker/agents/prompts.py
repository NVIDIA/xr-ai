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

ROUTER = (
    "One tool; never answer. Guide lifecycle only: start; next/continue/advance/skip; stop/reset; "
    "restart/start over; status. Timer/appliance uses ask_step, even with lifecycle words. If active and input "
    "concerns this step/item, a live fact, or an action, ask_step. Else ask_general."
)

__all__ = ["GENERAL", "HUMAN", "ROUTER", "STEP", "VOICE"]
