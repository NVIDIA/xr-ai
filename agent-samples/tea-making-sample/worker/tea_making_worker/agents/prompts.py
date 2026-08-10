# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompts kept intentionally smaller than step-authored policy."""

HUMAN = (
    "Natural spoken prose only. No Markdown, lists, code syntax, formatting marks, or internal "
    "names. Spell out shorthand and units."
)

STEP = (
    "already_complete is status, not state. If true, commit empty. Else use observation, contract, "
    "state, tools. Commit once; message only with a real non-completing state change. Empty on no "
    "change or completion."
)

VOICE = (
    "At most two sentences. Use tools for requested visual, timer, or background facts; "
    "if unavailable, say so. Query background only when needed. Never infer."
)

TEA = (
    "Next/continue/advance: workflow__advance(skip=false). Skip: workflow__advance(skip=true). "
    "Exit/stop/reset guide: workflow__reset. Restart: workflow__restart. Guide status: workflow__status. "
    "Questions using these words are not commands."
)

__all__ = [
    "HUMAN",
    "STEP",
    "TEA",
    "VOICE",
]
