#!/usr/bin/env -S uv run --quiet --with httpx --with fastmcp --with pyyaml --script
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "fastmcp>=0.4", "pyyaml"]
# ///
"""
agent-llm eval harness for xr-render-demo. Talks to the running stack
(no mocks for the LLMs / MCP servers — render-mcp tools are fake-succeeded
so the harness never mutates the live LOVR scene).

Usage:
  ./eval.py                  # run all built-in cases against system.txt
  ./eval.py "Move it down"   # one ad-hoc query, prints raw response
  ./eval.py --prompt PATH    # try an alternate prompt file

By default reads ../worker/prompts/system.txt (the live xr-render-demo
prompt). Edit it and re-run; no stack restart needed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import httpx
import yaml
from fastmcp import Client as McpClient

_HERE       = Path(__file__).resolve().parent
SYS_PROMPT  = (_HERE / "../worker/prompts/system.txt").resolve()

# Borrow the worker's own config loader so eval reads the exact same yaml
# (with the exact same code-side defaults) the live worker reads. Keeps
# the eval honest when MCP ports / URLs move.
sys.path.insert(0, str((_HERE / "../worker").resolve()))
from config import load_config  # noqa: E402  — must follow sys.path tweak
_WORKER_CFG = load_config((_HERE / "../yaml/xr_render_demo_worker.yaml").resolve())

def _agent_llm_base_url() -> str:
    """Read agent_llm.base_url from models.yaml."""
    # WorkerConfig.models_yaml is resolved relative to the live launcher's
    # cwd (the sample root); eval runs from eval/, so anchor it ourselves.
    p = Path(_WORKER_CFG.models_yaml)
    if not p.is_absolute():
        p = (_HERE / ".." / p).resolve()
    with open(p) as f:
        models = yaml.safe_load(f) or {}
    return str(models["agent_llm"]["base_url"]).rstrip("/")


AGENT_LLM   = f"{_agent_llm_base_url()}/v1/chat/completions"  # overridable via --agent-llm
AGENT_MODEL = "llm"                                                   # overridable via --agent-model
AGENT_KEY   = ""                                                      # overridable via --agent-api-key / NGC_API_KEY
RENDER_MCP  = f"{_WORKER_CFG.render_mcp}/mcp"
OXR_MCP     = f"{_WORKER_CFG.oxr_mcp}/mcp"
VLM_MCP     = f"{_WORKER_CFG.vlm_mcp}/mcp"
VIDEO_MCP   = f"{_WORKER_CFG.video_mcp}/mcp"
VEC_MCP     = f"{_WORKER_CFG.vec_mcp}/mcp"

# Tools the worker manages internally; hidden from the agent LLM so
# the eval and the live worker advertise the same tool surface.
WORKER_MANAGED = {"start_xr", "get_health"}

# Mirror the worker's WorkerConfig defaults — same fixture pose for every
# test, so prompt regressions are reproducible.
DEFAULT_POSE = {
    "is_valid": True,
    "position": {"x": 0.0, "y": 1.6, "z": 0.0},
    "forward": {"x": 0.0, "y": 0.0, "z": -1.0},
    "right":   {"x": 1.0, "y": 0.0, "z": 0.0},
    "up":      {"x": 0.0, "y": 1.0, "z": 0.0},
    "yaw_deg": 0.0,
    "pitch_deg": 0.0,
}

# Non-canonical pose (rolled head, off-origin) used by a couple of cases
# that exercise gravity-aligned axis math.
ROLLED_HEAD_POSE = {
    "is_valid": True,
    "position": {"x": 0.05, "y": 1.28, "z": 0.32},
    "forward":  {"x": -0.165, "y": 0.075, "z": -0.983},
    "right":    {"x": 0.926,  "y": 0.356, "z": -0.129},
    "up":       {"x": -0.340, "y": 0.931, "z": 0.128},
    "yaw_deg":   10.1,
    "pitch_deg": 4.3,
}

def _became(prim_type: str | None = None,
            *,
            r_min: float | None = None,
            g_min: float | None = None,
            b_min: float | None = None):
    """Predicate factory: returns a checker that asserts at least one
    add_primitive / update_primitive call sets ``prim_type`` AND each
    requested colour channel reaches the given lower bound.  Facets may
    appear in one call or be split across calls (e.g. shape on one
    update, colour on another).  All requested facets must be observed
    for the predicate to pass."""
    requirements: dict[str, str | float] = {}
    if prim_type is not None:
        requirements["prim_type"] = prim_type
    for ch, thresh in (("r", r_min), ("g", g_min), ("b", b_min)):
        if thresh is not None:
            requirements[ch] = thresh

    def _pred(muts: list[dict]) -> tuple[bool, str]:
        seen = dict.fromkeys(requirements, False)
        for tc in muts:
            if tc["function"]["name"] not in ("add_primitive", "update_primitive"):
                continue
            args = tc["function"]["arguments"]
            args = json.loads(args) if isinstance(args, str) else args
            for key, expected in requirements.items():
                if key == "prim_type":
                    if args.get("prim_type") == expected:
                        seen[key] = True
                else:
                    v = args.get(key)
                    if v is not None and float(v) >= float(expected):
                        seen[key] = True
        if all(seen.values()):
            return True, f"saw {requirements}"
        missing = [k for k, v in seen.items() if not v]
        return False, f"missing facets: {missing} (wanted {requirements})"

    return _pred


def _stacked_vertically(muts: list[dict]) -> tuple[bool, str]:
    """Predicate for ``stack_*`` cases: every add_primitive must share the
    same x/z column and have distinct y values, regardless of absolute
    base height.  Floor stack and eye-level stack are both accepted."""
    adds = [tc for tc in muts if tc["function"]["name"] == "add_primitive"]
    if len(adds) < 2:
        return False, f"need ≥2 add_primitive calls, got {len(adds)}"
    rows = []
    for tc in adds:
        a = tc["function"]["arguments"]
        a = json.loads(a) if isinstance(a, str) else a
        rows.append((a.get("x", 0.0), a.get("y", 0.0), a.get("z", 0.0)))
    xs = {round(r[0], 2) for r in rows}
    zs = {round(r[2], 2) for r in rows}
    if len(xs) > 1 or len(zs) > 1:
        return False, f"x/z not aligned across stack: {rows}"
    ys = sorted(round(r[1], 2) for r in rows)
    for a, b in zip(ys, ys[1:]):
        if b - a < 0.05:
            return False, f"y values not separated (need ≥5 cm gap): {ys}"
    return True, f"stacked at y={ys}"


CASES = [
    # ── direct render ops ─────────────────────────────────────────────────────
    {
        "name":  "make_red_sphere",
        "scene": [],
        "user":  "Make a red sphere.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4)}},
        ],
    },
    {
        "name":  "color_change",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Make it green.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "g": (0.5, 1.0), "r": (0.0, 0.3), "b": (0.0, 0.3)}},
        ],
    },
    {
        "name":  "remove_by_color",
        "scene": [
            {"id": "sphere-0", "type": "sphere", "pos": [0, 1.6, -1.5], "color": [1,0,0], "size": 0.1},
            {"id": "box-0",    "type": "box",    "pos": [0.5, 1.6, -1.5], "color": [0,0.4,1], "size": 0.1},
        ],
        "user":  "Remove the red one.",
        "result": [
            {"tool": "remove_primitive", "args": {"obj_id": "sphere-0"}},
        ],
    },

    # ── object-anchored move (bare direction) ─────────────────────────────────
    {
        "name":  "move_left_one_meter",
        "scene": [{"id": "box-0", "type": "box",
                   "pos": [0.0, 1.7, -1.5], "color": [0, 0.4, 1], "size": 0.1}],
        "user":  "Move the cube left one meter.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "x": (-1.05, -0.95),
                      "y": ( 1.65,  1.75),
                      "z": (-1.55, -1.45)}},
        ],
    },
    # User-anchored: object's current pos is irrelevant, lands relative to user.
    {
        "name":  "move_to_my_right_user_anchored",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [3.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Move it one meter to my right.",
        # "Move it N meters to my right" is a delta (shift by 1 m along
        # the user's right axis), not a teleport.  +x = user's right.
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "x": (3.95, 4.05),
                      "y": (1.55, 1.65),
                      "z": (-1.55, -1.45)}},
        ],
    },
    {
        "name":  "move_above_me_user_anchored",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [1.0, 1.25, -0.15], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Move it above my head.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "x": (-0.05, 0.05),
                      "z": (-0.05, 0.05),
                      "y": (1.9, 3.5)}},
        ],
    },
    {
        "name":  "rolled_head_move_left_1m",
        "scene": [{"id": "box-0", "type": "box",
                   "pos": [0.0, 1.7, -1.5], "color": [0.2, 0.9, 0.9], "size": 0.1}],
        "pose":  ROLLED_HEAD_POSE,
        "user":  "Move the cube left one meter.",
        # Gravity-aligned: head roll/pitch don't bleed into x/z; only horizontal
        # axes change. y stays at the cube's original y.
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "x": (-1.10, -0.85),
                      "y": ( 1.65,  1.75),
                      "z": (-1.55, -1.30)}},
        ],
    },
    {
        "name":  "my_left_when_turned_around",
        "scene": [{"id": "box-0", "type": "box",
                   "pos": [0.0, 1.7, -1.5], "color": [0, 0.4, 1], "size": 0.1}],
        "pose":  {"is_valid": True,
                  "position": {"x": 0.0, "y": 1.6, "z": 0.0},
                  "forward": {"x": 0.0, "y": 0.0, "z": 1.0},
                  "right":   {"x": -1.0, "y": 0.0, "z": 0.0},
                  "up":      {"x": 0.0, "y": 1.0, "z": 0.0},
                  "yaw_deg": 180.0, "pitch_deg": 0.0},
        "user":  "Move the cube one meter to my left.",
        # User facing +Z, so "my left" = world +X. Cube ends up at x ≈ +1.
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0", "x": (0.95, 1.05)}},
        ],
    },
    {
        "name":  "move_down_30cm",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.7, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Move it down 30 centimeters.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "y": (1.35, 1.45)}},
        ],
    },
    # ── object-relative placement (above, behind, etc.) ───────────────────────
    {
        "name":  "above_sphere_30cm",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.5, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Put a blue cube 30 cm above the sphere.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "box",
                      "b": (0.5, 1.0), "r": (0.0, 0.3), "g": (0.0, 0.5),
                      "x": (-0.05, 0.05),
                      "y": ( 1.75, 1.85),
                      "z": (-1.55, -1.45)}},
        ],
    },
    {
        "name":  "behind_cube_with_other_object",
        "scene": [
            {"id": "box-0", "type": "box",
             "pos": [-0.07, 1.59, -1.47], "color": [0, 0.8, 0], "size": 0.1},
            {"id": "sphere-2", "type": "sphere",
             "pos": [1.0, 1.6, -1.5], "color": [0, 0, 1], "size": 0.1},
        ],
        "user":  "Add a red sphere behind the green cube.",
        # Behind cube → z < cube.z (further from user). Anchor is the cube
        # alone — y/x align with cube, not midpoint with the other sphere.
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4),
                      "x": (-0.10, -0.04),
                      "y": ( 1.55,  1.65),
                      "z": (-3.0, -1.48)}},
        ],
    },

    # ── midpoint between two objects ──────────────────────────────────────────
    {
        "name":  "between_two_spheres",
        "scene": [
            {"id": "sphere-0", "type": "sphere", "pos": [-1.0, 1.6, -1.5], "color": [1,0,0], "size": 0.1},
            {"id": "sphere-1", "type": "sphere", "pos": [ 1.0, 1.6, -1.5], "color": [0,0.4,1], "size": 0.1},
        ],
        "user":  "Put a green sphere between the red and blue ones.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "g": (0.5, 1.0), "r": (0.0, 0.4), "b": (0.0, 0.4),
                      "x": (-0.05, 0.05),
                      "y": ( 1.55, 1.65),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── scale ────────────────────────────────────────────────────────────────
    {
        "name":  "scale_up_3x",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Make it three times bigger.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "size": (0.29, 0.31)}},
        ],
    },
    {
        "name":  "scale_half",
        "scene": [{"id": "box-0", "type": "box",
                   "pos": [0.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.2}],
        "user":  "Make it half the size.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0", "size": (0.09, 0.11)}},
        ],
    },
    {
        "name":  "double_its_size",
        "scene": [{"id": "sphere-1", "type": "sphere",
                   "pos": [0.13, 1.80, -1.59], "color": [0, 0, 1], "size": 0.1}],
        "user":  "Double its size.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-1", "size": (0.19, 0.21)}},
        ],
    },

    # ── multi-object: swap two object positions ───────────────────────────────
    {
        "name":  "swap_cube_and_sphere",
        "scene": [
            {"id": "box-0", "type": "box",
             "pos": [-0.5, 1.6, -1.5], "color": [0, 0.8, 0], "size": 0.1},
            {"id": "sphere-1", "type": "sphere",
             "pos": [1.0, 1.0, -2.0], "color": [0, 0.4, 1], "size": 0.1},
        ],
        "user":  "Swap the cube and the blue sphere.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "x": (0.95, 1.05),
                      "y": (0.95, 1.05),
                      "z": (-2.05, -1.95)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-1",
                      "x": (-0.55, -0.45),
                      "y": ( 1.55,  1.65),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── compound: two distinct objects in a single utterance ──────────────────
    {
        "name":  "compound_in_front_and_behind",
        "scene": [],
        "user":  "Put a green sphere in front of me and a blue cube behind me.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "g": (0.5, 1.0), "r": (0.0, 0.3), "b": (0.0, 0.3),
                      "z": (-3.0, -0.5)}},
            {"tool": "add_primitive",
             "args": {"prim_type": "box",
                      "b": (0.5, 1.0), "r": (0.0, 0.3), "g": (0.0, 0.5),
                      "z": (0.5, 3.0)}},
        ],
    },

    # ── compound: mixed add + update in one utterance ─────────────────────────
    {
        "name":  "compound_add_and_recolor",
        "scene": [{"id": "box-0", "type": "box",
                   "pos": [0.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1}],
        "user":  "Make a sphere and turn the cube red.",
        "result": [
            {"tool": "add_primitive", "args": {"prim_type": "sphere"}},
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4)}},
        ],
    },

    # ── multi-target: "all" plural pronoun ────────────────────────────────────
    {
        "name":  "make_them_all_blue",
        "scene": [
            {"id": "sphere-0", "type": "sphere",
             "pos": [-0.5, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "sphere-1", "type": "sphere",
             "pos": [0.5, 1.6, -1.5], "color": [0, 1, 0], "size": 0.1},
            {"id": "box-0", "type": "box",
             "pos": [0.0, 1.6, -1.5], "color": [1, 1, 0], "size": 0.1},
        ],
        "user":  "Make them all blue.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "b": (0.7, 1.0), "r": (0.0, 0.4), "g": (0.0, 0.4)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-1",
                      "b": (0.7, 1.0), "r": (0.0, 0.4), "g": (0.0, 0.4)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "b": (0.7, 1.0), "r": (0.0, 0.4), "g": (0.0, 0.4)}},
        ],
    },

    # ── midpoint between user and object ──────────────────────────────────────
    {
        "name":  "between_me_and_cube",
        "scene": [{"id": "box-0", "type": "box",
                   "pos": [0.0, 1.6, -2.0], "color": [0, 0.8, 0], "size": 0.1}],
        "user":  "Put a red sphere between me and the cube.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4),
                      "x": (-0.05, 0.05),
                      "y": ( 1.55, 1.65),
                      "z": (-1.05, -0.95)}},
        ],
    },

    # ── distance-specified placement ──────────────────────────────────────────
    {
        "name":  "two_meters_ahead",
        "scene": [],
        "user":  "Put a red sphere two meters in front of me.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0),
                      "x": (-0.05, 0.05),
                      "y": ( 1.55, 1.65),
                      "z": (-2.05, -1.95)}},
        ],
    },

    # ── stacking ──────────────────────────────────────────────────────────────
    {
        "name":  "stack_cube_on_sphere",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.5, 1.5, -1.5], "color": [1, 0, 0], "size": 0.2}],
        "user":  "Put a green cube on top of the sphere.",
        # render-mcp `size` is radius for spheres / half-edge for boxes.
        # Sphere top y = 1.5 + 0.2 = 1.7; a default cube (half-edge 0.1)
        # sits ON the sphere when its centre y ≈ 1.8.
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "box",
                      "g": (0.5, 1.0), "r": (0.0, 0.4), "b": (0.0, 0.4),
                      "x": (0.45, 0.55),
                      "y": (1.75, 2.0),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── three-object compound ─────────────────────────────────────────────────
    {
        "name":  "three_objects_around_me",
        "scene": [],
        "user":  "Put a red sphere in front of me, a blue cube to my right, "
                 "and a green pyramid behind me.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4),
                      "z": (-3.0, -0.3)}},
            {"tool": "add_primitive",
             "args": {"prim_type": "box",
                      "b": (0.5, 1.0), "r": (0.0, 0.3), "g": (0.0, 0.5),
                      "x": (0.3, 3.0)}},
            {"tool": "add_primitive",
             "args": {"prim_type": "pyramid",
                      "g": (0.5, 1.0), "r": (0.0, 0.4), "b": (0.0, 0.4),
                      "z": (0.3, 3.0)}},
        ],
    },

    # ── colour + place in one command ─────────────────────────────────────────
    {
        "name":  "add_red_sphere_1m_left",
        "scene": [],
        "user":  "Add a red sphere 1 meter to my left.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4),
                      "x": (-1.05, -0.95),
                      "y": ( 1.55, 1.65),
                      "z": (-0.05, 0.05)}},
        ],
    },

    # ── diagonal: combined offsets ────────────────────────────────────────────
    {
        "name":  "diagonal_up_and_left",
        "scene": [{"id": "box-0", "type": "box",
                   "pos": [0.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1}],
        "user":  "Move the cube up and to the left.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "x": (-2.0, -0.05),
                      "y": ( 1.65, 3.5),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── next to ───────────────────────────────────────────────────────────────
    {
        "name":  "next_to_cube",
        "scene": [{"id": "box-0", "type": "box",
                   "pos": [0.5, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1}],
        "user":  "Put a red sphere next to the cube.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0),
                      # Within 1 m of the cube on the horizontal plane.
                      "x": (-0.5, 1.5),
                      "y": ( 1.5,  1.7),
                      "z": (-1.7, -1.3)}},
        ],
    },

    # ── three same colour ────────────────────────────────────────────────────
    {
        "name":  "three_red_spheres_in_a_row",
        "scene": [],
        "user":  "Make three red spheres in a row in front of me.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4)}},
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4)}},
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4)}},
        ],
    },

    # ── remove all of a kind ──────────────────────────────────────────────────
    {
        "name":  "remove_all_spheres",
        "scene": [
            {"id": "sphere-0", "type": "sphere",
             "pos": [-0.5, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "sphere-1", "type": "sphere",
             "pos": [0.5, 1.6, -1.5], "color": [0, 0, 1], "size": 0.1},
            {"id": "box-0", "type": "box",
             "pos": [0.0, 1.6, -1.5], "color": [0, 0.8, 0], "size": 0.1},
        ],
        "user":  "Remove all the spheres.",
        "result": [
            {"tool": "remove_primitive", "args": {"obj_id": "sphere-0"}},
            {"tool": "remove_primitive", "args": {"obj_id": "sphere-1"}},
        ],
        "ignore_extra": False,  # the cube must NOT be removed
    },

    # ── closer to me ──────────────────────────────────────────────────────────
    {
        "name":  "bring_closer",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.6, -3.0], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Bring it closer to me.",
        # Closer to user → z grows toward 0.
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "z": (-2.99, -0.99)}},
        ],
    },

    # ── colour synonym ────────────────────────────────────────────────────────
    {
        "name":  "color_synonym_cyan",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Make it cyan.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "g": (0.5, 1.0), "b": (0.5, 1.0),
                      "r": (0.0, 0.4)}},
        ],
    },

    # ── unique reference ──────────────────────────────────────────────────────
    {
        "name":  "the_sphere_unique_ref",
        "scene": [
            {"id": "sphere-0", "type": "sphere",
             "pos": [0.5, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "box-0", "type": "box",
             "pos": [-0.5, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1},
        ],
        "user":  "Make the sphere bigger.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "size": (0.11, 1.0)}},
        ],
    },

    # ── three operations in one utterance ─────────────────────────────────────
    {
        "name":  "three_actions_compound",
        "scene": [
            {"id": "box-0", "type": "box",
             "pos": [0.0, 1.6, -1.5], "color": [0, 0.8, 0], "size": 0.1},
            {"id": "pyramid-0", "type": "pyramid",
             "pos": [-1.0, 1.6, -1.5], "color": [1, 1, 0], "size": 0.1},
        ],
        "user":  "Add a red sphere, turn the cube blue, and remove the pyramid.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "b": (0.7, 1.0), "r": (0.0, 0.4), "g": (0.0, 0.5)}},
            {"tool": "remove_primitive", "args": {"obj_id": "pyramid-0"}},
        ],
    },

    # ── named size: "huge" ────────────────────────────────────────────────────
    {
        "name":  "huge_red_sphere",
        "scene": [],
        "user":  "Make a huge red sphere.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4),
                      "size": (0.4, 1.5)}},
        ],
    },

    # ── numeric size in centimeters ──────────────────────────────────────────
    {
        "name":  "specific_size_30cm_cube",
        "scene": [],
        "user":  "Make a 30 centimeter wide red cube.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "box",
                      "r": (0.7, 1.0),
                      "size": (0.13, 0.32)}},
        ],
    },

    # ── user not at origin ────────────────────────────────────────────────────
    {
        "name":  "walked_off_origin_in_front",
        "scene": [],
        "pose":  {"is_valid": True,
                  "position": {"x": 2.0, "y": 1.6, "z": 1.5},
                  "forward": {"x": 0.0, "y": 0.0, "z": -1.0},
                  "right":   {"x": 1.0, "y": 0.0, "z": 0.0},
                  "up":      {"x": 0.0, "y": 1.0, "z": 0.0},
                  "yaw_deg": 0.0, "pitch_deg": 0.0},
        "user":  "Put a green sphere in front of me.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "g": (0.5, 1.0),
                      "x": (1.95, 2.05),
                      "z": (-0.5, 0.5)}},
        ],
    },

    # ── shape change ──────────────────────────────────────────────────────────
    {
        "name":  "shape_change_sphere_to_cube",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Turn the sphere into a cube.",
        # Either path is fine — update_primitive(prim_type=box) OR
        # remove + add(prim_type=box).  Predicate enforces "a cube
        # exists at the end" without pinning which path the LLM picked.
        "result": [],
        "predicate": _became(prim_type="box"),
    },

    # ── 1m above the cube ─────────────────────────────────────────────────────
    {
        "name":  "1m_above_the_cube",
        "scene": [{"id": "box-0", "type": "box",
                   "pos": [0.5, 1.0, -1.5], "color": [0, 0.4, 1], "size": 0.1}],
        "user":  "Put a yellow sphere 1 meter above the cube.",
        # "1m above" can mean center+1m (=2.0) or top+1m (=2.15 with
        # half-edge 0.1 + tolerance) — accept either.
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.7, 1.0),
                      # b not pinned — Nemotron occasionally leaks the cube's blue
                      "x": (0.45, 0.55),
                      "y": (1.95, 2.20),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── rolled head + diagonal user-anchored ──────────────────────────────────
    {
        "name":  "rolled_head_up_and_right_user_anchored",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "pose":  ROLLED_HEAD_POSE,
        "user":  "Move it up and to my right.",
        # Gravity-aligned: y grows (up). x grows (right, with ~10° yaw).
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "y": (1.7, 3.5)}},
        ],
    },

    # ── proximity to another object ──────────────────────────────────────────
    {
        "name":  "move_sphere_closer_to_cube",
        "scene": [
            {"id": "sphere-0", "type": "sphere",
             "pos": [-2.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "box-0", "type": "box",
             "pos": [1.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1},
        ],
        "user":  "Move the sphere closer to the cube.",
        # Closer to cube at x=1 means sphere.x grows from -2 toward 1.
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "x": (-1.95, 0.95)}},
        ],
    },

    # ── colour outside table ──────────────────────────────────────────────────
    {
        "name":  "color_brown_not_in_table",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.6, -1.5], "color": [1, 1, 1], "size": 0.1}],
        "user":  "Make it brown.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "r": (0.3, 0.8),
                      "g": (0.1, 0.5),
                      "b": (0.0, 0.4)}},
        ],
    },

    # ── ordinal disambiguation ────────────────────────────────────────────────
    {
        "name":  "ordinal_second_sphere",
        "scene": [
            {"id": "sphere-0", "type": "sphere",
             "pos": [-1.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "sphere-1", "type": "sphere",
             "pos": [ 0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "sphere-2", "type": "sphere",
             "pos": [ 1.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
        ],
        "user":  "Make the second sphere green.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-1",
                      "g": (0.5, 1.0), "r": (0.0, 0.4), "b": (0.0, 0.4)}},
        ],
    },

    # ── vague move ────────────────────────────────────────────────────────────
    {
        "name":  "vague_move_it",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Move it.",
        # Just require that the model emits SOME mutation rather than asking.
        "result": [
            {"tool": "update_primitive", "args": {"obj_id": "sphere-0"}},
        ],
    },

    # ── place where I am ──────────────────────────────────────────────────────
    {
        "name":  "place_where_i_am",
        "scene": [],
        "user":  "Make a sphere right where I'm standing.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "x": (-0.05, 0.05),
                      "y": ( 1.55, 1.65),
                      "z": (-0.55, 0.05)}},
        ],
    },

    # ── make spheres bigger ──────────────────────────────────────────────────
    {
        "name":  "make_all_spheres_bigger",
        "scene": [
            {"id": "sphere-0", "type": "sphere",
             "pos": [-0.5, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "sphere-1", "type": "sphere",
             "pos": [0.5, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1},
            {"id": "box-0", "type": "box",
             "pos": [0.0, 1.6, -1.5], "color": [0, 0.8, 0], "size": 0.1},
        ],
        "user":  "Make the spheres bigger.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "size": (0.11, 1.0)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-1", "size": (0.11, 1.0)}},
        ],
        # Plural-restricted target — the box must NOT also grow.
        "ignore_extra": False,
    },

    # ── between with distractors ──────────────────────────────────────────────
    {
        "name":  "between_red_and_blue_with_distractors",
        "scene": [
            {"id": "sphere-0", "type": "sphere",
             "pos": [-1.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "sphere-1", "type": "sphere",
             "pos": [ 1.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1},
            {"id": "sphere-2", "type": "sphere",
             "pos": [ 0.0, 1.6, -3.0], "color": [0, 0.8, 0], "size": 0.1},
            {"id": "box-0", "type": "box",
             "pos": [-2.0, 1.6, 0.0], "color": [1, 1, 0], "size": 0.1},
        ],
        "user":  "Put a green pyramid between the red sphere and the blue sphere.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "pyramid",
                      "g": (0.5, 1.0), "r": (0.0, 0.4), "b": (0.0, 0.4),
                      "x": (-0.05, 0.05),
                      "y": ( 1.55, 1.65),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── 10cm above ────────────────────────────────────────────────────────────
    {
        "name":  "small_distance_10cm",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.5, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Put a green cube 10 centimeters above the sphere.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "box",
                      "g": (0.5, 1.0), "r": (0.0, 0.4), "b": (0.0, 0.4),
                      "x": (-0.05, 0.05),
                      "y": ( 1.55, 1.65),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── shape + colour change ─────────────────────────────────────────────────
    {
        "name":  "shape_and_color_change",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Make the sphere a blue cube.",
        # Either path is fine (update with prim_type+colour, or remove+add).
        # Predicate enforces "a cube exists" AND "blue channel ≥ 0.5
        # somewhere in the mutations" without pinning which call carries
        # which facet.
        "result": [],
        "predicate": _became(prim_type="box", b_min=0.5),
    },

    # ── pitched up: above me is gravity-aligned ───────────────────────────────
    {
        "name":  "pitched_up_above_me_gravity_aligned",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.5, 1.0, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "pose":  {"is_valid": True,
                  "position": {"x": 0.0, "y": 1.6, "z": 0.0},
                  "forward": {"x": 0.0,  "y": 0.5,   "z": -0.866},
                  "right":   {"x": 1.0,  "y": 0.0,   "z": 0.0},
                  "up":      {"x": 0.0,  "y": 0.866, "z": 0.5},
                  "yaw_deg": 0.0, "pitch_deg": 30.0},
        "user":  "Move it above my head.",
        # User-anchored: x and z snap to user's column; only y grows.
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "x": (-0.05, 0.05),
                      "z": (-0.05, 0.05)}},
        ],
    },

    # ── place at my feet ──────────────────────────────────────────────────────
    {
        "name":  "place_at_my_feet",
        "scene": [],
        "user":  "Put a red sphere at my feet.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0),
                      "y": (-0.05, 0.5)}},  # near the floor
        ],
    },

    # ── ambiguous red sphere — pick one ───────────────────────────────────────
    {
        "name":  "ambiguous_red_sphere_pick_one",
        "scene": [
            {"id": "sphere-0", "type": "sphere",
             "pos": [0.5, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "sphere-1", "type": "sphere",
             "pos": [-0.5, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
        ],
        "user":  "Move the red sphere to the left.",
        # Either sphere is a valid pick.  Empty result asserts
        # "≥1 mutating call happened" — we don't pin which sphere.
        "result": [],
    },

    # ── pure remove ───────────────────────────────────────────────────────────
    {
        "name":  "remove_the_cube",
        "scene": [
            {"id": "box-0", "type": "box",
             "pos": [0.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1},
            {"id": "sphere-0", "type": "sphere",
             "pos": [0.5, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
        ],
        "user":  "Get rid of the cube.",
        "result": [
            {"tool": "remove_primitive", "args": {"obj_id": "box-0"}},
        ],
    },

    # ── stack three cubes ─────────────────────────────────────────────────────
    {
        "name":  "stack_three_cubes",
        "scene": [],
        "user":  "Stack three blue cubes.",
        # Three blue cubes at any base height, but stacked vertically:
        # x/z must coincide and y values must be distinct.  Predicate
        # below enforces the relative geometry the matcher can't express.
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "box", "b": (0.5, 1.0)}},
            {"tool": "add_primitive",
             "args": {"prim_type": "box", "b": (0.5, 1.0)}},
            {"tool": "add_primitive",
             "args": {"prim_type": "box", "b": (0.5, 1.0)}},
        ],
        "predicate": _stacked_vertically,
    },

    # ── way to the left, no number ────────────────────────────────────────────
    {
        "name":  "way_to_the_left_no_number",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Move it way to the left.",
        # No specific number → at least 0.5 m left.
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "x": (-3.0, -0.5)}},
        ],
    },

    # ── pronoun "it" follows the LAST agent reply, not LAST modified ─────────
    # Trap case: scene has TWO objects, the older one was modified more
    # recently in tool history but the agent's last reply confirmed the
    # newer one.  "it" must resolve to the newer (the just-added blue
    # sphere), NOT the yellow sphere whose y was just changed.
    {
        "name":  "pronoun_it_follows_last_reply",
        "scene": [
            {"id": "sphere-0", "type": "sphere",
             "pos": [0.0, 0.6, -1.5], "color": [1, 1, 0], "size": 0.1},   # yellow, just moved down
            {"id": "sphere-1", "type": "sphere",
             "pos": [0.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1}, # blue, just added
        ],
        "history": [
            ("Make a yellow sphere.",         "Added a yellow sphere."),
            ("Move the sphere down 1 metre.", "Moved the sphere down by one metre."),
            ("Make a blue sphere.",           "Added a blue sphere."),
        ],
        # Bare "right 1 m" (no "my") isolates pronoun resolution from
        # anchor selection.  "It" should resolve to the blue sphere
        # (subject of the last reply), which is at y=1.6 — guarding
        # against the model picking the yellow one at y=0.6.
        "user":  "Move it right by 1 metre.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-1",
                      "x": (0.95, 1.05),
                      "y": (1.55, 1.65),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── undo: "put it back" restores prior coords from [Recent moves] ────────
    {
        "name":  "undo_put_it_back",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [1.0, 1.6, -1.5], "color": [1, 1, 0], "size": 0.1}],
        "history": [
            ("Make a yellow sphere.",         "Added a yellow sphere."),
            ("Move it 1 metre to the right.", "Moved the sphere 1 metre to your right."),
        ],
        "recent_moves": [
            ("sphere-0", (0.0, 1.6, -1.5), (1.0, 1.6, -1.5)),
        ],
        "user":  "Put it back.",
        # Should restore to the previous position (0, 1.6, -1.5).
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "x": (-0.05, 0.05),
                      "y": ( 1.55, 1.65),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── undo: "undo that" — same intent, different phrasing ──────────────────
    {
        "name":  "undo_undo_that",
        "scene": [{"id": "box-0", "type": "box",
                   "pos": [0.0, 2.6, -1.5], "color": [0, 0.4, 1], "size": 0.1}],
        "history": [
            ("Make a blue cube.",         "Added a blue cube."),
            ("Lift it 1 metre over me.",  "Raised the cube above you."),
        ],
        "recent_moves": [
            ("box-0", (0.0, 1.6, -1.5), (0.0, 2.6, -1.5)),
        ],
        "user":  "Undo that.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "x": (-0.05, 0.05),
                      "y": ( 1.55, 1.65),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── spatial disambiguation: "the X on the right" picks rightmost x ───────
    {
        "name":  "remove_sphere_on_the_right_picks_rightmost",
        "scene": [
            {"id": "sphere-0", "type": "sphere",
             "pos": [ 0.0, 1.6, -1.5], "color": [1, 1, 0], "size": 0.1},
            {"id": "sphere-1", "type": "sphere",
             "pos": [-0.48, 1.4, -0.8], "color": [1, 1, 0], "size": 0.1},
        ],
        "user":  "Remove the sphere on the right.",
        "result": [
            {"tool": "remove_primitive", "args": {"obj_id": "sphere-0"}},
        ],
    },

    # ── plural pronoun "them" → every recently-named object ────────────────
    {
        "name":  "them_after_two_spheres_moves_both",
        "scene": [
            {"id": "sphere-0", "type": "sphere",
             "pos": [0.0, 1.6, -1.5], "color": [1, 1, 0], "size": 0.1},
            {"id": "sphere-2", "type": "sphere",
             "pos": [0.0, 1.5, -1.5], "color": [0, 0, 1], "size": 0.1},
        ],
        "history": [
            ("Make a yellow sphere.",                    "Added a yellow sphere."),
            ("Put a blue sphere under the yellow sphere.","Added a blue sphere under the yellow sphere."),
        ],
        "user":  "Move them one metre to the right.",
        # Both spheres should land near x=1, y unchanged, z unchanged.
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "x": ( 0.95, 1.05),
                      "y": ( 1.55, 1.65),
                      "z": (-1.55, -1.45)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-2",
                      "x": ( 0.95, 1.05),
                      "y": ( 1.45, 1.55),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── batch move: every math call must be paired with update_primitive ────
    {
        "name":  "move_everything_further_away_writes_each_object",
        "scene": [
            {"id": "box-0", "type": "box",
             "pos": [-0.96, 1.23, -0.08], "color": [1, 0, 0], "size": 0.1},
            {"id": "sphere-0", "type": "sphere",
             "pos": [ 1.00, 1.60, -1.44], "color": [1, 1, 0], "size": 0.1},
            {"id": "sphere-2", "type": "sphere",
             "pos": [ 0.14, 1.60, -0.92], "color": [0, 0, 1], "size": 0.1},
        ],
        "user":  "Move everything 1 meter further away.",
        # All three should end up 1 m further from the user — z more
        # negative by ~1 at canonical pose.  y / x unchanged.
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "x": (-1.05, -0.85),
                      "y": ( 1.18, 1.28),
                      "z": (-1.13, -1.03)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "x": ( 0.95, 1.05),
                      "y": ( 1.55, 1.65),
                      "z": (-2.50, -2.40)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-2",
                      "x": ( 0.10, 0.20),
                      "y": ( 1.55, 1.65),
                      "z": (-1.97, -1.87)}},
        ],
    },

    # ── origin must come from SCENE block, not [Recent moves] ───────────────
    {
        "name":  "move_named_object_uses_scene_origin_not_recent_moves",
        "scene": [
            {"id": "box-0", "type": "box",
             "pos": [-0.96, 1.23, -0.08], "color": [1, 0, 0], "size": 0.1},
            {"id": "sphere-0", "type": "sphere",
             "pos": [ 1.00, 1.60, -1.44], "color": [1, 1, 0], "size": 0.1},
            {"id": "sphere-2", "type": "sphere",
             "pos": [ 0.00, 1.50, -1.50], "color": [0, 0, 1], "size": 0.1},
        ],
        "recent_moves": [
            ("sphere-0", (0.0, 1.6, -1.5), (1.0, 1.6, -1.44)),
        ],
        "user":  "Move the blue sphere to the left.",
        # Sphere-2 should end up shifted by ~1m along the user's left
        # vector starting from its OWN position (0, 1.5, -1.5).  At
        # canonical pose head.right=(1,0,0) so left = (-1,0,0); the
        # result lands near (-1, 1.5, -1.5).
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-2",
                      "x": (-1.05, -0.95),
                      "y": ( 1.45, 1.55),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── swap with "in" instead of "and" (STT mishearing) ─────────────────────
    # STT often returns "swap A in B" for "swap A and B".  Both phrasings
    # must trigger the swap rule (two update_primitive calls), not a
    # midpoint add.
    {
        "name":  "swap_in_means_swap_and",
        "scene": [
            {"id": "box-0", "type": "box",
             "pos": [-0.96, 1.23, -0.08], "color": [1, 0, 0], "size": 0.1},
            {"id": "sphere-0", "type": "sphere",
             "pos": [ 0.0, 1.6, -1.5], "color": [1, 1, 0], "size": 0.1},
        ],
        "user":  "Swap the sphere in the cube.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "x": (-0.05, 0.05),
                      "y": ( 1.55, 1.65),
                      "z": (-1.55, -1.45)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "x": (-1.0, -0.92),
                      "y": ( 1.20, 1.27),
                      "z": (-0.13, -0.03)}},
        ],
    },

    # ── containment is NOT swap ──────────────────────────────────────────────
    # "Put X in Y" is containment: X moves to Y's centre, Y stays put.
    # Pairs with swap_in_means_swap_and to catch a model that collapses
    # every "X in Y" into a swap.
    {
        "name":  "put_sphere_in_cube_is_containment",
        "scene": [
            {"id": "box-0", "type": "box",
             "pos": [0.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.3},
            {"id": "sphere-0", "type": "sphere",
             "pos": [1.0, 1.6, -1.5], "color": [1, 0, 0],   "size": 0.1},
        ],
        "user":  "Put the sphere in the cube.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "x": (-0.05, 0.05),
                      "y": ( 1.55, 1.65),
                      "z": (-1.55, -1.45)}},
        ],
        # Cube must NOT move — that's what distinguishes this from swap.
        "ignore_extra": False,
    },

    # ── spatial disambiguation on the LEFT side (mirror of …rightmost) ───────
    # Same scene shape as the rightmost case but the cue is "leftmost".
    {
        "name":  "remove_pyramid_on_the_left_picks_leftmost",
        "scene": [
            {"id": "pyramid-0", "type": "pyramid",
             "pos": [-1.30, 1.50, -2.20], "color": [0.6, 0, 1], "size": 0.1},
            {"id": "pyramid-1", "type": "pyramid",
             "pos": [ 0.40, 1.50, -2.20], "color": [0.6, 0, 1], "size": 0.1},
        ],
        "user":  "Remove the pyramid on the left.",
        "result": [
            {"tool": "remove_primitive", "args": {"obj_id": "pyramid-0"}},
        ],
    },

    # ── existing subject → update_primitive, never add_primitive ────────────
    # Mirrors a live-demo bug: prior turns mentioned several objects
    # (user added pyramid-0, then swapped box and sphere); user then
    # says "Put it above the blue sphere" expecting the existing
    # pyramid to be raised.  Model has historically picked add_primitive
    # ("clone the recently-named object") instead of update_primitive on
    # the existing pyramid.  Pass-or-fail probe — captures the bug so
    # we can iterate; the prompt-side rule lives in the
    # "EXISTING ID → update_primitive" section.
    {
        "name":  "pronoun_after_swap_uses_update_not_add",
        "scene": [
            {"id": "box-0",     "type": "box",
             "pos": [0.5, 0.6, -1.5], "color": [1, 1, 0], "size": 0.1},
            {"id": "sphere-0",  "type": "sphere",
             "pos": [0.5, 0.7, -1.5], "color": [0, 0.4, 1], "size": 0.1},
            {"id": "pyramid-0", "type": "pyramid",
             "pos": [0.0, 1.6, 0.5], "color": [0, 0.8, 0], "size": 0.1},
        ],
        "history": [
            ("Add a green pyramid above me and a bit behind.",
             "Added a green pyramid."),
            ("Switch the box and the sphere.",
             "Swapped the box and the sphere."),
        ],
        "user":  "Put it above the blue sphere.",
        # Subject of the placement ("it") is the existing pyramid-0;
        # the rule REQUIRES update_primitive on pyramid-0, not
        # add_primitive of any kind.  Position lands ~above sphere-0
        # at (0.5, 0.7, -1.5); we accept any y >= 0.75 to be lenient
        # on the "above" offset the model picks.
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "pyramid-0",
                      "x": (0.45, 0.55),
                      "y": (0.75, 2.5),
                      "z": (-1.55, -1.45)}},
        ],
    },

    # ── companion probe: named existing subject → update, never add ─────────
    # Same rule, but the subject is named explicitly ("the cube") so
    # pronoun resolution doesn't enter the picture.  ignore_extra=False
    # is the teeth: an add_primitive alongside the update is also a fail.
    {
        "name":  "move_existing_cube_above_me_uses_update_not_add",
        "scene": [{"id": "box-0", "type": "box",
                   "pos": [0.0, 0.6, -1.5], "color": [0, 0.4, 1], "size": 0.1}],
        "user":  "Move the cube above where I am.",
        # box-0 should end up near the user's column (x≈0, z≈0) with y
        # raised above eye level (≥1.55).  No add_primitive allowed.
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "x": (-0.05, 0.05),
                      "y": ( 1.55, 3.5),
                      "z": (-0.05, 0.05)}},
        ],
        "ignore_extra": False,
    },

    # ── three sequential moves on one object via "up and down 3 times" ───────
    # Exercises the multi-update-in-one-utterance pattern on a single object.
    # Model often emits partial-update calls (just y= …) for vertical
    # bounces, so the matcher only constrains obj_id + y range; x and z
    # are left unspecified so partial updates pass.
    {
        "name":  "bounce_sphere_up_and_down_3x",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1}],
        "user":  "Move the sphere up and down three times.",
        # 3 ups + 3 downs = 6 mutating calls.  Up moves end above start
        # (y>1.6), down moves end at or below start (y<=1.6).
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "y": (1.7, 3.0)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "y": (1.7, 3.0)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "y": (1.7, 3.0)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "y": (0.5, 1.61)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "y": (0.5, 1.61)}},
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0", "y": (0.5, 1.61)}},
        ],
    },
]


def _format_scene(scene: list[dict]) -> str:
    if not scene:
        return "SCENE OBJECTS: (empty)"
    lines = ["SCENE OBJECTS:"]
    for o in scene:
        x, y, z = o["pos"]
        r, g, b = o["color"]
        lines.append(
            f"  {o['id']} ({o['type']})  "
            f"pos=({x:.2f}, {y:.2f}, {z:.2f})  "
            f"color=(r={r:.2f} g={g:.2f} b={b:.2f})  "
            f"size={o['size']:.3f}m"
        )
    return "\n".join(lines)


def _format_pose(pose: dict) -> str:
    if not pose.get("is_valid"):
        return "HEAD POSE: unavailable"
    p, fv, rv, uv = pose["position"], pose["forward"], pose["right"], pose["up"]

    def _off(vec, d):
        return (f"({p['x']+vec['x']*d:.2f}, "
                f"{p['y']+vec['y']*d:.2f}, "
                f"{p['z']+vec['z']*d:.2f})")

    return (
        "HEAD POSE:\n"
        f"  position : ({p['x']:.2f}, {p['y']:.2f}, {p['z']:.2f})\n"
        f"  forward  : ({fv['x']:.3f}, {fv['y']:.3f}, {fv['z']:.3f})  ← 'ahead/forward'\n"
        f"  right    : ({rv['x']:.3f}, {rv['y']:.3f}, {rv['z']:.3f})  ← 'right'\n"
        f"  up       : ({uv['x']:.3f}, {uv['y']:.3f}, {uv['z']:.3f})  ← 'up'\n"
        f"  yaw={pose.get('yaw_deg',0):.1f}°  pitch={pose.get('pitch_deg',0):.1f}°\n"
        "SPATIAL SHORTCUTS (pre-computed — use directly, no tool call needed):\n"
        f"  1.5m ahead of you     : {_off(fv,  1.5)}\n"
        f"  1m to your right      : {_off(rv,  1.0)}\n"
        f"  1m to your left       : {_off(rv, -1.0)}\n"
        f"  0.5m above eye level  : {_off(uv,  0.5)}\n"
        f"  1m behind you         : {_off(fv, -1.0)}\n"
        "  For other distances: new_pos = obj.pos + direction_vec × distance (per component)"
    )


async def _discover_tools() -> list[dict]:
    tools = []
    for url in (RENDER_MCP, OXR_MCP, VLM_MCP, VIDEO_MCP, VEC_MCP):
        try:
            async with McpClient(url) as c:
                for t in await c.list_tools():
                    if t.name in WORKER_MANAGED:
                        continue
                    schema = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
                    tools.append({"type": "function", "function": {
                        "name": t.name,
                        "description": (t.description or "").strip(),
                        "parameters": schema,
                    }})
        except Exception as exc:
            print(f"WARN: discovery failed for {url}: {exc}", file=sys.stderr)
    return tools


def _format_recent_moves(moves: list[tuple] | None) -> str:
    """Render the same `[Recent moves]` block the worker injects.  Each
    `moves` entry is (obj_id, (px, py, pz), (nx, ny, nz)).
    """
    if not moves:
        return ""
    lines = ["[Recent moves] (most recent last — prev → new)"]
    for obj_id, prev, new in moves:
        lines.append(
            f"  {obj_id}: ({prev[0]:.2f}, {prev[1]:.2f}, {prev[2]:.2f}) → "
            f"({new[0]:.2f}, {new[1]:.2f}, {new[2]:.2f})"
        )
    return "\n".join(lines)


def _format_recent_conversation(history: list[tuple[str, str]] | None) -> str:
    """Render the same `[Recent conversation]` block the worker injects.
    Each entry is (prior_user_text, prior_agent_reply).
    """
    if not history:
        return ""
    lines = ["[Recent conversation]"]
    for u, a in history:
        lines.append(f"  User: {u}")
        lines.append(f"  Agent: {a}")
    return "\n".join(lines)


def _build_messages(system_prompt: str, scene: list[dict], pose: dict, user: str,
                    history: list[tuple[str, str]] | None = None,
                    recent_moves: list[tuple] | None = None) -> list[dict]:
    """Build the worker-equivalent chat messages.  Prior turns go into
    a ``[Recent conversation]`` block inside the single user-role
    context message — injecting them as ``role=assistant`` biases
    Nemotron toward text-only replies and away from tool calls."""
    context_parts = [_format_scene(scene), _format_pose(pose)]
    moves_block = _format_recent_moves(recent_moves)
    if moves_block:
        context_parts.append(moves_block)
    conv_block = _format_recent_conversation(history)
    if conv_block:
        context_parts.append(conv_block)
    context = "\n".join(context_parts)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": (
            "[Pre-fetched context — do not call get_scene_state or "
            "get_head_pose unless you need to refresh after changes]\n"
            f"{context}\n\n[Request]\n{user}"
        )},
    ]


def _local_position_relative(args: dict, pose: dict) -> dict:
    """Mirror oxr-mcp.position_relative — gravity-aligned (yaw is honoured;
    pitch and roll are stripped). Up is world +Y."""
    f, r = pose["forward"], pose["right"]
    p = pose["position"]
    fwd = float(args.get("forward", 0.0))
    rgt = float(args.get("right",   0.0))
    up_ = float(args.get("up",      0.0))
    ox = float(args.get("origin_x", p["x"]))
    oy = float(args.get("origin_y", p["y"]))
    oz = float(args.get("origin_z", p["z"]))

    fx, fz = f["x"], f["z"]
    mag = math.sqrt(fx*fx + fz*fz)
    if mag < 1e-6:
        rx0, rz0 = r["x"], r["z"]
        mag2 = math.sqrt(rx0*rx0 + rz0*rz0)
        if mag2 < 1e-6:
            fx, fz = 0.0, -1.0
        else:
            rx0, rz0 = rx0 / mag2, rz0 / mag2
            fx, fz = rz0, -rx0
    else:
        fx, fz = fx / mag, fz / mag
    rx, rz = -fz, fx

    return {
        "x": round(ox + fx*fwd + rx*rgt, 3),
        "y": round(oy + up_,             3),
        "z": round(oz + fz*fwd + rz*rgt, 3),
    }


def _local_position_ahead(args: dict, pose: dict) -> dict:
    f, p = pose["forward"], pose["position"]
    d = float(args.get("distance", 1.5))
    return {
        "x": round(p["x"] + f["x"]*d, 3),
        "y": round(p["y"] + f["y"]*d, 3),
        "z": round(p["z"] + f["z"]*d, 3),
    }


def _ground_basis(pose: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Mirror oxr-mcp._ground_basis."""
    f, r = pose["forward"], pose["right"]
    fx, fz = f["x"], f["z"]
    mag = math.sqrt(fx * fx + fz * fz)
    if mag < 1e-6:
        rx0, rz0 = r["x"], r["z"]
        mag2 = math.sqrt(rx0 * rx0 + rz0 * rz0)
        if mag2 < 1e-6:
            fx, fz = 0.0, -1.0
        else:
            rx0, rz0 = rx0 / mag2, rz0 / mag2
            fx, fz = rz0, -rx0
    else:
        fx, fz = fx / mag, fz / mag
    return (fx, fz), (-fz, fx)


def _local_place_user_relative(args: dict, pose: dict) -> dict:
    direction = args.get("direction", "front")
    distance = float(args.get("distance", 1.5))
    if distance < 0:
        return {"error": "distance must be non-negative"}
    p = pose["position"]
    (fx, fz), (rx, rz) = _ground_basis(pose)
    dx = dy = dz = 0.0
    if direction == "front":
        dx, dz = fx * distance, fz * distance
    elif direction == "back":
        dx, dz = -fx * distance, -fz * distance
    elif direction == "right":
        dx, dz = rx * distance, rz * distance
    elif direction == "left":
        dx, dz = -rx * distance, -rz * distance
    elif direction == "above":
        dy = distance
    elif direction == "below":
        dy = -distance
    return {
        "x": round(p["x"] + dx, 3),
        "y": round(p["y"] + dy, 3),
        "z": round(p["z"] + dz, 3),
    }


def _local_world_offset(args: dict, _pose: dict) -> dict:
    """Mirror vec-mcp.world_offset — origin + (dx, dy, dz)."""
    ox = float(args.get("origin_x", 0.0))
    oy = float(args.get("origin_y", 0.0))
    oz = float(args.get("origin_z", 0.0))
    dx = float(args.get("dx", 0.0))
    dy = float(args.get("dy", 0.0))
    dz = float(args.get("dz", 0.0))
    return {"x": round(ox + dx, 3), "y": round(oy + dy, 3), "z": round(oz + dz, 3)}


def _local_along_direction(args: dict, _pose: dict) -> dict:
    """Mirror vec-mcp.along_direction — origin moved `distance` toward target."""
    ox = float(args.get("origin_x", 0.0))
    oy = float(args.get("origin_y", 0.0))
    oz = float(args.get("origin_z", 0.0))
    tx = float(args.get("target_x", 0.0))
    ty = float(args.get("target_y", 0.0))
    tz = float(args.get("target_z", 0.0))
    d  = float(args.get("distance", 0.5))
    vx, vy, vz = tx - ox, ty - oy, tz - oz
    mag = math.sqrt(vx*vx + vy*vy + vz*vz)
    if mag < 1e-9:
        return {"error": "origin and target coincide"}
    return {
        "x": round(ox + vx * d / mag, 3),
        "y": round(oy + vy * d / mag, 3),
        "z": round(oz + vz * d / mag, 3),
    }


def _local_scale_value(args: dict, _pose: dict) -> dict:
    """Mirror vec-mcp.scale_value — current * factor."""
    cur = float(args.get("current", 0.0))
    fac = float(args.get("factor",  1.0))
    return {"value": round(cur * fac, 3)}


def _local_place_inside_by_id(args: dict, _pose: dict) -> dict:
    """Mirror oxr-mcp.place_inside_by_id — container coords echoed back
    alongside the movee's id so the result feeds straight into
    update_primitive."""
    for field in ("movee_id", "container_x", "container_y", "container_z"):
        if args.get(field) is None:
            return {"error": f"missing {field}"}
    return {
        "obj_id": args["movee_id"],
        "x":      round(float(args["container_x"]), 3),
        "y":      round(float(args["container_y"]), 3),
        "z":      round(float(args["container_z"]), 3),
    }


def _local_between_anchors(args: dict, _pose: dict) -> dict:
    """Mirror vec-mcp.between_anchors — component-wise midpoint of A and B."""
    a_x, a_y, a_z = (float(args.get("a_x", 0.0)),
                     float(args.get("a_y", 0.0)),
                     float(args.get("a_z", 0.0)))
    b_x, b_y, b_z = (float(args.get("b_x", 0.0)),
                     float(args.get("b_y", 0.0)),
                     float(args.get("b_z", 0.0)))
    return {
        "x": round((a_x + b_x) / 2.0, 3),
        "y": round((a_y + b_y) / 2.0, 3),
        "z": round((a_z + b_z) / 2.0, 3),
    }


def _local_displace_objects(args: dict, pose: dict) -> dict:
    """Mirror oxr-mcp.displace_objects — same user-frame delta applied
    to every (id, x, y, z) entry; returns {items: [...]}."""
    for field in ("object_ids", "current_xs", "current_ys", "current_zs"):
        if args.get(field) is None:
            return {"error": f"missing {field}"}
    ids = list(args["object_ids"])
    xs  = list(args["current_xs"])
    ys  = list(args["current_ys"])
    zs  = list(args["current_zs"])
    n = len(ids)
    if not (len(xs) == n and len(ys) == n and len(zs) == n):
        return {"error": "object_ids / current_xs / current_ys / current_zs "
                         "must all be the same length"}
    if n == 0:
        return {"items": []}
    right   = float(args.get("right",   0.0))
    up_     = float(args.get("up",      0.0))
    forward = float(args.get("forward", 0.0))
    (fx, fz), (rx, rz) = _ground_basis(pose)
    items = []
    for i in range(n):
        cx, cy, cz = float(xs[i]), float(ys[i]), float(zs[i])
        items.append({
            "obj_id": ids[i],
            "x": round(cx + fx * forward + rx * right, 3),
            "y": round(cy + up_,                       3),
            "z": round(cz + fz * forward + rz * right, 3),
        })
    return {"items": items}


def _local_displace_object(args: dict, pose: dict) -> dict:
    """Mirror oxr-mcp.displace_object — current + user-frame delta."""
    for field in ("current_x", "current_y", "current_z"):
        if args.get(field) is None:
            return {"error": f"missing {field}"}
    cx = float(args["current_x"])
    cy = float(args["current_y"])
    cz = float(args["current_z"])
    right   = float(args.get("right",   0.0))
    up_     = float(args.get("up",      0.0))
    forward = float(args.get("forward", 0.0))
    (fx, fz), (rx, rz) = _ground_basis(pose)
    return {
        "x": round(cx + fx * forward + rx * right, 3),
        "y": round(cy + up_,                       3),
        "z": round(cz + fz * forward + rz * right, 3),
    }


def _local_place_object_relative(args: dict, pose: dict) -> dict:
    direction = args.get("direction", "front")
    distance = float(args.get("distance", 0.3))
    if distance < 0:
        return {"error": "distance must be non-negative"}
    ox = float(args.get("origin_x", 0.0))
    oy = float(args.get("origin_y", 0.0))
    oz = float(args.get("origin_z", 0.0))
    (fx, fz), (rx, rz) = _ground_basis(pose)
    dx = dy = dz = 0.0
    if direction == "front":
        dx, dz = -fx * distance, -fz * distance
    elif direction == "back":
        dx, dz = fx * distance, fz * distance
    elif direction == "right":
        dx, dz = rx * distance, rz * distance
    elif direction == "left":
        dx, dz = -rx * distance, -rz * distance
    elif direction == "next_to":
        dx, dz = rx * distance, rz * distance
    elif direction == "above":
        dy = distance
    elif direction == "below":
        dy = -distance
    return {
        "x": round(ox + dx, 3),
        "y": round(oy + dy, 3),
        "z": round(oz + dz, 3),
    }


_ADD_COUNTER: dict[str, int] = {}


def _reset_exec_state() -> None:
    _ADD_COUNTER.clear()


# Per-case scratch state read by the local tool mocks in _exec_tool.
# Reset before every rollout via _reset_exec_state / _set_*.
_FIXTURE_SCENE: list[dict] = []
_CASE_HISTORY: list[tuple[str, str]] = []
_CASE_MOVES: list[tuple] = []


def _set_fixture_scene(scene: list[dict]) -> None:
    _FIXTURE_SCENE.clear()
    _FIXTURE_SCENE.extend(scene)


def _set_case_history(history: list[tuple[str, str]] | None) -> None:
    _CASE_HISTORY.clear()
    if history:
        _CASE_HISTORY.extend(history)


def _set_case_moves(moves: list[tuple] | None) -> None:
    _CASE_MOVES.clear()
    if moves:
        _CASE_MOVES.extend(moves)


def _fixture_scene_as_render() -> dict:
    """Echo the case's fixture scene back in the same shape render-mcp's
    get_scene_state returns. Prevents Nemotron retry-loops where it asks
    for the scene, sees nothing, and asks again."""
    return {"objects": [
        {"id":       o["id"],
         "type":     o["type"],
         "position": {"x": o["pos"][0], "y": o["pos"][1], "z": o["pos"][2]},
         "color":    {"r": o["color"][0], "g": o["color"][1], "b": o["color"][2]},
         "size":     o.get("size", 0.1)}
        for o in _FIXTURE_SCENE
    ]}


async def _exec_tool(name: str, args_json: str, pose: dict) -> dict:
    """Execute a tool call.  oxr-mcp tools run locally against the
    case's fixture pose so rollouts are deterministic.  add_primitive
    returns a fresh per-rollout id (otherwise the model spawns the
    same object N times waiting to "see" it); update / remove return
    ok.  Unknown tools return a sentinel."""
    args = json.loads(args_json) if isinstance(args_json, str) else (args_json or {})
    if name == "position_relative":
        return _local_position_relative(args, pose)
    if name == "position_ahead":
        return _local_position_ahead(args, pose)
    if name == "place_user_relative":
        return _local_place_user_relative(args, pose)
    if name == "place_object_relative":
        return _local_place_object_relative(args, pose)
    if name == "place_inside_by_id":
        return _local_place_inside_by_id(args, pose)
    if name == "displace_object":
        return _local_displace_object(args, pose)
    if name == "displace_objects":
        return _local_displace_objects(args, pose)
    if name == "between_anchors":
        return _local_between_anchors(args, pose)
    if name == "world_offset":
        return _local_world_offset(args, pose)
    if name == "along_direction":
        return _local_along_direction(args, pose)
    if name == "scale_value":
        return _local_scale_value(args, pose)
    if name == "get_head_pose":
        return pose
    if name == "add_primitive":
        prim = args.get("prim_type", "sphere")
        n = _ADD_COUNTER.get(prim, -1) + 1
        _ADD_COUNTER[prim] = n
        return {"id": f"{prim}-{n}", "ok": True}
    if name == "update_primitive":
        return {"ok": True}
    if name == "remove_primitive":
        return {"ok": True}
    if name == "get_scene_state":
        return _fixture_scene_as_render()
    return {"_eval_skipped": True, "reason": f"{name} not in safe-exec list"}


async def _run_one(http: httpx.AsyncClient, system_prompt: str,
                   tools: list[dict], scene: list[dict], pose: dict,
                   user: str, *, thinking: bool = False,
                   max_steps: int = 1) -> dict:
    """Run up to ``max_steps`` LLM iterations against the agent LLM,
    mocking tool execution between turns via ``_exec_tool``.  Returns
    ``{latency_s, tool_calls, content, reasoning}``: ``tool_calls`` is
    every tool call emitted across all turns (in order), ``content`` /
    ``reasoning`` are from the final turn.
    """
    _reset_exec_state()
    _set_fixture_scene(scene)
    messages = _build_messages(system_prompt, scene, pose, user,
                               _CASE_HISTORY, _CASE_MOVES)
    all_calls: list[dict] = []
    last_msg: dict = {}
    t_total = 0.0

    for _step in range(max_steps):
        body = {
            "model": AGENT_MODEL,
            "messages": messages,
            "tools": tools,
            "max_tokens": 2048 if thinking else 1024,
            "temperature": 0.0,
            "chat_template_kwargs": {
                "enable_thinking": thinking,
                **({"thinking_budget": 1024} if thinking else {}),
            },
        }
        t0 = time.time()
        headers = {"Authorization": f"Bearer {AGENT_KEY}"} if AGENT_KEY else None
        # Retry on transient 5xx / network errors; non-5xx still raise.
        for attempt in range(3):
            try:
                r = await http.post(AGENT_LLM, json=body, timeout=180.0, headers=headers)
                if r.status_code >= 500 and attempt < 2:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt >= 2:
                    raise
                await asyncio.sleep(2.0 * (attempt + 1))
        t_total += time.time() - t0
        msg = r.json()["choices"][0]["message"]
        last_msg = msg
        tcs = msg.get("tool_calls") or []
        if not tcs:
            break
        # A turn can emit multiple parallel tool calls (e.g. compound
        # utterances), so extend rather than append.
        all_calls.extend(tcs)
        if _step + 1 >= max_steps:
            break
        new_msgs: list[dict] = [{"role": "assistant", "content": "", "tool_calls": tcs}]
        for tc in tcs:
            fn = tc["function"]
            result = await _exec_tool(fn["name"], fn["arguments"], pose)
            new_msgs.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result, default=str)})
        messages = messages + new_msgs

    return {"latency_s":  round(t_total, 2),
            "tool_calls": all_calls,
            "content":    (last_msg.get("content") or "").strip(),
            "reasoning":  (last_msg.get("reasoning_content") or "").strip()}


# update_primitive arg -> (scene-object-field, optional index).  Used to
# resolve "absent arg means kept original value" so partial updates
# (e.g. ``{x: -1.0}`` for "move left 1m") are checked against the
# effective resulting position, not just the bytes the LLM emitted.
_SCENE_ARG_LOOKUP = {
    "x": ("pos", 0), "y": ("pos", 1), "z": ("pos", 2),
    "r": ("color", 0), "g": ("color", 1), "b": ("color", 2),
    "size": ("size", None), "prim_type": ("type", None),
}


def _resolve_arg(obj_id: str, key: str, scene: list[dict]):
    field = _SCENE_ARG_LOOKUP.get(key)
    if not field:
        return None
    obj = next((o for o in scene if o.get("id") == obj_id), None)
    if obj is None:
        return None
    src, idx = field
    val = obj.get(src)
    if idx is None:
        return val
    return val[idx] if val is not None and idx < len(val) else None


def _match_call(call: dict, expect: dict, scene: list[dict] | None = None) -> tuple[bool, str]:
    fn = call["function"]
    if fn["name"] != expect["tool"]:
        return False, f"tool={fn['name']} want={expect['tool']}"
    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
    fails = []
    for k, want in expect.get("args", {}).items():
        got = args.get(k)
        if got is None and fn["name"] == "update_primitive" and scene:
            obj_id = args.get("obj_id")
            if obj_id:
                got = _resolve_arg(obj_id, k, scene)
        if got is None:
            fails.append(f"{k}=missing"); continue
        if isinstance(want, tuple):
            lo, hi = want
            # Some models emit numeric args as strings; coerce before compare.
            if isinstance(got, str):
                try:
                    got = float(got)
                except ValueError:
                    fails.append(f"{k}={got!r} not numeric, want [{lo},{hi}]")
                    continue
            if not (lo <= got <= hi):
                fails.append(f"{k}={got} not in [{lo},{hi}]")
        else:
            if got != want:
                fails.append(f"{k}={got!r} want={want!r}")
    return (not fails), ("ok" if not fails else "; ".join(fails))


_MUTATING_TOOLS = frozenset({"add_primitive", "update_primitive", "remove_primitive"})


def _check(actual: dict, case: dict) -> tuple[bool, str]:
    """Match ``case['result']`` against the mutating tool calls
    (add/update/remove_primitive) emitted during the rollout.  Order-
    independent; helper/math calls are ignored.  ``ignore_extra``
    (default True) allows extra mutations beyond the expectation.

    Empty ``result`` is the "any path is fine" mode: the case still
    requires at least one mutating call to have happened (otherwise a
    silent no-op would pass).
    """
    tcs = actual["tool_calls"]
    wanted = list(case.get("result") or [])
    muts = [tc for tc in tcs if tc["function"]["name"] in _MUTATING_TOOLS]
    if not wanted and not muts:
        names = [tc["function"]["name"] for tc in tcs]
        return False, f"no mutating calls: {names}"
    scene = case.get("scene") or []
    unmatched_actuals = list(muts)
    unmatched_expected: list[dict] = []
    for exp in wanted:
        for idx, ac in enumerate(unmatched_actuals):
            ok, _ = _match_call(ac, exp, scene)
            if ok:
                unmatched_actuals.pop(idx); break
        else:
            unmatched_expected.append(exp)
    if unmatched_expected:
        missing = "; ".join(
            f"{e['tool']}({e.get('args',{})})" for e in unmatched_expected
        )
        actual_summary = [
            f"{tc['function']['name']}({tc['function']['arguments']})"
            for tc in muts
        ]
        return False, f"unmatched result: {missing} | actual mutations: {actual_summary}"
    if not case.get("ignore_extra", True) and unmatched_actuals:
        extras = [tc["function"]["name"] for tc in unmatched_actuals]
        return False, f"extra mutating calls: {extras}"
    predicate = case.get("predicate")
    if predicate is not None:
        ok, msg = predicate(muts)
        if not ok:
            return False, f"predicate failed: {msg}"
    return True, f"matched {len(wanted)} mutation(s)"


# max LLM iterations per turn (mirrors processors.py _MAX_LOOP).
_MAX_STEPS = 10


# Reserved-prompt-vocabulary sets used by check #4 in
# _check_prompt_eval_overlap (see that docstring and eval/README.md).
_EVAL_VOCAB_COLORS = frozenset({
    "red", "green", "blue", "cyan", "brown", "yellow",
})
_EVAL_VOCAB_SHAPES = frozenset({
    "sphere", "spheres", "cube", "cubes", "box", "boxes",
    "pyramid", "pyramids",
})

# Worked-example section start markers (case-insensitive).  A section
# runs from the marker line through the first blank line; triple-backtick
# fences are also captured as blocks (everything between the fences).
_EXAMPLE_START_RE = re.compile(
    r"^\s*(?:"
    r"WORKED\s+EXAMPLE\b|WORKED\s+ANTI-?EXAMPLE\b|"
    r"Examples?:|"
    r"iter\s+\d+\s*:|"
    r"tool_call\s+\d+\s*:"
    r")",
    re.IGNORECASE,
)


def _extract_example_blocks(sp: str) -> list[tuple[int, str]]:
    """Slice the system prompt into worked-example sections.

    Returns ``[(start_line_1_indexed, block_text), …]``.  A section is
    either everything between a pair of triple-backtick fences, or
    everything from a marker line (``WORKED EXAMPLE``, ``Example:``,
    ``iter N:``, ``tool_call N:``) through the first following blank
    line.
    """
    blocks: list[tuple[int, str]] = []
    lines = sp.splitlines()
    in_fence = False
    fence_start = 0
    fence_buf: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            if in_fence:
                blocks.append((fence_start, "\n".join(fence_buf)))
                in_fence = False
                fence_buf = []
            else:
                in_fence = True
                fence_start = i + 1
            i += 1
            continue
        if in_fence:
            fence_buf.append(line)
            i += 1
            continue
        if _EXAMPLE_START_RE.match(line):
            start = i + 1
            buf = [line]
            i += 1
            while i < len(lines) and lines[i].strip():
                buf.append(lines[i])
                i += 1
            blocks.append((start, "\n".join(buf)))
            continue
        i += 1
    if in_fence and fence_buf:
        blocks.append((fence_start, "\n".join(fence_buf)))
    return blocks


def _case_fixture_vocab(c: dict) -> tuple[set[str], set[str]]:
    """Eval-vocab colour/shape words actually present in this case's
    fixture (user utterance, history dialogue, scene type tags, ids).
    Used to attribute reserved-vocab violations to specific cases."""
    parts: list[str] = [c.get("user") or ""]
    for pair in c.get("history") or []:
        parts.extend(pair)
    for o in c.get("scene") or []:
        if t := o.get("type"):
            parts.append(t)
        if oid := o.get("id"):
            parts.append(oid)
    blob = " ".join(parts).lower()
    colors = {w for w in _EVAL_VOCAB_COLORS if re.search(rf"\b{w}\b", blob)}
    shapes = {w for w in _EVAL_VOCAB_SHAPES if re.search(rf"\b{w}\b", blob)}
    return colors, shapes


def _check_prompt_eval_overlap(
    system_prompt: str, cases: list[dict]
) -> tuple[set[str], list[str]]:
    """Detect overlap between prompt worked-examples and eval case
    fixtures.  An overlap turns a generalization probe into a
    memorization check (see AGENTS.md "Prompt-driven samples").

    Four checks run, each across every case:
      1. Verbatim user utterance (≥12 chars) appearing in the prompt.
      2. Concrete scene coordinates rendered like ``(x.xx, y.yy, z.zz)``
         appearing in the prompt.
      3. ``recent_moves`` coords appearing in the prompt.
      4. Reserved-prompt-vocabulary: worked-example sections of
         system.txt must not use any colour/shape word from the
         eval-case vocabulary (``_EVAL_VOCAB_COLORS`` /
         ``_EVAL_VOCAB_SHAPES``).  Worked-example sections are
         triple-backtick blocks and any block starting with
         ``WORKED EXAMPLE`` / ``Example:`` / ``iter N:`` /
         ``tool_call N:``.  Rule narration outside those blocks
         is unrestricted — the colour table, anchor-routing rules,
         etc. may still mention ``red sphere`` generically.

    Returns ``(overlapping_case_names, issue_lines)``.  The set is the
    distinct cases that overlap (caller uses the count for the score
    caveat); the list is per-issue detail strings.  Both are empty
    when no overlaps.
    """
    sp = system_prompt
    issues: list[str] = []
    overlapping: set[str] = set()
    for c in cases:
        name = c.get("name", "<unnamed>")
        before = len(issues)
        # 1. Verbatim user utterance (case-insensitive substring) appearing
        #    in the prompt.  Short utterances <12 chars are skipped to
        #    avoid noise like "Move it." matching every example.
        u = (c.get("user") or "").strip().rstrip(".!?")
        if u and len(u) >= 12 and u.lower() in sp.lower():
            issues.append(f"  {name}: user utterance {u!r} appears verbatim in system.txt")
        # 2. Concrete scene coordinates (rendered like "(0.50, 1.60, -1.50)")
        #    appearing in the prompt.
        for o in c.get("scene") or []:
            x, y, z = o["pos"]
            coord = f"({x:.2f}, {y:.2f}, {z:.2f})"
            if coord in sp:
                issues.append(
                    f"  {name}: scene object {o['id']!r} coords {coord} "
                    f"appear verbatim in system.txt"
                )
                break
        # 3. recent_moves coords landing in the prompt.
        for entry in c.get("recent_moves") or []:
            _obj, prev, new = entry
            for triple in (prev, new):
                coord = f"({triple[0]:.2f}, {triple[1]:.2f}, {triple[2]:.2f})"
                if coord in sp:
                    issues.append(
                        f"  {name}: recent_moves coords {coord} appear "
                        f"verbatim in system.txt"
                    )
                    break
        if len(issues) > before:
            overlapping.add(name)

    # 4. Reserved-prompt-vocabulary.  Built second so it's reported as a
    #    block after the verbatim checks, but the case names it
    #    attributes still feed the same ``overlapping`` set used by the
    #    score-line suffix.
    case_index_colors: dict[str, list[str]] = {w: [] for w in _EVAL_VOCAB_COLORS}
    case_index_shapes: dict[str, list[str]] = {w: [] for w in _EVAL_VOCAB_SHAPES}
    for c in cases:
        cname = c.get("name", "<unnamed>")
        cc, cs = _case_fixture_vocab(c)
        for w in cc:
            case_index_colors[w].append(cname)
        for w in cs:
            case_index_shapes[w].append(cname)

    color_alt = "|".join(sorted(_EVAL_VOCAB_COLORS))
    shape_alt = "|".join(sorted(_EVAL_VOCAB_SHAPES))
    pair_re   = re.compile(rf"\b({color_alt})\s+({shape_alt})\b", re.IGNORECASE)
    color_re  = re.compile(rf"\b({color_alt})\b", re.IGNORECASE)
    shape_re  = re.compile(rf"\b({shape_alt})\b", re.IGNORECASE)

    for start_line, block_text in _extract_example_blocks(sp):
        seen_words: set[str] = set()
        # Adjacent "<color> <shape>" — the canonical violation shape.
        for m in pair_re.finditer(block_text):
            color = m.group(1).lower()
            shape = m.group(2).lower()
            offenders = sorted(set(case_index_colors.get(color, []))
                               | set(case_index_shapes.get(shape, [])))
            for case_name in offenders:
                issues.append(
                    f"  {case_name}: example block at line {start_line} "
                    f"uses '{color} {shape}' which also appears in case fixture"
                )
                overlapping.add(case_name)
            seen_words.add(color)
            seen_words.add(shape)
        # Lone colour or shape words not already counted in a pair.
        for m in color_re.finditer(block_text):
            w = m.group(1).lower()
            if w in seen_words:
                continue
            seen_words.add(w)
            for case_name in case_index_colors.get(w, []):
                issues.append(
                    f"  {case_name}: example block at line {start_line} "
                    f"uses '{w}' which also appears in case fixture"
                )
                overlapping.add(case_name)
        for m in shape_re.finditer(block_text):
            w = m.group(1).lower()
            if w in seen_words:
                continue
            seen_words.add(w)
            for case_name in case_index_shapes.get(w, []):
                issues.append(
                    f"  {case_name}: example block at line {start_line} "
                    f"uses '{w}' which also appears in case fixture"
                )
                overlapping.add(case_name)

    return overlapping, issues


async def main() -> None:
    global AGENT_LLM, AGENT_MODEL, AGENT_KEY

    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", help="ad-hoc query (skips case suite)")
    p.add_argument("--prompt", type=Path, default=SYS_PROMPT)
    p.add_argument("--only",
                   help="comma-separated list of case names to run; all other "
                        "cases are skipped.  Useful for fast iteration on a "
                        "single failing cluster.  Mutually exclusive with the "
                        "positional `query` arg.")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--verbose",  action="store_true")
    p.add_argument("--strict-overlap", action="store_true",
                   help="fail (rc=2) if any case fixture overlaps with the "
                        "system prompt's worked examples — turn on in CI to "
                        "guard against silent train-on-test drift")
    # agent-LLM endpoint overrides — default to whatever the worker yaml
    # points at (local vLLM on 8107 in dev); set to point at
    # build.nvidia.com etc. when scoring against a hosted model.
    p.add_argument("--agent-llm", default=os.environ.get("AGENT_LLM_URL", AGENT_LLM),
                   help="full /v1/chat/completions URL for the agent LLM")
    p.add_argument("--agent-model", default=os.environ.get("AGENT_LLM_MODEL", "llm"),
                   help="model name sent in the chat-completion request body")
    p.add_argument("--agent-api-key",
                   default=(os.environ.get("NVIDIA_API_KEY", "")
                            or os.environ.get("NGC_API_KEY", "")),
                   help="Bearer token for the agent LLM "
                        "(env NVIDIA_API_KEY or NGC_API_KEY)")
    args = p.parse_args()

    AGENT_LLM   = args.agent_llm
    AGENT_MODEL = args.agent_model
    AGENT_KEY   = args.agent_api_key

    if args.only and args.query:
        p.error("--only and a positional query are mutually exclusive")

    # Honour a sibling .only file as a shorthand for --only (see
    # eval/README.md "Watcher" section for the file format).
    only_file = _HERE / ".only"
    if not args.only and not args.query and only_file.exists():
        names: list[str] = []
        for raw in only_file.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            for tok in line.split(","):
                tok = tok.strip()
                if tok:
                    names.append(tok)
        if names:
            args.only = ",".join(names)
            print(f"FILTER: {only_file.name} → {names}")

    system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    print(f"PROMPT: {args.prompt}  ({len(system_prompt)} chars)")
    is_remote = not AGENT_LLM.lower().startswith(("http://localhost", "http://127.", "http://0.0.0.0"))
    print(f"AGENT-LLM: {AGENT_LLM}  model={AGENT_MODEL}"
          + ("  [remote, auth=on]" if is_remote and AGENT_KEY else "")
          + ("  [remote, auth=MISSING]" if is_remote and not AGENT_KEY else "")
          + ("  [local]" if not is_remote else ""))

    tools = await _discover_tools()
    tool_names = [t["function"]["name"] for t in tools]
    print(f"TOOLS:  {tool_names}")

    pose = DEFAULT_POSE
    if args.verbose:
        print("POSE:", json.dumps(pose))

    async with httpx.AsyncClient() as http:
        if args.query:
            r = await _run_one(http, system_prompt, tools, [], pose,
                               args.query, thinking=args.thinking)
            print(json.dumps(r, indent=2))
            return

        cases = list(CASES)
        if args.only:
            requested = [n.strip() for n in args.only.split(",") if n.strip()]
            valid = {c["name"] for c in cases}
            unknown = [n for n in requested if n not in valid]
            if unknown:
                p.error(f"--only: unknown case name(s) {unknown}. "
                        f"Valid names: {sorted(valid)}")
            cases = [c for c in cases if c["name"] in requested]

        # Audit: prompt worked-examples must not duplicate case fixtures.
        # Warns at startup so overlaps don't turn the score into a
        # memorization check.  Run before any LLM calls.
        overlap_names, overlap_issues = _check_prompt_eval_overlap(
            system_prompt, cases
        )
        if overlap_issues:
            print("\n⚠ PROMPT/EVAL OVERLAP DETECTED — these cases share specifics with "
                  "system.txt and may be measuring memorization rather than "
                  "generalization.  Fix by changing the prompt's worked example "
                  "(see AGENTS.md \"Prompt-driven samples\"):")
            for line in overlap_issues:
                print(line)
            print()
            if args.strict_overlap:
                print(f"--strict-overlap set: aborting with rc=2 "
                      f"({len(overlap_names)} overlapping case(s))",
                      file=sys.stderr)
                sys.exit(2)
        else:
            print("PROMPT/EVAL OVERLAP: clean (no verbatim utterances, coords, or "
                  "reserved-vocab leaks)")

        results = []
        for c in cases:
            scene_c = c["scene"]
            pose_c  = c.get("pose", pose)
            _set_case_history(c.get("history"))
            _set_case_moves(c.get("recent_moves"))
            try:
                r = await _run_one(http, system_prompt, tools, scene_c, pose_c,
                                   c["user"], thinking=args.thinking,
                                   max_steps=_MAX_STEPS)
            except Exception as exc:
                r = {"latency_s": 0.0, "tool_calls": [], "content": "",
                     "reasoning": ""}
                ok, why = False, f"network error: {type(exc).__name__}: {exc}"
            else:
                ok, why = _check(r, c)
            mark = "✓" if ok else "✗"
            print(f"{mark} {c['name']:32s} {r['latency_s']:5.1f}s  {why}")
            for i, tc in enumerate(r["tool_calls"]):
                fn = tc["function"]
                print(f"    [{i}] {fn['name']}({fn['arguments']})")
            results.append((c["name"], ok))

        passed = sum(1 for _, ok in results if ok)
        total  = len(results)
        score_line = f"\n{passed}/{total} passed"
        if overlap_names:
            score_line += (
                f" ({len(overlap_names)}/{total} too close to prompts — "
                f"may be memorization, not generalization)"
            )
        print(score_line)
        sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
