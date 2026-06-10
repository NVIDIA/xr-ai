#!/usr/bin/env -S uv run --quiet --with httpx --with fastmcp --with pyyaml --script
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "fastmcp>=0.4", "pyyaml"]
# ///
"""
agent-llm eval harness for xr-render-demo. Talks to the running stack
with a real LLM; render-mcp tools are fake-succeeded so the harness never
mutates the live LOVR scene, and look_at_current_frame is mocked in vision
cases so perception is deterministic.

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

# Mirrors the worker's _SUPERSEDED_PERCEPTION_TOOLS: MCP tools superseded by the
# brain-executed look_at_current_frame and withheld from the model. Keep in sync.
SUPERSEDED_PERCEPTION = {"get_latest_frame", "ask_image"}

# Brain-executed live-frame perception tool — not served by any MCP, so tool
# discovery misses it. The eval must advertise the same look_at_current_frame
# surface the live worker does, so keep this in sync with processors.py
# `_PERCEPTION_TOOL_DEF` (no shared import: that would drag the worker's deps
# into this stdlib-light harness and cross the eval↔worker layering line).
_PERCEPTION_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "look_at_current_frame",
        "description": (
            "Look at the user's LIVE camera feed right now and answer a "
            "question about the real world — what they are holding, pointing "
            "at, or looking at; a real-world colour, shape, text, or object. "
            "Turns the camera on automatically and inspects the current frame. "
            "Use this whenever the answer cannot be known from the XR scene "
            "state alone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The specific question to answer about the live camera "
                        "frame, e.g. 'What colour is the object the user is "
                        "holding?'"
                    ),
                },
            },
            "required": ["question"],
        },
    },
}

# Synthetic participant id injected into every rollout's context, mirroring
# the worker's "Participant: {pid}" line. look_at_current_frame resolves the
# active participant worker-side, so it is context only (not a tool argument).
EVAL_PID = "eval-user"

# Robustness sweep (opt-in): re-score selected cases under semantically-
# irrelevant context perturbations to tell a robust result from one balanced on
# a decision boundary. Off by default; see eval/README.md.
EVAL_REF_US_A = 1749760000000000
EVAL_REF_US_B = 1783000123456789

# Read by _build_messages; the robustness loop swaps it per variant and restores
# it afterward, so its resting value is the clean default.
_ROBUSTNESS_VARIANT: dict = {"order": "clean", "ref_us": None}

# Mirror of the worker's needs_thinking scaffold, prepended to the system
# prompt when a variant turns thinking on.
_THINK_SCAFFOLD = (
    "Use your private <think> block to work through these steps. "
    "NEVER output these steps as your response — your only text output "
    "to the user is ONE SHORT sentence AFTER all tool calls are done.\n"
    "\n"
    "THINK STEP 1 — RESOLVE: Which object? "
    "Pronouns ('it', 'that') = most recently added/modified object. "
    "Named ('the blue sphere') = match by color/type in scene.\n"
    "\n"
    "THINK STEP 2 — LOCATE: Copy the exact x, y, z of the target object "
    "and the head pose right/forward/up vectors from the context.\n"
    "\n"
    "THINK STEP 3 — COMPUTE: Calculate new coordinates with explicit arithmetic. "
    "User-relative move: new = old + head_vec × distance (per component). "
    "Near object: new = obj.pos ± world_offset. "
    "Midpoint: new = (A + B) / 2 per component. "
    "Write out each component: x=…, y=…, z=…\n"
    "\n"
    "THINK STEP 4 — EXECUTE: call the tool with the computed values, "
    "then reply with ONE short sentence to the user.\n\n"
)

# clean is the default context; the rest perturb it in ways a robust prompt
# must be invariant to.
_ROBUSTNESS_VARIANTS = [
    {"tag": "clean",  "order": "clean",  "ref_us": None,          "thinking": False, "scaffold": False},
    {"tag": "refB",   "order": "clean",  "ref_us": EVAL_REF_US_B, "thinking": False, "scaffold": False},
    {"tag": "order2", "order": "worker", "ref_us": EVAL_REF_US_A, "thinking": False, "scaffold": False},
    {"tag": "think",  "order": "worker", "ref_us": EVAL_REF_US_A, "thinking": True,  "scaffold": True},
]

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


def _stacked_exactly(n: int):
    """Predicate factory for a stack case that must emit EXACTLY ``n``
    add_primitive calls (catching over-stacking the default ``ignore_extra``
    matcher would hide) AND stack them vertically."""
    def _pred(muts: list[dict]) -> tuple[bool, str]:
        adds = [tc for tc in muts if tc["function"]["name"] == "add_primitive"]
        if len(adds) != n:
            return False, f"need exactly {n} add_primitive calls, got {len(adds)}"
        return _stacked_vertically(muts)
    return _pred


def _containment_sequence(container: tuple[float, float, float], *, tol: float = 0.1):
    """Ordered-call predicate factory for containment: assert the rollout
    calls place_inside_by_id and then IMMEDIATELY update_primitive on the
    placed object at the container centre. Reads the full ordered tool-call
    list (helper calls included), so a model that skips the place_inside_by_id
    compute step fails even if its final coordinates happen to land."""
    cx, cy, cz = container

    def _pred(tcs: list[dict]) -> tuple[bool, str]:
        names = [tc["function"]["name"] for tc in tcs]
        if "place_inside_by_id" not in names:
            return False, f"no place_inside_by_id call; calls={names}"
        i = names.index("place_inside_by_id")
        pi = tcs[i]["function"]["arguments"]
        pi = json.loads(pi) if isinstance(pi, str) else pi
        movee = pi.get("movee_id")
        if i + 1 >= len(names) or names[i + 1] != "update_primitive":
            return False, (f"place_inside_by_id not immediately followed by "
                           f"update_primitive; calls={names}")
        up = tcs[i + 1]["function"]["arguments"]
        up = json.loads(up) if isinstance(up, str) else up
        oid = up.get("obj_id") or up.get("object_id")
        if movee and oid and oid != movee:
            return False, (f"update_primitive targets {oid!r}, not the placed "
                           f"object {movee!r}")
        for k, want in (("x", cx), ("y", cy), ("z", cz)):
            v = up.get(k)
            if v is None:
                return False, f"update_primitive missing {k}"
            if abs(float(v) - want) > tol:
                return False, f"update {k}={v} not ≈ container {k}={want}"
        return True, (f"place_inside_by_id → update_primitive at "
                      f"container centre ({cx}, {cy}, {cz})")
    return _pred


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
    # ── CONTEXT METADATA is not a coordinate ──────────────────────────────────
    # A Reference-time line is injected whose leading digits ("2.5…") read like
    # a tempting coordinate/offset. Coordinates must come from the cube's
    # position + head pose; a model that mistakes the µs timestamp for an
    # offset lands far outside the box. Proves the timestamp is treated as
    # bookkeeping, not spatial input.
    {
        "name":  "context_metadata_not_a_coordinate",
        "scene": [{"id": "box-0", "type": "box",
                   "pos": [0.2, 1.45, -1.8], "color": [0, 0.4, 1], "size": 0.1}],
        "ref_us": 2500000000000000,
        "user":  "Move the cube to my right by half a metre.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "x": (0.65, 0.75),
                      "y": (1.40, 1.50),
                      "z": (-1.85, -1.75)}},
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

    # ── stack EXACTLY the requested count ─────────────────────────────────────
    # "Stack two" must emit exactly two adds: ignore_extra=False plus a
    # predicate pinning the count catches the over-stacking the >=2 predicate
    # on stack_three_cubes lets through.
    {
        "name":  "stack_two_yellow_pyramids",
        "scene": [],
        "user":  "Stack two yellow pyramids.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "pyramid",
                      "r": (0.7, 1.0), "g": (0.7, 1.0), "b": (0.0, 0.4)}},
            {"tool": "add_primitive",
             "args": {"prim_type": "pyramid",
                      "r": (0.7, 1.0), "g": (0.7, 1.0), "b": (0.0, 0.4)}},
        ],
        "predicate": _stacked_exactly(2),
        "ignore_extra": False,
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

    # ── containment is exactly TWO ordered calls ──────────────────────────────
    # place_inside_by_id COMPUTES the destination; update_primitive MOVES the
    # object there. ordered_calls enforces the SEQUENCE, so a model that jumps
    # straight to update_primitive (skipping the compute step) fails even if
    # its coordinates happen to land at the container centre.
    {
        "name":  "containment_place_inside_then_update",
        "scene": [
            {"id": "box-0", "type": "box",
             "pos": [-0.6, 1.4, -2.0], "color": [0, 0.8, 0], "size": 0.3},
            {"id": "pyramid-0", "type": "pyramid",
             "pos": [1.2, 1.5, -0.8], "color": [1, 1, 0], "size": 0.1},
        ],
        "user":  "Put the pyramid in the cube.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "pyramid-0",
                      "x": (-0.65, -0.55),
                      "y": ( 1.35, 1.45),
                      "z": (-2.05, -1.95)}},
        ],
        "ordered_calls": _containment_sequence((-0.6, 1.4, -2.0)),
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

    # ══ COLOUR FIDELITY ═══════════════════════════════════════════════════════
    # category="color" cases catch two failures that matter in the live demo:
    # gross colour-family confusion (white when asked for magenta) and
    # built-vs-said dishonesty (narrating one colour while painting another).
    # The r/g/b boxes are intentionally wide — each admits any reasonable
    # rendering of the requested colour and fails only a clearly wrong one;
    # in-family hue drift is not policed. `reply_colors` is the honesty
    # allow-set: the spoken reply may not name a colour outside it, nor (via
    # _built_rgb) one inconsistent with what was actually built.
    {
        "name":     "color_make_magenta_sphere",
        "category": "color",
        "scene":    [],
        "user":     "Make a magenta sphere.",
        # high R, LOW G, high B — excludes white (G high), red (B low),
        # blue (R low). This is the white↔magenta trap, magenta direction.
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.7, 1.0)}},
        ],
        "reply_colors": {"magenta", "pink"},
    },
    {
        "name":     "color_make_white_sphere",
        "category": "color",
        "scene":    [],
        "user":     "Make a white sphere.",
        # all channels high — excludes any saturated hue (magenta has G=0).
        # This is the white↔magenta trap, white direction.
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.8, 1.0), "g": (0.8, 1.0), "b": (0.8, 1.0)}},
        ],
        "reply_colors": {"white"},
    },
    {
        "name":     "color_make_sphere_default_white",
        "category": "color",
        "scene":    [],
        # No colour specified → the prompt defaults to WHITE. Beyond the RGB
        # default, this guards say==build on the unspecified-default path (the
        # live "built white / said orange" shape): reply_colors={"white"} lets
        # the reply name no colour at all OR name white, but a non-white colour
        # word — cross-checked against the built rgb — FAILS.
        "user":     "Make a sphere.",
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.8, 1.0), "g": (0.8, 1.0), "b": (0.8, 1.0)}},
        ],
        "reply_colors": {"white"},
    },
    {
        "name":     "color_make_coloured_sphere_self_consistent",
        "category": "color",
        "scene":    [],
        # User-side underspecified hue: "coloured" names no specific colour.
        # BOTH honest behaviors are accepted — the model may ask which colour,
        # OR pick one and build it — and only a say!=built colour mismatch
        # fails. T1 accepts a clarifying question OR a self-consistent build.
        # T2 ("just pick one") additionally requires a self-consistent coloured
        # sphere to exist (built now or carried from T1). No fixed target: the
        # allowed reply colours are derived from whatever the model built.
        "turns": [
            {"user": "Make a coloured sphere.",
             "accept_clarify_or_consistent": True},
            {"user": "Just pick one.",
             "accept_clarify_or_consistent": True,
             "require_coloured_result": True},
        ],
    },
    {
        "name":     "color_make_red_sphere",
        "category": "color",
        "scene":    [],
        "user":     "Make a red sphere.",
        # high R, low G, low B — excludes green/blue (R low), yellow
        # (G high), white (G/B high), magenta (B high).
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.6, 1.0), "g": (0.0, 0.45), "b": (0.0, 0.45)}},
        ],
        "reply_colors": {"red"},
    },
    {
        "name":     "color_make_green_cube",
        "category": "color",
        "scene":    [],
        "user":     "Create a green cube.",
        # low R, high G, low B — excludes red/yellow (R high), cyan/blue
        # (B high), white.
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "box",
                      "r": (0.0, 0.45), "g": (0.55, 1.0), "b": (0.0, 0.45)}},
        ],
        "reply_colors": {"green"},
    },
    {
        "name":     "color_make_blue_pyramid",
        "category": "color",
        "scene":    [],
        "user":     "Add a blue pyramid.",
        # low R, high B — excludes red/orange/magenta (R high), green
        # (B low), cyan (G maxed), white.
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "pyramid",
                      "r": (0.0, 0.45), "g": (0.0, 0.65), "b": (0.55, 1.0)}},
        ],
        "reply_colors": {"blue"},
    },
    {
        "name":     "color_make_yellow_sphere",
        "category": "color",
        "scene":    [],
        "user":     "Make a yellow sphere.",
        # high R, high G, low B — excludes white (B high), orange (G mid),
        # green (R low), red (G low).
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.6, 1.0), "g": (0.6, 1.0), "b": (0.0, 0.4)}},
        ],
        "reply_colors": {"yellow"},
    },
    {
        "name":     "color_make_orange_cube",
        "category": "color",
        "scene":    [],
        "user":     "Create an orange cube.",
        # high R, MID G, low B — excludes red (G low), yellow (G high),
        # white (B high), magenta (B high).
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "box",
                      "r": (0.7, 1.0), "g": (0.25, 0.7), "b": (0.0, 0.4)}},
        ],
        "reply_colors": {"orange"},
    },
    {
        "name":     "color_make_purple_pyramid",
        "category": "color",
        "scene":    [],
        "user":     "Add a purple pyramid.",
        # mid-high R, low G, high B — excludes blue (R low), magenta/pink
        # (R near 1), white (G high).
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "pyramid",
                      "r": (0.25, 0.8), "g": (0.0, 0.45), "b": (0.5, 1.0)}},
        ],
        "reply_colors": {"purple", "violet"},
    },
    {
        "name":     "color_make_cyan_sphere",
        "category": "color",
        "scene":    [],
        "user":     "Make a cyan sphere.",
        # low R, high G, high B — excludes blue (G low), green (B low),
        # white (R high). Admits a teal-leaning cyan.
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.0, 0.45), "g": (0.5, 1.0), "b": (0.5, 1.0)}},
        ],
        "reply_colors": {"cyan", "teal", "turquoise"},
    },
    {
        "name":     "color_make_black_cube",
        "category": "color",
        "scene":    [],
        "user":     "Add a black cube.",
        # all channels low — excludes any bright colour.
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "box",
                      "r": (0.0, 0.2), "g": (0.0, 0.2), "b": (0.0, 0.2)}},
        ],
        "reply_colors": {"black"},
    },
    {
        "name":     "color_make_gray_sphere",
        "category": "color",
        "scene":    [],
        "user":     "Make a gray sphere.",
        # all channels mid — excludes any saturated hue (which maxes one
        # channel and zeroes another). Admits dark grey through silver.
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.25, 0.75), "g": (0.25, 0.75), "b": (0.25, 0.75)}},
        ],
        "reply_colors": {"gray", "grey", "silver"},
    },
    # Recolour a WHITE object to magenta — the dedicated built-white/
    # said-magenta lie catcher. update_primitive is a partial update, so a
    # call that omits channels resolves them against the white scene object:
    # built reads white, which fails BOTH the magenta box (G stays 1) and
    # the build-vs-said honesty check if the reply still says "magenta".
    {
        "name":     "color_recolor_to_magenta",
        "category": "color",
        "scene":    [{"id": "box-0", "type": "box",
                      "pos": [0.0, 1.6, -1.5], "color": [1, 1, 1], "size": 0.1}],
        "user":     "Make it magenta.",
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "box-0",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.7, 1.0)}},
        ],
        "reply_colors": {"magenta", "pink"},
    },

    # ── real-world visual queries (camera routing) ────────────────────────────
    # These cases catch the agent answering a real-world question ("what am I
    # holding?") from a VIRTUAL scene primitive instead of the live camera.
    # Each case mocks the vision path — look_at_current_frame returns the
    # case's ``vision`` description as {"answer": "<KNOWN colour>"} — then
    # asserts (a) the FIRST tool call is look_at_current_frame, (b) no scene
    # mutation happens, and (c) the FINAL reply repeats the camera's colour.
    # ``reply_forbids`` (distractor cases) additionally fails any reply that
    # leaks the virtual primitive's colour. ``first_call`` triggers the
    # vision-routing branch in _check.
    {
        "name":        "vision_what_am_i_holding",
        "category":    "vision",
        "scene":       [],
        "user":        "What am I holding right now?",
        "vision":      "The user is holding a lavender ceramic vase.",
        "first_call":  ["look_at_current_frame"],
        "reply_names": ["lavender"],
    },
    {
        "name":        "vision_color_of_held_object",
        "category":    "vision",
        "scene":       [],
        "user":        "What colour is the object I'm holding?",
        "vision":      "The object in the user's hand is a turquoise rubber ball.",
        "first_call":  ["look_at_current_frame"],
        "reply_names": ["turquoise"],
    },
    {
        "name":        "vision_what_am_i_looking_at",
        "category":    "vision",
        "scene":       [],
        "user":        "What am I looking at?",
        "vision":      "The user is looking at a maroon leather armchair.",
        "first_call":  ["look_at_current_frame"],
        "reply_names": ["maroon"],
    },
    {
        "name":        "vision_whats_in_front_of_me",
        "category":    "vision",
        "scene":       [],
        "user":        "What is that thing in front of me?",
        "vision":      "Directly in front of the user is an olive backpack.",
        "first_call":  ["look_at_current_frame"],
        "reply_names": ["olive"],
    },
    # Distractor: a VIRTUAL orange sphere sits in the scene, but the user asks
    # about a REAL mug. The camera says teal; a correct turn reports teal and
    # never the sphere's orange. This is the proof that scene state is ignored.
    {
        "name":         "vision_real_mug_vs_virtual_sphere",
        "category":     "vision",
        "scene":        [{"id": "sphere-0", "type": "sphere",
                          "pos": [0.42, 1.53, -1.83], "color": [1, 0.5, 0],
                          "size": 0.1}],
        "user":         "What colour is my coffee mug?",
        "vision":       "The user is holding a teal coffee mug.",
        "first_call":   ["look_at_current_frame"],
        "reply_names":  ["teal"],
        "reply_forbids": ["orange"],
        "reply_forbids_always": True,
    },
    # Distractor #2: a VIRTUAL navy cone in the scene; the user asks about a
    # REAL shirt. Camera says gold; the reply must say gold, not navy.
    {
        "name":         "vision_real_shirt_vs_virtual_cone",
        "category":     "vision",
        "scene":        [{"id": "cone-0", "type": "cone",
                          "pos": [-0.63, 1.47, -2.05], "color": [0, 0, 0.5],
                          "size": 0.1}],
        "user":         "What colour is my shirt?",
        "vision":       "The user is wearing a gold button-up shirt.",
        "first_call":   ["look_at_current_frame"],
        "reply_names":  ["gold"],
        "reply_forbids": ["navy"],
        "reply_forbids_always": True,
    },
    # Multi-turn re-query: the real world can change between asks. Turn 1 the
    # camera sees colour B1; turn 2 (a fresh "and now?" ask) it sees B2. The
    # ``vision`` SEQUENCE feeds B1 then B2 to successive look_at_current_frame
    # calls. The bug: on turn 2 the model sees its own turn-1 reply in [Recent
    # conversation] and repeats B1 instead of looking again. A correct turn 2
    # routes to the camera AGAIN (first_call) and reports the FRESH B2, never
    # the stale B1 (reply_forbids). Each turn is checked independently; the
    # case passes only if both pass.
    {
        "name":      "vision_requery_returns_fresh_colour",
        "category":  "vision",
        "scene":     [],
        "vision":    ["The user is holding a vermilion water bottle.",
                      "The user is now holding a chartreuse water bottle."],
        "turns": [
            {"user":        "What am I holding?",
             "first_call":  ["look_at_current_frame"],
             "reply_names": ["vermilion"]},
            {"user":         "And what about now?",
             "first_call":   ["look_at_current_frame"],
             "reply_names":  ["chartreuse"],
             "reply_forbids": ["vermilion"]},
        ],
    },
    # Multi-turn APPLY: the model must USE the camera-perceived colour in a
    # build, not just report it, and RE-QUERY on every ask (T3 reuses neither
    # T2's colour nor its frame). Each turn asserts look_at_current_frame first,
    # the correct-colour mutation on the carried sphere, and build-vs-said
    # honesty via ``reply_colors``. Mock-only colours (no train-on-test).
    {
        "name":     "vision_apply_perceived_colour",
        "category": "vision",
        "scene":    [],
        "vision":   ["The user is holding a green apple.",
                     "The user's shirt is blue.",
                     "The user's shirt is red."],
        "turns": [
            {"user":       "Make a sphere the colour of the thing I'm holding.",
             "first_call": ["look_at_current_frame"],
             "result": [
                 {"tool": "add_primitive",
                  "args": {"prim_type": "sphere",
                           "r": (0.0, 0.5), "g": (0.5, 1.0), "b": (0.0, 0.5)}},
             ],
             "reply_colors": {"green"}},
            {"user":       "Now change its colour to match my shirt.",
             "first_call": ["look_at_current_frame"],
             # Recolour must target the carried T1 sphere (sphere-0, pinned by
             # the exactly-one-build guard), so updating a distractor fails.
             "result": [
                 {"tool": "update_primitive",
                  "args": {"obj_id": "sphere-0",
                           "r": (0.0, 0.5), "g": (0.0, 0.5), "b": (0.5, 1.0)}},
             ],
             "reply_colors": {"blue"}},
            {"user":       "Actually, match my shirt colour again.",
             "first_call": ["look_at_current_frame"],
             "result": [
                 {"tool": "update_primitive",
                  "args": {"obj_id": "sphere-0",
                           "r": (0.5, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4)}},
             ],
             "reply_colors": {"red"}},
        ],
    },
    # Achromatic realism — the exact live-demo scenario: the held object is
    # BLACK (a mouse), so the model must build a BLACK sphere AND say "black",
    # not fabricate a hue. Asserts look_at_current_frame first, an all-low RGB
    # box, and (via reply_colors) build-vs-said honesty: a reply naming any
    # non-black colour, or a reply colour inconsistent with the built RGB,
    # fails. Mock-only colour (no train-on-test).
    {
        "name":         "vision_apply_neutral_black",
        "category":     "vision",
        "scene":        [],
        "user":         "Make a sphere the same colour as the mouse I'm holding.",
        "vision":       "The user is holding a black computer mouse.",
        "first_call":   ["look_at_current_frame"],
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.0, 0.25), "g": (0.0, 0.25), "b": (0.0, 0.25)}},
        ],
        "reply_colors": {"black"},
    },
    # Hard perception failure: ``vision_error`` short-circuits look_at_current_frame
    # to its canned spoken line (mirroring the worker's _PerceptionUnavailableError),
    # so the model never sees the error. Contract: the query is routed to the
    # camera (first_call) and the failed look produces NO scene mutation (no
    # fabricated build).
    {
        "name":         "vision_camera_error_canned_message",
        "category":     "vision",
        "scene":        [],
        "user":         "Make a sphere the colour of the thing I'm holding.",
        "vision_error": True,
        "first_call":   ["look_at_current_frame"],
    },
    # Vague success answer: the VLM answered but named no usable colour, so the
    # answer goes back to the model unguarded and honesty is purely prompt-driven.
    # Assert no fabrication: with no ``result`` the turn must not mutate, and
    # ``reply_colors=set()`` forbids the reply naming ANY colour (it should say it
    # couldn't tell / ask the user). Mock answer has no colour word (no train-on-test).
    {
        "name":         "vision_vague_answer_no_fabrication",
        "category":     "vision",
        "scene":        [],
        "user":         "Make a sphere the colour of the thing I'm holding.",
        "vision":       "I can't quite tell — it's partly out of frame and the "
                        "lighting is too dim to make it out.",
        "first_call":   ["look_at_current_frame"],
        "reply_colors": set(),
    },
    # Ambiguous-colour SELF-CONSISTENCY — different from the no-fabrication
    # cases: the camera answer IS a real colour, just one a model could
    # legitimately name two ways (a blue-green object → "teal" or "blue"). The
    # model is EXPECTED to pick one and build + speak a colour. Two checks
    # combine: the build must land in the blue-green family (the ``result`` rgb
    # box admits teal/cyan/green/blue but rejects warm/neutral hues — so a model
    # that builds RED and says "red" fails even though it's self-consistent),
    # AND ``reply_self_consistent`` derives the allowed spoken names from the
    # BUILT rgb so a "build teal / say green" split still FAILS. The box pins no
    # single hue — either honest reading of the percept passes. Mock colour
    # avoids the prompt's worked examples (no train-on-test).
    {
        "name":         "vision_ambiguous_colour_self_consistent",
        "category":     "vision",
        "scene":        [],
        "user":         "Make a sphere the colour of the thing I'm holding.",
        "vision":       "It's a blue-green object — somewhere between blue and "
                        "green, hard to pin down exactly.",
        "first_call":   ["look_at_current_frame"],
        # Blue-green family box: low red rejects red/orange/yellow/purple/white;
        # a green floor rejects black and warm neutrals. Admits the correct
        # teal/cyan/green/blue percepts (the blue anchor carries g≈0.4).
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere", "r": (0.0, 0.5), "g": (0.3, 1.0)}},
        ],
        "reply_self_consistent": True,
    },
    # Maximally ambiguous — the camera answer acknowledges a colour EXISTS but
    # names NONE, so NO colour is derivable. Distinct from
    # vision_ambiguous_colour_self_consistent (which gives a dual-readable real
    # colour): here there is nothing to pick from, so the honest path is the
    # no-fabrication contract — ask which colour / say it can't tell, and do NOT
    # silently build a default sphere or name a colour. Same assertion shape as
    # the other no-fabrication cases: no ``result`` forbids a coloured mutation,
    # ``reply_colors=set()`` forbids the reply naming ANY colour. Mock avoids the
    # prompt's worked examples (no train-on-test).
    {
        "name":         "vision_colour_unspecified_no_fabrication",
        "category":     "vision",
        "scene":        [],
        "user":         "Make a sphere the colour of the thing I'm holding.",
        "vision":       "It's a coloured object.",
        "first_call":   ["look_at_current_frame"],
        "reply_colors": set(),
        # The honest behaviour is to ASK (not build); a clarifying question may
        # list sample colours without that being a fabrication. Fail only a
        # DECLARATIVE colour claim, not colours offered inside the question.
        "clarify_examples_ok": True,
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
                    if t.name in WORKER_MANAGED or t.name in SUPERSEDED_PERCEPTION:
                        continue
                    schema = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
                    tools.append({"type": "function", "function": {
                        "name": t.name,
                        "description": (t.description or "").strip(),
                        "parameters": schema,
                    }})
        except Exception as exc:
            print(f"WARN: discovery failed for {url}: {exc}", file=sys.stderr)
    # look_at_current_frame is brain-executed (not served by any MCP), so
    # discovery misses it — append it so the model is actually offered the tool.
    tools.append(_PERCEPTION_TOOL_DEF)
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
    v = _ROBUSTNESS_VARIANT
    scene_block = _format_scene(scene)
    pose_block  = _format_pose(pose)
    moves_block = _format_recent_moves(recent_moves)
    conv_block  = _format_recent_conversation(history)
    participant = f"Participant: {EVAL_PID}"
    ref_us = v.get("ref_us") or _CASE_REF_US[0]
    reftime = (f"Reference time (when user spoke): {ref_us} µs"
               if ref_us else None)
    if v.get("order") == "worker":
        context_parts = [scene_block, pose_block, participant]
        if reftime:
            context_parts.append(reftime)
        if moves_block:
            context_parts.append(moves_block)
        if conv_block:
            context_parts.append(conv_block)
    else:
        context_parts = [scene_block, pose_block]
        if reftime:
            context_parts.append(reftime)
        if moves_block:
            context_parts.append(moves_block)
        if conv_block:
            context_parts.append(conv_block)
        context_parts.append(participant)
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
# Canned look_at_current_frame answer(s) for the current case (real-world
# visual-query cases). A case may give a single string or a SEQUENCE of
# strings indexed by TURN: _run_turns advances _VISION_IDX once per turn (the
# last entry repeats), so every look within a turn sees the same frame while a
# re-query in the next turn "sees" a different colour — proving the agent
# re-fetches rather than reusing a stale answer, without a model that looks
# twice in one turn desyncing the later turns. Set ONCE per case; _VISION_IDX
# survives across the case's turns (it is NOT cleared by _reset_exec_state) and
# resets only in _set_case_vision.
_CASE_VISION: list[str] = []
_VISION_IDX: list[int] = [0]
# When set, look_at_current_frame returns a graceful FAILURE (no colour) this
# case instead of an answer — mirrors the worker's {"error":.., "spoken":..}
# contract. Lets a case assert the model does NOT fabricate a colour when the
# camera look fails (the live "Unknown tool → invented colour" regression).
_CASE_VISION_ERROR: list[bool] = [False]
# Per-case Reference-time injection (µs). When set, _build_messages emits the
# "Reference time …" metadata line for this case under normal scoring, so a
# case can prove the model treats the timestamp as bookkeeping, not a coordinate.
_CASE_REF_US: list[int | None] = [None]


def _set_case_ref_us(ref_us: int | None) -> None:
    _CASE_REF_US[0] = ref_us


def _set_fixture_scene(scene: list[dict]) -> None:
    _FIXTURE_SCENE.clear()
    _FIXTURE_SCENE.extend(scene)


def _set_case_vision(answer: str | list[str] | None) -> None:
    _CASE_VISION.clear()
    if isinstance(answer, str):
        _CASE_VISION.append(answer)
    elif answer:
        _CASE_VISION.extend(answer)
    _VISION_IDX[0] = 0


def _set_case_vision_error(err: bool | None) -> None:
    _CASE_VISION_ERROR[0] = bool(err)


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
    # Visual-query mock: look_at_current_frame is the worker's brain-executed
    # live-frame perception tool. It returns {"answer": "<vlm text>"}; we feed
    # the case's canned camera description so the harness can check the model
    # consumed the camera result, not the scene block. A ``vision`` sequence is
    # indexed by TURN (advanced once per turn in _run_turns, last repeats), so
    # every look within one turn sees the same frame and a re-query in the next
    # turn "sees" the fresh colour — a model that looks twice in a turn can't
    # desync the later turns.
    if name == "look_at_current_frame":
        if _CASE_VISION_ERROR[0]:
            return {"error": "perception unavailable",
                    "spoken": "I can't see a camera feed right now — "
                              "please check your camera."}
        if not _CASE_VISION:
            return {"answer": "I can't make out the image."}
        idx = min(_VISION_IDX[0], len(_CASE_VISION) - 1)
        return {"answer": _CASE_VISION[idx]}
    return {"_eval_skipped": True, "reason": f"{name} not in safe-exec list"}


def _extract_json(text: str) -> str | None:
    """First balanced top-level ``{…}`` object in ``text`` (worker's
    _extract_json), used to recover a tool call emitted as plain-text JSON."""
    depth, start, in_string, escape = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escape:        escape = False
            elif ch == "\\":  escape = True
            elif ch == '"':   in_string = False
            continue
        if ch == '"':   in_string = True; continue
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            if depth == 0: continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    return None


def _recover_text_tool_call(content: str, all_names: set[str]) -> dict | None:
    """Recover a tool call the model emitted as text instead of via the
    tool_calls field, matching the worker: a bare tool name, or a JSON object
    naming a known tool. Returns a tool-call dict (or None when nothing
    recoverable)."""
    if not content:
        return None
    name: str | None = None
    args: dict = {}
    if content in all_names:
        name = content
    else:
        obj_text = _extract_json(content)
        if obj_text:
            try:
                obj = json.loads(obj_text)
            except json.JSONDecodeError:
                return None
            cand = obj.get("name") or obj.get("tool") or obj.get("function")
            if isinstance(cand, str) and cand in all_names:
                name = cand
                a = obj.get("arguments") or obj.get("args") or {}
                args = a if isinstance(a, dict) else {}
    if name is None:
        return None
    return {"id": f"recovered_{name}",
            "function": {"name": name, "arguments": json.dumps(args)}}


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
    all_names = {t["function"]["name"] for t in tools}
    all_calls: list[dict] = []
    last_msg: dict = {}
    t_total = 0.0
    # local_thinking drops thinking and retries the same step when thinking
    # fills the token budget, as the worker does.
    local_thinking = thinking

    step = 0
    while step < max_steps:
        body = {
            "model": AGENT_MODEL,
            "messages": messages,
            "tools": tools,
            "max_tokens": 2048 if local_thinking else 1024,
            "temperature": 0.0,
            "chat_template_kwargs": {
                "enable_thinking": local_thinking,
                **({"thinking_budget": 1024} if local_thinking else {}),
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
        choice = r.json()["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason") or ""
        last_msg = msg
        tcs = msg.get("tool_calls") or []
        if not tcs:
            # Thinking filled the budget: turn it off and retry the same step.
            if local_thinking and finish == "length":
                local_thinking = False
                continue
            # Recover a tool call the model emitted as plain text, as the
            # worker does; a genuine final response ends the turn.
            recovered = _recover_text_tool_call(
                (msg.get("content") or "").strip(), all_names)
            if recovered is None:
                break
            tcs = [recovered]
        # Final iteration runs no execution turn, so record the model's
        # emitted batch verbatim — there is nothing to truncate against.
        if step + 1 >= max_steps:
            all_calls.extend(tcs)
            break
        new_msgs: list[dict] = [{"role": "assistant", "content": "", "tool_calls": tcs}]
        spoken_end: str | None = None
        for tc in tcs:
            # Record each call only as it is executed: a perception
            # short-circuit (below) stops the live worker mid-batch, so calls
            # after it never run and must not count as scene mutations.
            all_calls.append(tc)
            fn = tc["function"]
            result = await _exec_tool(fn["name"], fn["arguments"], pose)
            # Mirror the worker's _PerceptionUnavailableError short-circuit
            # (processors.py): a perception result carrying a "spoken" message
            # deterministically ends the turn with that canned line, so the
            # model never sees the error and cannot fabricate.
            if isinstance(result, dict) and result.get("spoken"):
                spoken_end = str(result["spoken"])
                break
            new_msgs.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result, default=str)})
        if spoken_end is not None:
            last_msg = {"content": spoken_end}
            break
        messages = messages + new_msgs
        step += 1

    return {"latency_s":  round(t_total, 2),
            "tool_calls": all_calls,
            "content":    (last_msg.get("content") or "").strip(),
            "reasoning":  (last_msg.get("reasoning_content") or "").strip()}


def _apply_mutations(scene: list[dict], tool_calls: list[dict],
                     add_counter: dict[str, int] | None = None) -> list[dict]:
    """Return a NEW scene with the turn's add/update/remove calls applied,
    mirroring _exec_tool's add_primitive id scheme (a counter per prim_type, so
    the first sphere is ``sphere-0``). Pass a persistent ``add_counter`` to carry
    the numbering across turns so consecutive-turn adds don't both become
    ``sphere-0``. Lets a multi-turn case carry a created object forward so a
    later turn can update/remove it by id, and resolves a partial update's
    omitted colour channels against the object's carried-forward colour."""
    out = [dict(o) for o in scene]
    if add_counter is None:
        add_counter = {}
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name")
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (TypeError, ValueError):
            continue
        if name == "add_primitive":
            prim = args.get("prim_type", "sphere")
            n = add_counter.get(prim, -1) + 1
            add_counter[prim] = n
            out.append({
                "id":    f"{prim}-{n}",
                "type":  prim,
                "pos":   [float(args.get("x", 0.0)),
                          float(args.get("y", 1.6)),
                          float(args.get("z", -1.5))],
                # Default to white, matching the prompt's unspecified-colour rule.
                "color": [float(args.get("r", 1.0)),
                          float(args.get("g", 1.0)),
                          float(args.get("b", 1.0))],
                "size":  float(args.get("size", 0.1)),
            })
        elif name == "update_primitive":
            oid = args.get("obj_id") or args.get("object_id")
            for o in out:
                if o["id"] == oid:
                    if any(k in args for k in ("r", "g", "b")):
                        o["color"] = [float(args.get("r", o["color"][0])),
                                      float(args.get("g", o["color"][1])),
                                      float(args.get("b", o["color"][2]))]
                    if any(k in args for k in ("x", "y", "z")):
                        o["pos"] = [float(args.get("x", o["pos"][0])),
                                    float(args.get("y", o["pos"][1])),
                                    float(args.get("z", o["pos"][2]))]
                    if "size" in args:
                        o["size"] = float(args["size"])
                    break
        elif name == "remove_primitive":
            oid = args.get("obj_id") or args.get("object_id")
            out = [o for o in out if o["id"] != oid]
    return out


async def _run_turns(http: httpx.AsyncClient, system_prompt: str,
                     tools: list[dict], scene: list[dict], pose: dict,
                     turns: list[dict], *, thinking: bool = False,
                     max_steps: int = 1) -> list[dict]:
    """Run a sequence of user turns, mirroring the worker's rolling-history
    flow: each turn is a FRESH rollout (system + scene + [Recent conversation]
    of prior turns' spoken replies + this turn's request), exactly as
    ``XRRenderProcessor`` rebuilds messages per turn — turn N never sees turn
    N-1's raw tool calls, only its text reply. The case's ``vision`` sequence
    is consumed across turns (set once by the caller), so a re-query can be
    answered by a DIFFERENT camera colour.

    Scene mutations are carried forward: turn N+1 sees the objects turn N
    created/updated (via _apply_mutations), so an "update the sphere I just
    made" turn has a valid obj_id to target. Each result carries the
    ``scene`` snapshot used for that turn so the per-turn check resolves
    partial updates against the right object state. Returns one ``_run_one``
    result per turn, in order."""
    history: list[tuple[str, str]] = []
    results: list[dict] = []
    running_scene = [dict(o) for o in scene]
    # Carried across turns so a later turn's add_primitive doesn't reuse a prior
    # turn's id (two ``sphere-0``s when consecutive turns each add a sphere).
    add_counter: dict[str, int] = {}
    for turn_i, turn in enumerate(turns):
        # Advance the camera frame once per TURN (not per look_at_current_frame
        # call), so a turn that looks twice doesn't desync the next turn's frame.
        _VISION_IDX[0] = turn_i
        _set_case_history(history)
        turn_scene = [dict(o) for o in running_scene]
        r = await _run_one(http, system_prompt, tools, turn_scene, pose,
                            turn["user"], thinking=thinking, max_steps=max_steps)
        r["scene"] = turn_scene
        results.append(r)
        history = history + [(turn["user"], r["content"])]
        running_scene = _apply_mutations(running_scene, r["tool_calls"], add_counter)
    return results


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


# Colour words scanned in the model's spoken reply to catch misreporting
# (e.g. "Here's your white sphere" after building a magenta one). Kept
# broad on purpose so a stray wrong colour anywhere in the reply trips it.
_REPLY_COLOR_WORDS = frozenset({
    "red", "orange", "yellow", "green", "blue", "purple", "violet",
    "magenta", "pink", "cyan", "teal", "turquoise", "white", "black",
    "gray", "grey", "silver", "brown", "maroon", "navy", "gold",
    "golden", "indigo", "lavender", "lilac", "lime", "olive", "salmon",
    "coral", "crimson", "beige", "tan", "cream", "mint",
})

# Compound colour phrases name ONE hue; the constituent word ("green" inside
# "blue-green") must not be mistaken for the standalone hue when scanning a
# reply. Collapse these to their single canonical token first so honesty is
# judged on the phrase's real meaning (blue-green == teal), not a substring.
_COMPOUND_COLORS: dict[str, str] = {
    "blue-green": "teal",   "blue green": "teal",   "bluegreen": "teal",
    "green-blue": "teal",   "green blue": "teal",
    "bluish-green": "teal", "bluish green": "teal",
    "greenish-blue": "teal", "greenish blue": "teal",
    "blue-ish green": "teal", "green-ish blue": "teal",
}


def _collapse_compound_colours(blob: str) -> str:
    """Replace multi-word colour phrases with their single canonical hue so a
    constituent word isn't scanned as a separate (possibly wrong) colour."""
    # Models often join compound hues with a non-ASCII hyphen (e.g. "blue‑green"
    # using U+2011); fold the Unicode hyphen/dash variants to a plain "-" so the
    # phrase table matches and the constituent word isn't scanned as a hue.
    blob = re.sub(r"[\u2010\u2011\u2012\u2013\u2014]", "-", blob)
    for phrase, canon in _COMPOUND_COLORS.items():
        blob = blob.replace(phrase, canon)
    return blob


def _declarative_colours(content: str) -> set[str]:
    """Colour words the reply asserts as FACT, excluding any that appear only
    inside a clarifying question or an example list — e.g. "what colour would
    you like? (e.g. red, blue)" declares NO colour, whereas "it's red" does.
    Lets an undeterminable-colour case accept an honest clarifying question
    that happens to list sample colours, while still catching a fabricated
    declarative claim."""
    s = _collapse_compound_colours(content.lower())
    s = re.sub(r"\([^)]*\)", " ", s)                       # parenthetical asides
    s = re.sub(r"\b(e\.?\s*g\.?|for example|such as|"      # example clauses …
               r"examples?\s*:|options?\s*:)\b[^.?!]*", " ", s)  # … to sentence end
    # Strip only the clause that actually ends in '?', bounded on the left by a
    # clause separator, so a declarative before the question survives ("it's
    # red, what did you want?" keeps "red") while colours offered inside the
    # question are still dropped.
    s = re.sub(r"[^.?!,;:—–-]*\?", " ", s)                 # interrogative clause
    return {w for w in _REPLY_COLOR_WORDS if re.search(rf"\b{w}\b", s)}

# Canonical RGB for every word the reply scanner knows, used to map an
# actually-built colour back to the names that honestly describe it.
# Values are nominal hue anchors, not the prompt's worked-example triples.
_COLOR_ANCHORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),     "orange": (1.0, 0.5, 0.0),
    "yellow": (1.0, 1.0, 0.0),  "green": (0.0, 0.8, 0.0),
    "blue": (0.0, 0.4, 1.0),    "purple": (0.6, 0.0, 1.0),
    "violet": (0.6, 0.0, 1.0),  "magenta": (1.0, 0.0, 1.0),
    "pink": (1.0, 0.45, 0.7),   "cyan": (0.0, 1.0, 1.0),
    "teal": (0.0, 0.5, 0.5),    "turquoise": (0.0, 0.8, 0.8),
    "white": (1.0, 1.0, 1.0),   "black": (0.0, 0.0, 0.0),
    "gray": (0.5, 0.5, 0.5),    "grey": (0.5, 0.5, 0.5),
    "silver": (0.75, 0.75, 0.75), "brown": (0.4, 0.2, 0.05),
    "maroon": (0.5, 0.0, 0.0),  "navy": (0.0, 0.0, 0.5),
    "gold": (1.0, 0.84, 0.0),   "golden": (1.0, 0.84, 0.0),
    "indigo": (0.3, 0.0, 0.5),  "lavender": (0.7, 0.6, 0.9),
    "lilac": (0.78, 0.6, 0.85), "lime": (0.6, 1.0, 0.0),
    "olive": (0.5, 0.5, 0.0),   "salmon": (0.98, 0.5, 0.45),
    "coral": (1.0, 0.5, 0.31),  "crimson": (0.86, 0.08, 0.24),
    "beige": (0.8, 0.75, 0.6),  "tan": (0.82, 0.71, 0.55),
    "cream": (1.0, 0.99, 0.82), "mint": (0.6, 1.0, 0.75),
}

# Words that may honestly stand in for one another in a reply, so an
# accurate paraphrase ("navy" object called "blue") is not flagged as a
# build/say mismatch regardless of raw RGB distance.
_COLOR_SYNONYMS: dict[str, set[str]] = {
    "gray": {"grey", "silver"}, "grey": {"gray", "silver"},
    "silver": {"gray", "grey"},
    "purple": {"violet"}, "violet": {"purple"},
    "magenta": {"pink"}, "pink": {"magenta", "salmon"},
    "cyan": {"teal", "turquoise"}, "teal": {"cyan", "turquoise"},
    "turquoise": {"cyan", "teal"},
    "navy": {"blue"}, "indigo": {"purple", "violet", "blue"},
    "gold": {"golden", "yellow"}, "golden": {"gold", "yellow"},
    "maroon": {"red", "brown"}, "crimson": {"red"},
    "olive": {"green", "yellow"}, "lime": {"green"},
    "salmon": {"pink", "coral"}, "coral": {"salmon", "orange", "pink"},
    "lavender": {"lilac", "purple", "violet"},
    "lilac": {"lavender", "purple", "violet"},
    "beige": {"tan", "cream"}, "tan": {"beige", "brown"},
    "cream": {"beige", "white"}, "mint": {"green"},
}

# A reply colour farther than this (Euclidean, unit RGB cube) from every
# built-consistent name is treated as a build/say mismatch.  Generous
# enough that honest paraphrases pass via the synonym map; tight enough
# that "built white, said magenta" (distance 1.0) trips.
_BUILD_NEAR = 0.55


def _built_color_names(rgb: tuple[float, float, float]) -> set[str]:
    """Names that honestly describe an as-built colour: every anchor
    within ``_BUILD_NEAR`` plus the single globally-nearest anchor,
    expanded through the synonym map both ways."""
    scored = sorted(
        (math.dist(rgb, anc), name) for name, anc in _COLOR_ANCHORS.items()
    )
    names = {name for d, name in scored if d <= _BUILD_NEAR}
    names.add(scored[0][1])
    for n in list(names):
        names |= _COLOR_SYNONYMS.get(n, set())
    for base, syns in _COLOR_SYNONYMS.items():
        if names & syns:
            names.add(base)
    return names


def _color_family(word: str) -> set[str]:
    """Synonym-connected family of a colour word, via _COLOR_SYNONYMS in both
    directions (the word, its listed synonyms, and any base that lists it).
    Lets the requested-colour check accept an honest in-family paraphrase
    (e.g. "crimson"/"lime" for a "red"/"green" allow-set) while a cross-family
    name still fails. Falls back to {word} for words with no synonyms."""
    fam = {word} | _COLOR_SYNONYMS.get(word, set())
    for base, syns in _COLOR_SYNONYMS.items():
        if word in syns:
            fam.add(base)
    return fam


def _reply_color_ok(content: str, allowed: set[str],
                    built_rgb: tuple[float, float, float] | None = None
                    ) -> tuple[bool, str]:
    """Colour-focused cases vet the spoken reply two ways.  First the
    requested check: the reply must not name a colour outside the case's
    allow-set (in-family synonyms of an allowed colour are accepted via
    _color_family; cross-family names fail).  Second the build/say check:
    any colour the reply does name must also be consistent with what was
    ACTUALLY built — catching "built X, narrated Y" even when Y was the
    requested colour.  Only colours the reply utters are checked, so a reply
    naming no colour passes."""
    blob = _collapse_compound_colours(content.lower())
    named = {w for w in _REPLY_COLOR_WORDS if re.search(rf"\b{w}\b", blob)}
    allowed_family: set[str] = set()
    for a in allowed:
        allowed_family |= _color_family(a)
    wrong = {w for w in named if not (_color_family(w) & allowed_family)}
    if wrong:
        snippet = " ".join(content.split())[:160]
        return False, (f"reply names colour {sorted(wrong)} ≠ requested "
                       f"(allowed {sorted(allowed)}) | reply={snippet!r}")
    if built_rgb is not None and named:
        honest = _built_color_names(built_rgb)
        mismatched = {
            w for w in named
            if not ({w} | _COLOR_SYNONYMS.get(w, set())) & honest
        }
        if mismatched:
            return False, (
                f"reply names {sorted(mismatched)} but built rgb "
                f"({built_rgb[0]:.2f}, {built_rgb[1]:.2f}, {built_rgb[2]:.2f}) "
                f"reads as {sorted(honest)}"
            )
    return True, "reply colour matches"


def _built_rgb(muts: list[dict],
               scene: list[dict]) -> tuple[float, float, float] | None:
    """Extract the as-built (r, g, b) from the latest colour-bearing
    add/update_primitive, resolving channels omitted on an update against
    the named scene object.  Returns None when all three aren't known."""
    for tc in reversed(muts):
        if tc["function"]["name"] not in ("add_primitive", "update_primitive"):
            continue
        args = tc["function"]["arguments"]
        args = json.loads(args) if isinstance(args, str) else args
        if not any(ch in args for ch in ("r", "g", "b")):
            continue
        obj_id = args.get("obj_id")
        chans: list[float] = []
        for ch in ("r", "g", "b"):
            v = args.get(ch)
            if v is None and obj_id:
                v = _resolve_arg(obj_id, ch, scene)
            if v is None:
                break
            try:
                chans.append(float(v))
            except (TypeError, ValueError):
                break
        if len(chans) == 3:
            return (chans[0], chans[1], chans[2])
    return None


def _is_achromatic(rgb: tuple[float, float, float]) -> bool:
    """True for white / gray / black-ish colours — the channel spread is tiny,
    so there is no real hue. ``make a coloured sphere`` demands a saturated hue,
    so an achromatic build is a silent dodge of the request."""
    return (max(rgb) - min(rgb)) < 0.2


def _scene_last_rgb(scene: list[dict]) -> tuple[float, float, float] | None:
    """The (r, g, b) of the most-recent colour-bearing object already in the
    scene — used by self-consistency turns to check a reply that confirms a
    PREVIOUSLY-built object's colour without mutating again."""
    for obj in reversed(scene or []):
        col = obj.get("color")
        if isinstance(col, (list, tuple)) and len(col) == 3:
            try:
                return (float(col[0]), float(col[1]), float(col[2]))
            except (TypeError, ValueError):
                continue
    return None


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

    # Opt-in ordered-call assertion: runs against the FULL ordered call list
    # (helper/compute tools included), unlike the order-independent matcher
    # below. Only cases that set ``ordered_calls`` use it.
    ordered = case.get("ordered_calls")
    if ordered is not None:
        ok, why = ordered(tcs)
        if not ok:
            return False, f"ordered-call check failed: {why}"

    # ── exactly-one-build guard ──────────────────────────────────────────────
    # A single-object create case must emit EXACTLY ONE add_primitive; >1 means
    # the model re-added the same object in several colours, which the default
    # matcher would hide by finding one matching add among many. Scoped to the
    # colour/vision single-object cases (a ``result`` with exactly one
    # add_primitive in those categories, or forced via ``expect_single_build``);
    # cases that create 2+ objects on purpose (2+ adds, or ``allow_multi_build``)
    # and spatial cases are unaffected.
    _n_add_expected = sum(1 for e in (case.get("result") or [])
                          if e.get("tool") == "add_primitive")
    _single_create_cat = case.get("category") in ("color", "vision")
    if (case.get("expect_single_build")
            or (_single_create_cat and _n_add_expected == 1
                and not case.get("allow_multi_build"))):
        adds = sum(1 for tc in tcs if tc["function"]["name"] == "add_primitive")
        if adds != 1:
            return False, (f"expected exactly 1 add_primitive for a single-object "
                           f"request, model created {adds}")

    # ── real-world visual-query routing cases ────────────────────────────────
    # A case with ``first_call`` asserts the model treated the utterance as a
    # REAL-WORLD query: its FIRST tool call routes to the camera/VLM (never a
    # scene mutation, never a scene-state text answer). ``reply_names`` /
    # ``reply_forbids`` then check the FINAL reply repeats the colour the mocked
    # camera returned (and NOT a virtual SCENE primitive's colour) — proving the
    # model consumed the camera result instead of reading the scene block.
    first_allowed = case.get("first_call")
    if first_allowed is not None:
        allowed = set(first_allowed)
        if not tcs:
            return False, (f"no tool call — a real-world query must route to the "
                           f"camera {sorted(allowed)}, not be answered from "
                           f"scene state")
        first = tcs[0]["function"]["name"]
        if first not in allowed:
            names = [tc["function"]["name"] for tc in tcs]
            return False, f"first call {first!r} not in {sorted(allowed)}; calls={names}"
        # Two modes, distinguished by whether the case also carries a ``result``:
        #   - pure query (no result): the turn must NOT mutate the scene; check
        #     the spoken reply repeats / avoids the camera colour.
        #   - apply-colour (result present): the turn SHOULD mutate using the
        #     PERCEIVED colour — camera-first is verified here, then we fall
        #     through to the RGB/tool-call matcher below so the mutation's colour
        #     is checked. This proves the model USED the camera answer, not just
        #     reported it.
        if not case.get("result"):
            bad = [tc["function"]["name"] for tc in tcs
                   if tc["function"]["name"] in _MUTATING_TOOLS]
            if bad:
                return False, f"mutated the scene on a real-world query: {bad}"
            content = (actual.get("content") or "")
            blob = content.lower()
            names_wanted = case.get("reply_names") or []
            for want in names_wanted:
                if not re.search(rf"\b{re.escape(want.lower())}\b", blob):
                    return False, (f"final reply {content!r} omits the camera-reported "
                                   f"colour {want!r}")
            # reply_forbids guards against naming a stale or virtual-scene
            # colour. Stale re-query cases enforce it conditionally: once the
            # fresh colour is named, an honest "it's not <stale> anymore, it's
            # <fresh>" should pass, so the forbid only bites when the fresh
            # colour is absent. The virtual-distractor cases set
            # ``reply_forbids_always`` — naming the scene primitive's colour is
            # always wrong, even alongside the real one. Forbid-only cases (no
            # reply_names) stay fully guarded.
            names_satisfied = bool(names_wanted) and all(
                re.search(rf"\b{re.escape(w.lower())}\b", blob) for w in names_wanted
            )
            if case.get("reply_forbids_always") or not names_satisfied:
                for forbid in case.get("reply_forbids") or []:
                    if re.search(rf"\b{re.escape(forbid.lower())}\b", blob):
                        return False, (f"final reply {content!r} names the VIRTUAL scene "
                                       f"colour {forbid!r} instead of the real one")
            # No-grounding honesty: with reply_colors={} the reply must name NO
            # colour at all — so a failed look that still narrates a fabricated
            # colour fails (the live "couldn't see → invented a colour" bug).
            reply_colors = case.get("reply_colors")
            if reply_colors is not None:
                if reply_colors == set() and case.get("clarify_examples_ok"):
                    # Undeterminable colour: an honest clarifying question is
                    # correct even if it lists sample colours. Fail ONLY a
                    # declarative colour claim, not colours offered as examples.
                    decl = _declarative_colours(content)
                    if decl:
                        snippet = " ".join(content.split())[:160]
                        return False, (f"reply declares colour {sorted(decl)} on an "
                                       f"undeterminable colour | reply={snippet!r}")
                else:
                    rc_ok, rc_msg = _reply_color_ok(content, set(reply_colors), None)
                    if not rc_ok:
                        return False, rc_msg
            return True, (f"routed first to {first}"
                          + (f"; reply reports {case['reply_names']}"
                             if case.get("reply_names") else "")
                          + ("; no fabricated colour"
                             if reply_colors == set() else ""))
        # apply-colour mode: camera verified as the first call; fall through to
        # the result matcher to check the colour was applied to the mutation.

    # Accept-either honesty for a USER-underspecified hue ("make a coloured
    # sphere"): BOTH a clarifying question and a self-consistent build are
    # correct — only a say!=built colour mismatch fails. Passes if the reply
    # names no committed colour (a clarifying question / neutral confirm), or
    # if a named colour is consistent with what was built this turn OR already
    # exists in the carried scene. ``require_coloured_result`` (turn 2) further
    # demands a coloured sphere actually exist by the end of the turn.
    if case.get("accept_clarify_or_consistent"):
        scene = case.get("scene") or []
        adds = sum(1 for tc in tcs if tc["function"]["name"] == "add_primitive")
        if adds > 1:
            return False, (f"expected at most 1 add_primitive (one sphere), "
                           f"model created {adds} (multiple add_primitive calls)")
        muts = [tc for tc in tcs if tc["function"]["name"] in _MUTATING_TOOLS]
        content = actual.get("content", "")
        # Both honest paths speak — a clarifying question or a one-line confirm
        # of the built hue — so an empty reply is never acceptable.
        if not content.strip():
            return False, "empty reply on a 'coloured' request (expected a question or a colour)"
        named = {w for w in _REPLY_COLOR_WORDS
                 if re.search(rf"\b{w}\b", _collapse_compound_colours(content.lower()))}
        built = _built_rgb(muts, scene) if muts else _scene_last_rgb(scene)
        if case.get("require_coloured_result") and built is None:
            return False, "no self-consistent coloured sphere exists after this turn"
        if muts and built is None:
            return False, "built an object with no colour on a 'coloured' request"
        # "Coloured" means a saturated hue: a white/gray/black build dodges the
        # request even when narrated honestly.
        if built is not None and _is_achromatic(built):
            return False, (f"built an achromatic colour "
                           f"({built[0]:.2f}, {built[1]:.2f}, {built[2]:.2f}) "
                           f"on a 'coloured' request")
        if not named:
            return True, ("asked to clarify (no colour committed)" if not muts
                          else f"built rgb ({built[0]:.2f}, {built[1]:.2f}, "
                               f"{built[2]:.2f}); named no colour")
        if built is None:
            return False, f"reply names {sorted(named)} but nothing coloured was built"
        ok, msg = _reply_color_ok(content, _built_color_names(built), built)
        if not ok:
            return False, msg
        return True, (f"said {sorted(named)} consistent with built rgb "
                      f"({built[0]:.2f}, {built[1]:.2f}, {built[2]:.2f})"
                      + ("" if muts else " [carried]"))

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
    # Self-consistency mode: no fixed expected colour. The spoken reply must
    # name SOME colour and that colour must be consistent with the colour the
    # model ACTUALLY built — allowed names are derived from the build itself
    # (_built_color_names) rather than pinned, so any honest interpretation of
    # an ambiguous object passes ("build teal / say teal") while a said!=built
    # mismatch fails ("build blue / say green", "build white / say orange").
    if case.get("reply_self_consistent"):
        built = _built_rgb(muts, scene)
        if built is None:
            return False, "self-consistency case built no colour to check"
        content = actual.get("content", "")
        named = {w for w in _REPLY_COLOR_WORDS
                 if re.search(rf"\b{w}\b", _collapse_compound_colours(content.lower()))}
        route = (f"routed first to {tcs[0]['function']['name']}; "
                 if case.get("first_call") else "")
        # The invariant is consistency, not mandatory speaking: naming NO
        # colour is never a say!=built mismatch, so it's acceptable. Only a
        # named colour that disagrees with the built rgb fails.
        if not named:
            return True, (f"{route}built rgb ({built[0]:.2f}, {built[1]:.2f}, "
                          f"{built[2]:.2f}); named no colour")
        ok, msg = _reply_color_ok(content, _built_color_names(built), built)
        if not ok:
            return False, msg
        return True, (f"{route}said {sorted(named)} consistent with built rgb "
                      f"({built[0]:.2f}, {built[1]:.2f}, {built[2]:.2f})")
    allowed = case.get("reply_colors")
    if allowed is not None:
        ok, msg = _reply_color_ok(actual.get("content", ""), set(allowed),
                                  _built_rgb(muts, scene))
        if not ok:
            return False, msg
    if case.get("first_call"):
        return True, (f"routed first to {tcs[0]['function']['name']}; "
                      f"matched {len(wanted)} colour mutation(s)")
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
    """Reserved spatial-vocab colour/shape words (``_EVAL_VOCAB_*``) present in
    this case's MODEL-VISIBLE fixture: user utterance, history dialogue, scene
    type tags, ids. Scans only model-visible fields, not ``vision`` /
    ``reply_names`` / ``reply_colors`` / ``result`` — those hold runtime-only
    mock colours that never reach the prompt, so flagging them would false-
    positive on the recommended worked-example palette. Used to attribute
    reserved-vocab violations to specific cases."""
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
        # Colour-focused cases (category="color") deliberately sweep the
        # palette, so their colour WORDS can't constrain the prompt's worked
        # examples and are exempt. Only the colour vocabulary is dropped, not
        # the whole case: their shapes (here) and coords/utterances (checks
        # 1-3 above) are still audited.
        if c.get("category") != "color":
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


async def _robustness_eval_case(http: httpx.AsyncClient, base_prompt: str,
                              tools: list[dict], c: dict, pose: dict,
                              variant: dict) -> tuple[bool, str]:
    """Run one case under one robustness perturbation variant; return
    ``(ok, why)``.  Sets the module-level _ROBUSTNESS_VARIANT (read by
    _build_messages) and the thinking flag / scaffold per variant, mirroring
    the worker's surfaces.  ``why`` is a short reason on failure (for the
    iteration log), empty on pass.  Restores _ROBUSTNESS_VARIANT to its resting
    value on exit so an in-process reuse isn't poisoned by the last variant."""
    global _ROBUSTNESS_VARIANT
    prior_variant = _ROBUSTNESS_VARIANT
    _ROBUSTNESS_VARIANT = variant
    thinking = variant.get("thinking", False)
    sys_prompt = (_THINK_SCAFFOLD + base_prompt) if variant.get("scaffold") else base_prompt
    scene_c = c["scene"]
    pose_c  = c.get("pose", pose)
    _set_case_moves(c.get("recent_moves"))
    _set_case_vision(c.get("vision"))
    _set_case_vision_error(c.get("vision_error"))
    _set_case_ref_us(c.get("ref_us"))
    try:
        if c.get("turns"):
            turn_rs = await _run_turns(http, sys_prompt, tools, scene_c, pose_c,
                                       c["turns"], thinking=thinking,
                                       max_steps=_MAX_STEPS)
            ok = True
            why = ""
            for ti, (tr, td) in enumerate(zip(turn_rs, c["turns"])):
                check_case = {**{k: v for k, v in c.items() if k != "turns"},
                              **td, "scene": tr.get("scene", [])}
                tok, twhy = _check(tr, check_case)
                if not tok and not why:
                    why = f"t{ti + 1}: {twhy}"
                ok = ok and tok
            return ok, why
        _set_case_history(c.get("history"))
        r = await _run_one(http, sys_prompt, tools, scene_c, pose_c,
                           c["user"], thinking=thinking, max_steps=_MAX_STEPS)
        ok, why = _check(r, c)
        return ok, ("" if ok else why)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        _ROBUSTNESS_VARIANT = prior_variant


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
    p.add_argument("--robustness", action="store_true",
                   help="robustness sweep: re-score the --only/.only subset "
                        "under clean + irrelevant context perturbations, ROBUST "
                        "only when every variant passes. Requires a case subset. "
                        "See eval/README.md.")
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
    robustness_from_only = False
    if not args.only and not args.query and only_file.exists():
        names: list[str] = []
        for raw in only_file.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            for tok in line.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                # A bare ROBUSTNESS token enables the sweep (see eval/README.md).
                if tok.upper() == "ROBUSTNESS":
                    if any(c["name"] == tok for c in CASES):
                        print(f"WARNING: {only_file.name} token {tok!r} is the "
                              f"robustness directive, NOT the case of that name; "
                              f"that case will not be selected by this token.")
                    robustness_from_only = True
                    continue
                names.append(tok)
        if names:
            args.only = ",".join(names)
            print(f"FILTER: {only_file.name} → {names}")
        if robustness_from_only:
            print(f"ROBUSTNESS: enabled via {only_file.name} directive")

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
            print("PROMPT/EVAL OVERLAP: clean (no verbatim utterances/coords; no "
                  "reserved spatial-vocab leaks in worked examples — vision-case "
                  "mock colours are runtime-only and out of scope)")

        # Robustness sweep, opt-in via --robustness or a ROBUSTNESS line in
        # eval/.only (see eval/README.md). Off by default.
        robustness_mode = args.robustness or robustness_from_only
        if robustness_mode and not args.only:
            print("\nROBUSTNESS: refusing to sweep the FULL suite ×"
                  f"{len(_ROBUSTNESS_VARIANTS)} variants: that is a large, "
                  "rarely-intended backend cost. Restrict the sweep to a case "
                  "subset via --only or case names in eval/.only alongside the "
                  "ROBUSTNESS line.", file=sys.stderr)
            sys.exit(2)
        if robustness_mode:
            print("\n── ROBUSTNESS SWEEP (clean + irrelevant perturbations) ──")
            tags = [v["tag"] for v in _ROBUSTNESS_VARIANTS]
            all_robust = True
            for c in cases:
                verdict = {}
                reasons = {}
                for v in _ROBUSTNESS_VARIANTS:
                    ok, why = await _robustness_eval_case(
                        http, system_prompt, tools, c, pose, v)
                    verdict[v["tag"]] = ok
                    reasons[v["tag"]] = why
                robust = all(verdict.values())
                all_robust = all_robust and robust
                cells = " ".join(
                    f"{t}={'PASS' if verdict[t] else 'FAIL'}" for t in tags)
                print(f"{c['name']:48s} {cells}  → "
                      f"{'ROBUST' if robust else 'STILL FRAGILE'}")
                for t in tags:
                    if not verdict[t]:
                        print(f"    {t} FAIL: {reasons[t]}")
            print(f"\nROBUSTNESS: {'ALL ROBUST' if all_robust else 'SOME FRAGILE'}")
            sys.exit(0 if all_robust else 1)

        results = []
        for c in cases:
            scene_c = c["scene"]
            pose_c  = c.get("pose", pose)
            _set_case_moves(c.get("recent_moves"))
            _set_case_vision(c.get("vision"))
            _set_case_vision_error(c.get("vision_error"))
            _set_case_ref_us(c.get("ref_us"))
            # Multi-turn case: run each turn as a fresh rollout with rolling
            # text history; the case passes only if every turn's check passes.
            if c.get("turns"):
                try:
                    turn_rs = await _run_turns(http, system_prompt, tools,
                                               scene_c, pose_c, c["turns"],
                                               thinking=args.thinking,
                                               max_steps=_MAX_STEPS)
                except Exception as exc:
                    ok, why = False, f"network error: {type(exc).__name__}: {exc}"
                    print(f"✗ {c['name']:32s}   ----  {why}")
                    results.append((c["name"], ok))
                    continue
                ok = True
                lat = round(sum(r["latency_s"] for r in turn_rs), 2)
                print(f"  {c['name']:32s} {lat:5.1f}s  (multi-turn)")
                for ti, (tr, td) in enumerate(zip(turn_rs, c["turns"])):
                    # Merge the parent case under the per-turn dict so case-level
                    # keys (category, expect_single_build, …) reach the per-turn
                    # check while per-turn keys still override.
                    check_case = {**{k: v for k, v in c.items() if k != "turns"},
                                  **td, "scene": tr.get("scene", [])}
                    # Guard the check too: a malformed model tool-call JSON
                    # (json.loads in _built_rgb/_match_call) must fail this one
                    # turn, not abort the whole batch with a traceback.
                    try:
                        tok, twhy = _check(tr, check_case)
                    except Exception as exc:
                        tok, twhy = False, f"check error: {type(exc).__name__}: {exc}"
                    ok = ok and tok
                    tmark = "✓" if tok else "✗"
                    print(f"  {tmark} turn {ti + 1}: {td['user'][:40]!r}  {twhy}")
                    for i, tc in enumerate(tr["tool_calls"]):
                        fn = tc["function"]
                        print(f"      [{i}] {fn['name']}({fn['arguments']})")
                results.append((c["name"], ok))
                continue
            _set_case_history(c.get("history"))
            try:
                r = await _run_one(http, system_prompt, tools, scene_c, pose_c,
                                   c["user"], thinking=args.thinking,
                                   max_steps=_MAX_STEPS)
            except Exception as exc:
                r = {"latency_s": 0.0, "tool_calls": [], "content": "",
                     "reasoning": ""}
                ok, why = False, f"network error: {type(exc).__name__}: {exc}"
            else:
                # Guard the check separately from the rollout: a malformed model
                # tool-call JSON (json.loads in _built_rgb/_match_call) must fail
                # this one case gracefully, not abort the batch with a traceback.
                try:
                    ok, why = _check(r, c)
                except Exception as exc:
                    ok, why = False, f"check error: {type(exc).__name__}: {exc}"
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
