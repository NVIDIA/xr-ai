# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-turn scene context for the render agent.

Pre-fetches scene state, head pose, and the most common spatial position, then
renders them — plus the move log and conversation history — into the text block
injected into the reasoning loop's user message. Pre-fetching saves 1-3 tool-call
iterations per turn.

This module owns *formatting the world for the model*, so it stays independent
of the turn loop: it takes a ``call_tool`` callable and the caller's history, and
returns a value object. That keeps the context surface reusable by any agent
that needs the same view of the scene.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger

# Mirrors the brain's curated session transcript sink (see ``app.main``).
_trace_log = logger.bind(trace=True)

# (obj_id, (prev_x, prev_y, prev_z), (new_x, new_y, new_z))
Move = tuple[str, tuple[float, float, float], tuple[float, float, float]]

ToolCaller = Callable[..., Awaitable[object]]


@dataclass(frozen=True)
class SceneContext:
    """The rendered context block plus the scene snapshot it was built from."""

    #: Text injected into the reasoning loop's user message.
    text: str
    #: Object id → position at turn start, so ``update_primitive`` calls made
    #: during the turn can be recorded as (prev → new) move-log entries.
    pre_move_positions: dict[str, tuple[float, float, float]] = field(default_factory=dict)


async def build_turn_context(
    call_tool: ToolCaller,
    *,
    pid: str = "",
    ref_us: int = 0,
    recent_moves: list[Move] | None = None,
    history: list[tuple[str, str]] | None = None,
) -> SceneContext:
    """Pre-fetch scene/pose and format the turn-context block.

    ``call_tool`` is the turn's native-tool invoker; scene state, head pose, and
    the 1.5 m-ahead position are fetched concurrently.
    """
    scene, pose, ahead = await asyncio.gather(
        call_tool("get_scene_state", {}, silent=True),
        call_tool("get_head_pose", {}, silent=True),
        call_tool("position_ahead", {"distance": 1.5}, silent=True),
    )

    ctx_parts: list[str] = []
    pre_move_positions: dict[str, tuple[float, float, float]] = {}

    # ── Scene ──────────────────────────────────────────────────────────────
    if isinstance(scene, dict) and scene.get("objects"):
        objs = scene["objects"]
        lines = ["SCENE OBJECTS:"]
        for o in objs:
            pos = o.get("position", {})
            col = o.get("color", {})
            pre_move_positions[o["id"]] = (
                float(pos.get("x", 0)),
                float(pos.get("y", 0)),
                float(pos.get("z", 0)),
            )
            lines.append(
                f"  {o['id']} ({o['type']})  "
                f"pos=({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})  "
                f"color=(r={col.get('r', 0):.2f} g={col.get('g', 0):.2f} b={col.get('b', 0):.2f})  "
                f"size={o.get('size', 0.1):.3f}m"
            )
        ctx_parts.append("\n".join(lines))
    else:
        ctx_parts.append("SCENE OBJECTS: (empty)")

    # ── Head pose + derived spatial shortcuts ─────────────────────────────
    if isinstance(pose, dict) and pose.get("is_valid"):
        p = pose["position"]
        fv = pose["forward"]
        rv = pose["right"]
        uv = pose.get("up", {"x": 0, "y": 1, "z": 0})

        # Compute common offsets directly — no extra tool calls needed.
        def _off(vec: dict, d: float) -> str:
            return f"({p['x'] + vec['x'] * d:.2f}, {p['y'] + vec['y'] * d:.2f}, {p['z'] + vec['z'] * d:.2f})"

        ahead_str = (
            f"({ahead['x']:.2f}, {ahead['y']:.2f}, {ahead['z']:.2f})"
            if isinstance(ahead, dict) and "x" in ahead
            else _off(fv, 1.5)
        )

        ctx_parts.append(
            "HEAD POSE:\n"
            f"  position : ({p['x']:.2f}, {p['y']:.2f}, {p['z']:.2f})\n"
            f"  forward  : ({fv['x']:.3f}, {fv['y']:.3f}, {fv['z']:.3f})  ← 'ahead/forward'\n"
            f"  right    : ({rv['x']:.3f}, {rv['y']:.3f}, {rv['z']:.3f})  ← 'right'\n"
            f"  up       : ({uv['x']:.3f}, {uv['y']:.3f}, {uv['z']:.3f})  ← 'up'\n"
            f"  yaw={pose.get('yaw_deg', 0):.1f}°  pitch={pose.get('pitch_deg', 0):.1f}°\n"
            "SPATIAL SHORTCUTS (pre-computed — use directly, no tool call needed):\n"
            f"  1.5m ahead of you     : {ahead_str}\n"
            f"  1m to your right      : {_off(rv, 1.0)}\n"
            f"  1m to your left       : {_off(rv, -1.0)}\n"
            f"  0.5m above eye level  : {_off(uv, 0.5)}\n"
            f"  1m behind you         : {_off(fv, -1.0)}\n"
            "  For other distances: new_pos = obj.pos + direction_vec × distance (per component)"
        )
    else:
        ctx_parts.append("HEAD POSE: unavailable")

    if pid:
        ctx_parts.append(f"Participant: {pid}")
    if ref_us:
        ctx_parts.append(f"Reference time (when user spoke): {ref_us} µs")

    # Structured move log — machine-readable prior coords for "put it
    # back" / "undo" / "revert" so the model doesn't have to parse free
    # text out of the conversation history.
    if recent_moves:
        move_lines = []
        for obj_id, prev, new in recent_moves:
            move_lines.append(
                f"  {obj_id}: ({prev[0]:.2f}, {prev[1]:.2f}, {prev[2]:.2f}) → "
                f"({new[0]:.2f}, {new[1]:.2f}, {new[2]:.2f})"
            )
        ctx_parts.append("[Recent moves] (most recent last — prev → new)\n" + "\n".join(move_lines))

    # Recent conversation history — lets the agent understand "fix that",
    # "undo", "the sphere I just added", etc.
    if history:
        hist_lines = []
        for u, a in history:
            hist_lines.append(f"  User: {u}")
            hist_lines.append(f"  Agent: {a}")
        ctx_parts.append("[Recent conversation]\n" + "\n".join(hist_lines))

    context = "\n".join(ctx_parts)
    logger.debug("pre-fetched context for turn")
    _trace_log.debug("CTX   {}", context.replace("\n", " | "))
    return SceneContext(text=context, pre_move_positions=pre_move_positions)


__all__ = ["Move", "SceneContext", "build_turn_context"]
