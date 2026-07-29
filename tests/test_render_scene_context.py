# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the render worker's per-turn scene context builder.

``scene.build_turn_context`` pre-fetches scene state, head pose, and the
1.5 m-ahead position, then renders the block injected into the reasoning loop.
It is exercised directly here (no pipeline, no LLM) so the context contract is
pinned independently of the turn loop that consumes it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKER_DIR = (
    Path(__file__).resolve().parent.parent
    / "agent-samples" / "xr-render-demo" / "worker"
)
sys.path.insert(0, str(_WORKER_DIR))

from xr_render_demo_worker.scene import build_turn_context  # noqa: E402

_SCENE = {
    "objects": [
        {
            "id": "sphere_1",
            "type": "sphere",
            "position": {"x": 1.0, "y": 1.5, "z": -2.0},
            "color": {"r": 1.0, "g": 0.0, "b": 0.0},
            "size": 0.25,
        }
    ]
}

_POSE = {
    "is_valid": True,
    "position": {"x": 0.0, "y": 1.6, "z": 0.0},
    "forward": {"x": 0.0, "y": 0.0, "z": -1.0},
    "right": {"x": 1.0, "y": 0.0, "z": 0.0},
    "up": {"x": 0.0, "y": 1.0, "z": 0.0},
    "yaw_deg": 0.0,
    "pitch_deg": 0.0,
}


def _caller(scene=_SCENE, pose=_POSE, ahead=None):
    """Native-tool invoker double recording the pre-fetch calls."""
    calls: list[str] = []

    async def call_tool(tool: str, args: dict, *, silent: bool = False):
        calls.append(tool)
        return {
            "get_scene_state": scene,
            "get_head_pose": pose,
            "position_ahead": ahead if ahead is not None else {"x": 0.0, "y": 1.6, "z": -1.5},
        }[tool]

    return call_tool, calls


async def test_build_turn_context_prefetches_and_renders_scene_and_pose() -> None:
    call_tool, calls = _caller()

    context = await build_turn_context(call_tool, pid="pid-1", ref_us=42)

    # All three pre-fetches happen (they run concurrently, so order is not fixed).
    assert sorted(calls) == ["get_head_pose", "get_scene_state", "position_ahead"]
    # Scene objects, head pose, participant, and reference time are rendered.
    assert "sphere_1 (sphere)" in context.text
    assert "HEAD POSE:" in context.text
    assert "Participant: pid-1" in context.text
    assert "Reference time (when user spoke): 42 µs" in context.text
    # The scene snapshot is returned so the caller can diff moves after the turn.
    assert context.pre_move_positions == {"sphere_1": (1.0, 1.5, -2.0)}


async def test_build_turn_context_handles_empty_scene_and_invalid_pose() -> None:
    call_tool, _ = _caller(scene={"objects": []}, pose={"is_valid": False})

    context = await build_turn_context(call_tool, pid="", ref_us=0)

    assert "SCENE OBJECTS: (empty)" in context.text
    assert "HEAD POSE: unavailable" in context.text
    assert context.pre_move_positions == {}
    # Optional sections are omitted when the caller has no history.
    assert "[Recent moves]" not in context.text
    assert "[Recent conversation]" not in context.text


async def test_build_turn_context_renders_move_log_and_history() -> None:
    call_tool, _ = _caller()

    context = await build_turn_context(
        call_tool,
        pid="pid-1",
        ref_us=1,
        recent_moves=[("sphere_1", (0.0, 0.0, 0.0), (1.0, 1.5, -2.0))],
        history=[("add a sphere", "Done.")],
    )

    assert "[Recent moves] (most recent last — prev → new)" in context.text
    assert "sphere_1: (0.00, 0.00, 0.00) → (1.00, 1.50, -2.00)" in context.text
    assert "[Recent conversation]" in context.text
    assert "  User: add a sphere" in context.text
    assert "  Agent: Done." in context.text


async def test_build_turn_context_falls_back_when_position_ahead_unavailable() -> None:
    # position_ahead can fail independently; the block still renders using the
    # forward vector so the model keeps its pre-computed shortcut.
    call_tool, _ = _caller(ahead={"error": "unavailable"})

    context = await build_turn_context(call_tool, pid="", ref_us=0)

    assert "1.5m ahead of you     : (0.00, 1.60, -1.50)" in context.text
