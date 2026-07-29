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


# The complete rendered block for the fixture above. The substring tests below
# pin individual sections; this one pins the whole thing, so an added line, a
# reordered section, or whitespace drift fails rather than passing unnoticed.
_GOLDEN_CONTEXT = """\
SCENE OBJECTS:
  sphere_1 (sphere)  pos=(1.00, 1.50, -2.00)  color=(r=1.00 g=0.00 b=0.00)  size=0.250m
HEAD POSE:
  position : (0.00, 1.60, 0.00)
  forward  : (0.000, 0.000, -1.000)  ← 'ahead/forward'
  right    : (1.000, 0.000, 0.000)  ← 'right'
  up       : (0.000, 1.000, 0.000)  ← 'up'
  yaw=0.0°  pitch=0.0°
SPATIAL SHORTCUTS (pre-computed — use directly, no tool call needed):
  1.5m ahead of you     : (0.00, 1.60, -1.50)
  1m to your right      : (1.00, 1.60, 0.00)
  1m to your left       : (-1.00, 1.60, 0.00)
  0.5m above eye level  : (0.00, 2.10, 0.00)
  1m behind you         : (0.00, 1.60, 1.00)
  For other distances: new_pos = obj.pos + direction_vec × distance (per component)
Participant: pid-1
Reference time (when user spoke): 42 µs
[Recent moves] (most recent last — prev → new)
  sphere_1: (0.00, 0.00, 0.00) → (1.00, 1.50, -2.00)
[Recent conversation]
  User: add a sphere
  Agent: Done."""


async def test_build_turn_context_renders_the_exact_expected_block() -> None:
    call_tool, _ = _caller()

    context = await build_turn_context(
        call_tool,
        pid="pid-1",
        ref_us=42,
        recent_moves=[("sphere_1", (0.0, 0.0, 0.0), (1.0, 1.5, -2.0))],
        history=[("add a sphere", "Done.")],
    )

    assert context.text == _GOLDEN_CONTEXT


def test_bundled_system_prompt_resolves() -> None:
    """The bundled prompt must exist at the path the worker and the eval both
    resolve from. The default eval run reads it immediately after argparse, so a
    relocation that leaves this stale fails at run time — this makes it fail in
    CI instead."""
    from xr_render_demo_worker import PROMPTS_DIR, SYSTEM_PROMPT

    assert SYSTEM_PROMPT.is_file(), f"bundled system prompt missing: {SYSTEM_PROMPT}"
    assert SYSTEM_PROMPT.read_text(encoding="utf-8").strip()
    # The two companion prompts ship alongside it.
    for name in ("quick_ack.txt", "still_working.txt"):
        assert (PROMPTS_DIR / name).is_file(), f"bundled prompt missing: {name}"


def test_importing_a_seam_does_not_pull_the_pipeline() -> None:
    """Importing one seam must not drag in Pipecat, NAT, or the model clients.

    That isolation is the point of the extraction: ``scene`` is unit-testable
    precisely because it needs none of them. A package ``__init__`` that imported
    ``app`` would quietly undo it, so this asserts the seam stays cheap.
    """
    import subprocess
    import sys as _sys

    code = (
        "import sys; "
        "sys.path.insert(0, %r); "
        "import xr_render_demo_worker.scene; "
        "heavy = [m for m in ('pipecat', 'nat', 'xr_ai_models', 'xr_ai_pipecat', 'xr_ai_nat') "
        "         if m in sys.modules]; "
        "print(','.join(heavy))" % str(_WORKER_DIR)
    )
    out = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "", f"importing scene pulled in heavy modules: {out}"
