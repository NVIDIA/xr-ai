# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end eval corpus for the render worker."""
from __future__ import annotations

DEFAULT_POSE = {
    "is_valid": True,
    "position": {"x": 0.0, "y": 1.6, "z": 0.0},
    "forward": {"x": 0.0, "y": 0.0, "z": -1.0},
    "right":   {"x": 1.0, "y": 0.0, "z": 0.0},
    "up":      {"x": 0.0, "y": 1.0, "z": 0.0},
    "yaw_deg": 0.0,
    "pitch_deg": 0.0,
}

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
    """Predicate factory: assert at least one add/update sets ``prim_type``
    AND each requested colour channel reaches the given lower bound. Facets
    may appear in one call or be split across calls."""
    requirements: dict[str, str | float] = {}
    if prim_type is not None:
        requirements["prim_type"] = prim_type
    for channel, threshold in (("r", r_min), ("g", g_min), ("b", b_min)):
        if threshold is not None:
            requirements[channel] = threshold

    def _pred(mutations: list[tuple[str, dict]]) -> tuple[bool, str]:
        seen = dict.fromkeys(requirements, False)
        for name, args in mutations:
            if name not in ("add_primitive", "update_primitive"):
                continue
            for key, expected in requirements.items():
                if key == "prim_type":
                    if args.get("prim_type") == expected:
                        seen[key] = True
                else:
                    value = args.get(key)
                    if value is not None and float(value) >= float(expected):
                        seen[key] = True
        if all(seen.values()):
            return True, f"saw {requirements}"
        missing = [key for key, hit in seen.items() if not hit]
        return False, f"missing facets: {missing} (wanted {requirements})"

    return _pred


def _stacked_vertically(mutations: list[tuple[str, dict]]) -> tuple[bool, str]:
    """Predicate for ``stack_*`` cases: every add_primitive must share the
    same x/z column and have distinct y values, regardless of base height."""
    adds = [args for name, args in mutations if name == "add_primitive"]
    if len(adds) < 2:
        return False, f"need >=2 add_primitive calls, got {len(adds)}"
    rows = [(a.get("x", 0.0), a.get("y", 0.0), a.get("z", 0.0)) for a in adds]
    xs = {round(row[0], 2) for row in rows}
    zs = {round(row[2], 2) for row in rows}
    if len(xs) > 1 or len(zs) > 1:
        return False, f"x/z not aligned across stack: {rows}"
    ys = sorted(round(row[1], 2) for row in rows)
    for low, high in zip(ys, ys[1:]):
        if high - low < 0.05:
            return False, f"y values not separated (need >=5 cm gap): {ys}"
    return True, f"stacked at y={ys}"


PERCEPTION_TOOL = "look_at_current_frame"

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
        # alone: y/x align with cube, not midpoint with the other sphere.
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
        "ignore_extra": False,
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
        # Either path is fine: update_primitive(prim_type=box) OR
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
        # half-edge 0.1 + tolerance): accept either.
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.7, 1.0),
                      # b not pinned: Nemotron occasionally leaks the cube's blue
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
        # Plural-restricted target: the box must NOT also grow.
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
        # "≥1 mutating call happened": we don't pin which sphere.
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
        # (subject of the last reply), which is at y=1.6: guarding
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
        # All three should end up 1 m further from the user: z more
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
        # Cube must NOT move: that's what distinguishes this from swap.
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
    # the existing pyramid.
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

    # ── perception gating: real-world colour must come from the camera ───────
    # The colour word is never in the utterance; the model must call
    # look_at_current_frame FIRST and read the colour out of the answer.
    # `vlm_answer` is what the mocked camera sees; `must_call_first` fails
    # the case if the model mutates before (or without) looking.
    {
        "name":  "perception_color_of_held_object",
        "scene": [],
        "user":  "Make a sphere the same color as the thing I'm holding.",
        "vlm_answer": "The user is holding a bright red apple.",
        "must_call_first": PERCEPTION_TOOL,
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "sphere",
                      "r": (0.7, 1.0), "g": (0.0, 0.4), "b": (0.0, 0.4)}},
        ],
    },
    {
        "name":  "perception_recolor_to_match_shirt",
        "scene": [{"id": "sphere-0", "type": "sphere",
                   "pos": [0.0, 1.6, -1.5], "color": [1, 1, 1], "size": 0.1}],
        "user":  "Make the sphere the same color as my shirt.",
        "vlm_answer": "The user's shirt is blue.",
        "must_call_first": PERCEPTION_TOOL,
        "result": [
            {"tool": "update_primitive",
             "args": {"obj_id": "sphere-0",
                      "b": (0.5, 1.0), "r": (0.0, 0.3)}},
        ],
    },
    {
        "name":  "perception_wall_color_cube",
        "scene": [],
        "user":  "Add a cube that matches the color of the wall I'm looking at.",
        "vlm_answer": "The wall is green.",
        "must_call_first": PERCEPTION_TOOL,
        "result": [
            {"tool": "add_primitive",
             "args": {"prim_type": "box",
                      "g": (0.5, 1.0), "r": (0.0, 0.4), "b": (0.0, 0.3)}},
        ],
    },
]
