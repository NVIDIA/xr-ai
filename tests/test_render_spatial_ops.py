# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the render worker's deterministic reference resolvers.

The eval tiers exercise these only through a live LLM, where a resolver
regression reads as unexplained score variance; these tests pin the logic
directly.
"""

import pytest
from xr_render_demo_worker.spatial_ops import CreationLedger, TurnGuard, _Leaves
from xr_render_scene import SceneObject, SceneState


class _Fn:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def ainvoke(self, request):
        self.calls.append(request)
        return self.result


def _obj(object_id, kind, color, position=(0.0, 1.6, -1.5), size=0.1):
    r, g, b = color
    x, y, z = position
    return SceneObject.model_validate(
        {"id": object_id, "type": kind, "position": {"x": x, "y": y, "z": z},
         "color": {"r": r, "g": g, "b": b}, "size": size}
    )


def _leaves(objects, guard=None, ledger=None):
    state = SceneState(objects=objects)
    functions = {
        "scene_state__get_scene_state": _Fn(state),
        "scene_updates__update_primitive": _Fn({}),
        "scene_objects__remove_primitive": _Fn({}),
    }
    return _Leaves(functions, ledger=ledger, guard=guard), functions


async def test_exact_id_wins():
    leaves, _ = _leaves([_obj("sphere-0", "sphere", (0, 0, 1))])
    assert (await leaves.find("sphere-0")).id == "sphere-0"


async def test_unicode_dash_id_normalizes():
    leaves, _ = _leaves([_obj("sphere-0", "sphere", (0, 0, 1))])
    assert (await leaves.find("sphere‑0")).id == "sphere-0"


async def test_color_word_resolves_without_shape():
    leaves, _ = _leaves([
        _obj("sphere-0", "sphere", (1, 0, 0)),
        _obj("box-0", "box", (0, 0, 1)),
    ])
    assert (await leaves.find("the red one")).id == "sphere-0"


async def test_everyday_words_do_not_become_shapes():
    # "one"/"thing"/"right" sound like cone/ring; the color must still win.
    leaves, _ = _leaves([
        _obj("sphere-0", "sphere", (1, 0, 0)),
        _obj("sphere-1", "sphere", (0, 0.8, 0)),
    ])
    assert (await leaves.find("the red one on the right")).id == "sphere-0"


async def test_mangled_shape_noun_resolves():
    leaves, _ = _leaves([
        _obj("sphere-0", "sphere", (0, 0.8, 0)),
        _obj("box-0", "box", (0, 0.8, 0)),
    ])
    assert (await leaves.find("green spear")).id == "sphere-0"


async def test_absent_color_reports_back_not_nearest():
    guard = TurnGuard()
    leaves, _ = _leaves(
        [_obj("sphere-0", "sphere", (1, 1, 1)), _obj("sphere-1", "sphere", (0, 0.8, 0))],
        guard=guard,
    )
    with pytest.raises(ValueError, match="blue sphere"):
        await leaves.find("blue sphere")
    assert guard.halted


async def test_color_tie_raises_ambiguous():
    leaves, _ = _leaves([
        _obj("sphere-0", "sphere", (0, 0.8, 0)),
        _obj("sphere-1", "sphere", (0, 0.8, 0)),
    ])
    with pytest.raises(ValueError, match="ambiguous"):
        await leaves.find("green sphere")


async def test_capitalized_id_still_hits_id_branch():
    leaves, _ = _leaves([_obj("box-3", "box", (1, 0, 0)), _obj("box-4", "box", (0, 0.8, 0))])
    assert (await leaves.find("Box-3")).id == "box-3"
    with pytest.raises(ValueError, match="box-3"):
        await leaves.find("Box-9")


async def test_numeric_rgb_ignores_id_bearing_strings():
    leaves, _ = _leaves([_obj("capsule-0", "capsule", (0.25, 0.5, 0.75))])
    assert await leaves.color("same as capsule-0") == (0.25, 0.5, 0.75)
    assert await leaves.color("RGB (1.0, 0.5, 0.0)") == (1.0, 0.5, 0.0)
    # A sign-invalid triple has no letters either; it falls through to the
    # standard default rather than a wrong saturated color.
    assert await leaves.color("-0.5 0.2 0.3") == (0.2, 0.9, 1.0)


async def test_synonym_prefixed_id_resolves():
    leaves, _ = _leaves([_obj("box-39", "box", (1, 1, 0))])
    assert (await leaves.find("cube-39")).id == "box-39"


async def test_id_shaped_miss_lists_ids_without_halting():
    guard = TurnGuard()
    leaves, _ = _leaves([_obj("sphere-0", "sphere", (0, 0, 1))], guard=guard)
    with pytest.raises(ValueError, match="sphere-0"):
        await leaves.find("sphere-9")
    assert not guard.halted


async def test_halt_blocks_moves_but_not_creates():
    guard = TurnGuard()
    ledger = CreationLedger()
    leaves, functions = _leaves([_obj("sphere-0", "sphere", (0, 0, 1))], guard=guard, ledger=ledger)
    functions["scene_objects__add_primitive"] = _Fn(
        type("R", (), {"id": "box-0"})()
    )
    guard.halted = True
    with pytest.raises(ValueError, match="report that failure back"):
        await leaves.write("sphere-0", (0, 0, 0))
    with pytest.raises(ValueError, match="report that failure back"):
        leaves.check_writable()
    created = await leaves.add("box", (0, 1, 0), (1, 0, 0), 0.1)
    assert created.id == "box-0"


def test_shape_words_resolve_and_reject():
    leaves, _ = _leaves([])
    assert leaves.shape("spear") == "sphere"
    assert leaves.shape("kube") == "box"
    assert leaves.shape("cone") == "cone"
    with pytest.raises(ValueError, match="Unknown shape"):
        leaves.shape("xylophone")


async def test_color_words_resolve_fuzzy_default_and_copy():
    leaves, _ = _leaves([_obj("cone-3", "cone", (0.25, 0.5, 0.75))])
    assert await leaves.color("teal") == (0, 0.8, 0.8)
    assert await leaves.color("blew") == (0, 0.4, 1)
    assert await leaves.color("") == (0.2, 0.9, 1.0)
    assert await leaves.color("same as cone-3") == (0.25, 0.5, 0.75)
    assert await leaves.color("normalized RGB (1.0, 0.5, 0.0)") == (1.0, 0.5, 0.0)
    with pytest.raises(ValueError, match="Unknown color"):
        await leaves.color("wibble")


async def test_ledger_dedupes_identical_creates():
    ledger = CreationLedger()
    leaves, functions = _leaves([], ledger=ledger)
    results = iter([type("R", (), {"id": "box-0"})(), type("R", (), {"id": "box-1"})()])

    class _Adder:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, request):
            self.calls.append(request)
            return next(results)

    adder = _Adder()
    functions["scene_objects__add_primitive"] = adder
    first = await leaves.add("box", (0.001, 1.0, 0.0), (1, 0, 0), 0.1)
    second = await leaves.add("box", (0.004, 1.0, 0.0), (1, 0, 0), 0.1)
    assert first.id == second.id == "box-0"
    assert len(adder.calls) == 1
    assert first.created_this_turn == 1
    ledger.reset()
    third = await leaves.add("box", (0.001, 1.0, 0.0), (1, 0, 0), 0.1)
    assert third.id == "box-1"
