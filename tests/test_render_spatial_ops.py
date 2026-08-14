# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the render worker's deterministic reference resolvers.

The eval tiers exercise these only through a live LLM, where a resolver
regression reads as unexplained score variance; these tests pin the logic
directly.
"""

import pytest
from xr_ai_tools import Tool
from xr_ai_tools.types import EmptyRequest, SpatialFrame, Vector3
from xr_render_demo_worker.spatial_ops import CreationLedger, TurnGuard, _Leaves
from xr_render_scene import (
    AddPrimitiveRequest,
    AddPrimitiveResult,
    MutationResult,
    RemovePrimitiveRequest,
    SceneObject,
    SceneState,
    UpdatePrimitiveRequest,
)
from xr_render_scene import (
    EmptyRequest as SceneEmptyRequest,
)


def _obj(object_id, kind, color, position=(0.0, 1.6, -1.5), size=0.1):
    r, g, b = color
    x, y, z = position
    return SceneObject.model_validate(
        {"id": object_id, "type": kind, "position": {"x": x, "y": y, "z": z},
         "color": {"r": r, "g": g, "b": b}, "size": size}
    )


class _FakeScene:
    def __init__(self, objects):
        self._state = SceneState(objects=list(objects))
        self._objects = {o.id: o for o in objects}
        self.calls = []
        counter = {}
        for o in objects:
            kind = o.id.rsplit("-", 1)[0]
            n = int(o.id.rsplit("-", 1)[1]) if "-" in o.id else 0
            counter[kind] = max(counter.get(kind, -1), n)
        self._counter = counter

    async def get_scene_state(self, req): return self._state
    async def update_primitive(self, req):
        self.calls.append(("update_primitive", req.model_dump(exclude_none=True)))
        if req.obj_id in self._objects:
            cur = self._objects[req.obj_id].model_dump()
            for f in ("x", "y", "z"):
                if getattr(req, f, None) is not None:
                    cur["position"][f] = getattr(req, f)
            for f in ("r", "g", "b"):
                if getattr(req, f, None) is not None:
                    cur["color"][f] = getattr(req, f)
            if req.size is not None:
                cur["size"] = req.size
            self._objects[req.obj_id] = SceneObject.model_validate(cur)
            self._state = SceneState(objects=list(self._objects.values()))
        return MutationResult(ok=True)
    async def remove_primitive(self, req):
        self.calls.append(("remove_primitive", req.model_dump()))
        self._objects.pop(req.obj_id, None)
        self._state = SceneState(objects=list(self._objects.values()))
        return MutationResult(ok=True)
    async def add_primitive(self, req):
        self.calls.append(("add_primitive", req.model_dump()))
        kind = req.prim_type
        n = self._counter.get(kind, -1) + 1
        self._counter[kind] = n
        obj_id = f"{kind}-{n}"
        self._objects[obj_id] = SceneObject.model_validate({
            "id": obj_id, "type": kind,
            "position": {"x": req.x, "y": req.y, "z": req.z},
            "color": {"r": req.r, "g": req.g, "b": req.b}, "size": req.size,
        })
        self._state = SceneState(objects=list(self._objects.values()))
        return AddPrimitiveResult(id=obj_id, ok=True)


_DEFAULT_FRAME = SpatialFrame(
    origin=Vector3(x=0, y=1.6, z=0),
    forward=Vector3(x=0, y=0, z=-1),
    right=Vector3(x=1, y=0, z=0),
    up=Vector3(x=0, y=1, z=0),
)


class _FakeSceneTools:
    def __init__(self, fake):
        self.get_scene_state = Tool(
            "get_scene_state", ".", SceneEmptyRequest, SceneState, fake.get_scene_state)
        self.update_primitive = Tool(
            "update_primitive", ".", UpdatePrimitiveRequest, MutationResult, fake.update_primitive)
        self.add_primitive = Tool(
            "add_primitive", ".", AddPrimitiveRequest, AddPrimitiveResult, fake.add_primitive)
        self.remove_primitive = Tool(
            "remove_primitive", ".", RemovePrimitiveRequest, MutationResult, fake.remove_primitive)


class _FakeTrackingTools:
    def __init__(self, frame=None):
        f = frame or _DEFAULT_FRAME
        self.get_user_frame = Tool("get_user_frame", ".", EmptyRequest, SpatialFrame, lambda _: f)


def _leaves(objects, guard=None, ledger=None):
    fake = _FakeScene(objects)
    scene = _FakeSceneTools(fake)
    tracking = _FakeTrackingTools()
    return _Leaves(scene, tracking, ledger=ledger, guard=guard), fake


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
    leaves, _ = _leaves([_obj("sphere-0", "sphere", (0, 0, 1))], guard=guard, ledger=ledger)
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
    leaves, fake = _leaves([], ledger=ledger)
    # Both calls have the same rounded key → ledger dedupes to one actual add.
    first = await leaves.add("box", (0.001, 1.0, 0.0), (1, 0, 0), 0.1)
    second = await leaves.add("box", (0.004, 1.0, 0.0), (1, 0, 0), 0.1)
    assert first.id == second.id == "box-0"
    assert len([c for c in fake.calls if c[0] == "add_primitive"]) == 1
    assert first.created_this_turn == 1
    ledger.reset()
    third = await leaves.add("box", (0.001, 1.0, 0.0), (1, 0, 0), 0.1)
    assert third.id == "box-1"
