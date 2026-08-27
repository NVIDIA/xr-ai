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


class _FakePhysicalColor:
    def __init__(self, color=(0.1, 0.2, 0.3)):
        from xr_render_demo_worker._physical_color import ResolvePhysicalColorRequest
        self.request_model = ResolvePhysicalColorRequest
        self.calls = []
        self._color = color

    async def execute(self, req):
        from xr_render_demo_worker._physical_color import ResolvedColor
        self.calls.append(req.source_words)
        r, g, b = self._color
        return ResolvedColor(r=r, g=g, b=b)


def _leaves_with_camera(objects, color=(0.1, 0.2, 0.3), guard=None):
    fake = _FakeScene(objects)
    physical = _FakePhysicalColor(color)
    leaves = _Leaves(_FakeSceneTools(fake), _FakeTrackingTools(), guard=guard,
                     physical_color=physical)
    return leaves, physical


# ── typed color-source dispatch: each variant touches only its dependency ────

async def test_literal_variant_never_touches_scene_or_camera():
    leaves, physical = _leaves_with_camera([_obj("cone-3", "cone", (0.25, 0.5, 0.75))])
    assert await leaves.resolve_color("literal", "teal") == (0, 0.8, 0.8)
    assert await leaves.resolve_color("literal", "blew") == (0, 0.4, 1)
    assert await leaves.resolve_color("literal", "teel") == (0, 0.8, 0.8)
    assert await leaves.resolve_color("literal", "1.0, 0.5, 0.0") == (1.0, 0.5, 0.0)
    assert await leaves.resolve_color("literal", "") == (0.2, 0.9, 1.0)
    assert await leaves.resolve_color("literal", " ") == (0.2, 0.9, 1.0)
    assert physical.calls == []


async def test_literal_garble_fails_closed():
    leaves, physical = _leaves_with_camera([])
    with pytest.raises(ValueError, match="Unknown color"):
        await leaves.resolve_color("literal", "blerg")
    with pytest.raises(ValueError, match="Unknown color"):
        await leaves.resolve_color("literal", "burnt sienna")
    assert physical.calls == []


async def test_scene_variant_copies_and_never_falls_to_camera():
    guard = TurnGuard()
    leaves, physical = _leaves_with_camera(
        [_obj("cone-0", "cone", (0.25, 0.5, 0.75)), _obj("ring-1", "ring", (1, 0, 0))],
        guard=guard)
    copied = await leaves.resolve_color("scene_object", "cone-0")
    assert copied == (0.25, 0.5, 0.75)
    with pytest.raises(ValueError, match="No scene object matches"):
        await leaves.resolve_color("scene_object", "the wall")
    assert physical.calls == []
    assert guard.halted


async def test_scene_variant_ambiguity_surfaces_ask_back():
    guard = TurnGuard()
    leaves, physical = _leaves_with_camera(
        [_obj("cone-0", "cone", (1, 1, 1)), _obj("cone-1", "cone", (1, 1, 1))], guard=guard)
    with pytest.raises(ValueError, match="ambiguous"):
        await leaves.resolve_color("scene_object", "the cone")
    assert physical.calls == []
    assert guard.halted


async def test_physical_variant_observes_exact_words():
    leaves, physical = _leaves_with_camera([_obj("cone-0", "cone", (1, 1, 1))])
    assert await leaves.resolve_color("physical", "the cone I'm holding") == (0.1, 0.2, 0.3)
    assert await leaves.resolve_color("physical", "the ceiling") == (0.1, 0.2, 0.3)
    assert physical.calls == ["the cone I'm holding", "the ceiling"]


async def test_physical_variant_without_camera_fails_closed():
    leaves, _ = _leaves([])
    with pytest.raises(ValueError, match="no camera"):
        await leaves.resolve_color("physical", "my scarf")


async def test_out_of_range_numeric_literal_fails_closed():
    leaves, physical = _leaves_with_camera([])
    with pytest.raises(ValueError, match="out of range"):
        await leaves.resolve_color("literal", "255 0 0")
    assert physical.calls == []


async def test_empty_value_required_for_non_literal_kinds():
    leaves, physical = _leaves_with_camera([])
    with pytest.raises(ValueError, match="color_value is required"):
        await leaves.resolve_color("scene_object", "")
    with pytest.raises(ValueError, match="color_value is required"):
        await leaves.resolve_color("physical", "  ")
    assert physical.calls == []


async def test_recolor_records_applied_and_satisfied_evidence():
    from xr_render_demo_worker._trace import TurnEvidence, current_turn_evidence
    from xr_render_demo_worker.spatial_ops import _RecolorRequest, make_appearance_tools

    fake = _FakeScene([_obj("cone-0", "cone", (0, 0.8, 0))])
    tools = {tool.name: tool for tool in make_appearance_tools(_FakeSceneTools(fake))}
    evidence = TurnEvidence()
    token = current_turn_evidence.set(evidence)
    try:
        await tools["recolor"].execute(_RecolorRequest(
            object_words="cone-0", color_kind="literal", color_value="green"))
        assert (evidence.applied, evidence.satisfied) == (0, 1)
        assert not any(name == "update_primitive" for name, _ in fake.calls)
        await tools["recolor"].execute(_RecolorRequest(
            object_words="cone-0", color_kind="literal", color_value="red"))
        assert (evidence.applied, evidence.satisfied) == (1, 1)
    finally:
        current_turn_evidence.reset(token)
