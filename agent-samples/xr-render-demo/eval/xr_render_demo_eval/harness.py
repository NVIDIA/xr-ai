# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the real native-NAT render workflow against an in-memory XR scene."""

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import ConfigDict, Field
from xr_ai_models import load_models_config, make_llm
from xr_ai_nat.functions.spatial_math import SpatialMathFunctionsConfig
from xr_ai_nat.functions.text_memory import ConversationEntry, RecallConversationRequest, RecallConversationResult
from xr_ai_nat.functions.types import SpatialFrame, Vector3
from xr_ai_nat.functions.vision import HistoricalVisionRequest, LiveVisionRequest, LiveVisionResult
from xr_ai_nat.functions.xr_tracking import HeadPose, HeadPoseRequest
from xr_render_demo_worker.config import load_config
from xr_render_demo_worker.models import SceneRequest
from xr_render_demo_worker.scene import SceneContext
from xr_render_demo_worker.supervisor import scene_supervisor
from xr_render_scene import (
    AddPrimitiveRequest,
    AddPrimitiveResult,
    EmptyRequest,
    MutationResult,
    RemovePrimitiveRequest,
    SceneObject,
    SceneState,
    UpdatePrimitiveRequest,
)

from .cases import CASES as CORPUS_CASES

_HERE = Path(__file__).resolve().parent
_CONFIG = load_config((_HERE / "../../yaml/xr_render_demo_worker.yaml").resolve())
_PARTICIPANT = "eval-user"

_DEFAULT_POSE = {
    "is_valid": True,
    "position": {"x": 0.0, "y": 1.6, "z": 0.0},
    "forward": {"x": 0.0, "y": 0.0, "z": -1.0},
    "right": {"x": 1.0, "y": 0.0, "z": 0.0},
    "up": {"x": 0.0, "y": 1.0, "z": 0.0},
    "yaw_deg": 0.0,
    "pitch_deg": 0.0,
    "ts": 1,
}


class _EvalSceneStateConfig(FunctionGroupBaseConfig, name="xr_render_eval_scene_state"):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scene: Any = Field(exclude=True, repr=False)


@register_function_group(config_type=_EvalSceneStateConfig)
async def _eval_scene_state(config: _EvalSceneStateConfig, _builder: Builder):
    group = FunctionGroup(config=config)
    group.add_function(
        "get_scene_state",
        config.scene.get_scene_state,
        description="Return the scene.",
    )
    yield group


class _EvalSceneUpdatesConfig(FunctionGroupBaseConfig, name="xr_render_eval_scene_updates"):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scene: Any = Field(exclude=True, repr=False)


@register_function_group(config_type=_EvalSceneUpdatesConfig)
async def _eval_scene_updates(config: _EvalSceneUpdatesConfig, _builder: Builder):
    group = FunctionGroup(config=config)
    group.add_function(
        "update_primitive",
        config.scene.update_primitive,
        description="Update an object.",
    )
    yield group


class _EvalSceneObjectsConfig(FunctionGroupBaseConfig, name="xr_render_eval_scene_objects"):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scene: Any = Field(exclude=True, repr=False)


@register_function_group(config_type=_EvalSceneObjectsConfig)
async def _eval_scene_objects(config: _EvalSceneObjectsConfig, _builder: Builder):
    group = FunctionGroup(config=config)
    group.add_function("add_primitive", config.scene.add_primitive, description="Add an object.")
    group.add_function("remove_primitive", config.scene.remove_primitive, description="Remove an object.")
    yield group


class _EvalTrackingConfig(FunctionGroupBaseConfig, name="xr_render_eval_tracking"):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tracking: Any = Field(exclude=True, repr=False)


@register_function_group(config_type=_EvalTrackingConfig)
async def _eval_tracking(config: _EvalTrackingConfig, _builder: Builder):
    group = FunctionGroup(config=config)
    group.add_function(
        "get_user_frame",
        config.tracking.get_user_frame,
        description="Return a test user frame.",
    )
    yield group


class _EvalVisionConfig(FunctionGroupBaseConfig, name="xr_render_eval_vision"):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scene: Any = Field(exclude=True, repr=False)


@register_function_group(config_type=_EvalVisionConfig)
async def _eval_vision(config: _EvalVisionConfig, _builder: Builder):
    async def look(request: LiveVisionRequest) -> LiveVisionResult:
        config.scene.calls.append(("look_at_current_frame", request.model_dump()))
        if config.scene.vision_error:
            raise RuntimeError(config.scene.vision_error)
        # A real VLM always says something; an empty fixture must not derail
        # the rollout, only decline to reward the spurious look.
        return LiveVisionResult(answer=config.scene.vision_answer or "Nothing notable is visible.")

    async def look_past(request: HistoricalVisionRequest) -> LiveVisionResult:
        config.scene.calls.append(("look_at_past_frame", request.model_dump()))
        if config.scene.vision_error:
            raise RuntimeError(config.scene.vision_error)
        return LiveVisionResult(answer=config.scene.vision_answer or "Nothing notable is visible.")

    group = FunctionGroup(config=config)
    group.add_function(
        "look_at_current_frame",
        look,
        description="Observe the user's live camera.",
    )
    group.add_function(
        "look_at_past_frame",
        look_past,
        description="Observe a recorded camera frame.",
    )
    yield group


class _EvalConversationConfig(FunctionGroupBaseConfig, name="xr_render_eval_conversations"):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scene: Any = Field(exclude=True, repr=False)


@register_function_group(config_type=_EvalConversationConfig)
async def _eval_conversations(config: _EvalConversationConfig, _builder: Builder):
    async def recall(request: RecallConversationRequest) -> RecallConversationResult:
        config.scene.calls.append(("recall_conversation", request.model_dump()))
        entries = []
        for index, (user_text, agent_text) in enumerate(config.scene.history):
            entries.append(ConversationEntry(timestamp_us=(index + 1) * 1_000_000, role="user", text=user_text))
            entries.append(ConversationEntry(timestamp_us=(index + 1) * 1_000_000 + 1, role="agent", text=agent_text))
        if config.scene.memory_answer:
            entries.append(ConversationEntry(timestamp_us=1, role="agent", text=config.scene.memory_answer))
        return RecallConversationResult(entries=entries)

    group = FunctionGroup(config=config)
    group.add_function(
        "recall_conversation",
        recall,
        description="Recall older conversation.",
    )
    yield group


@dataclass(frozen=True)
class Case:
    name: str
    request: str
    scene: tuple[dict[str, Any], ...] = ()
    pose: HeadPose | None = None
    vision: str = ""
    vision_error: str = ""
    required_tools: frozenset[str] = frozenset()
    forbidden_tools: frozenset[str] = frozenset()
    required_order: tuple[str, ...] = ()
    expected_call_counts: tuple[tuple[str, int], ...] = ()
    expected_sizes: tuple[tuple[str, float], ...] = ()
    expected_colors: tuple[tuple[str, tuple[float, float, float]], ...] = ()
    expected_positions: tuple[tuple[str, tuple[float, float, float]], ...] = ()
    memory: str = ""
    history: tuple[tuple[str, str], ...] = ()


CASES = (
    Case(
        name="create_object",
        request="Add a blue sphere.",
        required_tools=frozenset({"add_primitive"}),
        expected_colors=(("sphere-0", (0.0, 0.4, 1.0)),),
    ),
    Case(
        name="delete_object",
        request="Remove the white cylinder.",
        scene=(
            {
                "id": "cylinder-0",
                "type": "cylinder",
                "position": {"x": -0.3, "y": 1.3, "z": -1.2},
                "color": {"r": 1, "g": 1, "b": 1},
                "size": 0.15,
            },
        ),
        required_tools=frozenset({"remove_primitive"}),
    ),
    Case(
        name="place_existing_object",
        request="Move the blue box to my left.",
        scene=(
            {
                "id": "box-0",
                "type": "box",
                "position": {"x": 0, "y": 1.6, "z": -1.5},
                "color": {"r": 0, "g": 0.4, "b": 1},
                "size": 0.1,
            },
        ),
        required_tools=frozenset({"get_user_frame", "update_primitive"}),
        required_order=("get_user_frame", "update_primitive"),
    ),
    Case(
        name="vision_then_appearance",
        request="Make the cone the color of the wall.",
        scene=(
            {
                "id": "cone-0",
                "type": "cone",
                "position": {"x": 0, "y": 1.5, "z": -1.4},
                "color": {"r": 0.2, "g": 0.9, "b": 1},
                "size": 0.1,
            },
        ),
        vision="The wall is orange: normalized RGB (1.0, 0.5, 0.0).",
        required_tools=frozenset({"look_at_current_frame", "update_primitive"}),
    ),
    Case(
        name="compound_object_and_placement",
        request="Create a green box and place it one metre to my right.",
        required_tools=frozenset({"add_primitive", "get_user_frame"}),
        forbidden_tools=frozenset({"update_primitive"}),
        required_order=("get_user_frame", "add_primitive"),
        expected_positions=(("box-0", (1.0, 1.6, 0.0)),),
    ),
    Case(
        name="create_in_front_of_moved_user",
        request="Add a yellow cone ahead of me.",
        pose=HeadPose(
            is_valid=True,
            position=Vector3(x=2.0, y=1.6, z=1.5),
            forward=Vector3(x=0, y=0, z=-1),
            right=Vector3(x=1, y=0, z=0),
            up=Vector3(x=0, y=1, z=0),
            yaw_deg=0,
            pitch_deg=0,
            ts=1,
        ),
        required_tools=frozenset({"add_primitive", "get_user_frame"}),
        forbidden_tools=frozenset({"update_primitive"}),
        required_order=("get_user_frame", "add_primitive"),
        expected_positions=(("cone-0", (2.0, 1.6, 0.0)),),
    ),
    Case(
        name="resize_existing_object",
        request="Make the pyramid 1.75 times larger.",
        scene=(
            {
                "id": "pyramid-0",
                "type": "pyramid",
                "position": {"x": 0.2, "y": 1.2, "z": -1.1},
                "color": {"r": 0.4, "g": 0.4, "b": 0.4},
                "size": 0.2,
            },
        ),
        required_tools=frozenset({"update_primitive"}),
        expected_sizes=(("pyramid-0", 0.35),),
    ),
    Case(
        name="historical_vision",
        request="What color was the object I held ten seconds ago?",
        vision="The previously held object was purple.",
        required_tools=frozenset({"look_at_past_frame"}),
    ),
    Case(
        name="durable_memory",
        request="What object did we discuss in the earlier session?",
        memory="We discussed a small cyan sphere.",
        required_tools=frozenset({"recall_conversation"}),
    ),
    Case(
        name="placement_despite_camera_off",
        request="Put a red sphere two meters ahead of me.",
        vision_error="No current camera frame is available.",
        required_tools=frozenset({"add_primitive"}),
    ),
    Case(
        name="camera_meta_comment",
        request="No, I meant the webcam discussion, not a request.",
        forbidden_tools=frozenset({"look_at_current_frame", "look_at_past_frame"}),
    ),
    Case(
        name="unavailable_live_camera",
        request="Read the maker's mark on the cup in front of me.",
        vision_error="No current camera frame is available.",
        required_tools=frozenset({"look_at_current_frame"}),
        forbidden_tools=frozenset({"look_at_past_frame"}),
        expected_call_counts=(("look_at_current_frame", 1),),
    ),
)


def _scene_object(item: dict[str, Any]) -> SceneObject:
    x, y, z = item["pos"]
    r, g, b = item["color"]
    return SceneObject.model_validate(
        {
            "id": item["id"],
            "type": item["type"],
            "position": {"x": x, "y": y, "z": z},
            "color": {"r": r, "g": g, "b": b},
            "size": item["size"],
        }
    )


@dataclass
class FakeScene:
    objects: dict[str, SceneObject]
    pose: HeadPose
    vision_answer: str
    vision_error: str
    memory_answer: str
    history: tuple[tuple[str, str], ...] = ()
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_case(cls, case: Case) -> "FakeScene":
        objects = [SceneObject.model_validate(item) for item in case.scene]
        pose = case.pose or HeadPose.model_validate(_DEFAULT_POSE)
        return cls(
            {item.id: item for item in objects},
            pose,
            case.vision,
            case.vision_error,
            case.memory,
            case.history,
        )

    @classmethod
    def from_corpus_case(cls, case: dict[str, Any]) -> "FakeScene":
        objects = [_scene_object(item) for item in case.get("scene", ())]
        pose = HeadPose.model_validate({**_DEFAULT_POSE, **(case.get("pose") or {})})
        return cls(
            {item.id: item for item in objects},
            pose,
            case.get("vlm_answer", ""),
            "",
            case.get("memory", ""),
            tuple(case.get("history", ())),
        )

    async def add_primitive(self, request: AddPrimitiveRequest) -> AddPrimitiveResult:
        arguments = request.model_dump()
        self.calls.append(("add_primitive", arguments))
        # Mirror the engine's per-type monotonic counters; length-based ids
        # collide after removals.
        for item in self.objects.values():
            kind, _, index = item.id.rpartition("-")
            if kind and index.isdigit():
                self.counters[kind] = max(self.counters.get(kind, 0), int(index) + 1)
        number = self.counters.get(request.prim_type, 0)
        self.counters[request.prim_type] = number + 1
        object_id = f"{request.prim_type}-{number}"
        self.objects[object_id] = SceneObject.model_validate(
            {
                "id": object_id,
                "type": request.prim_type,
                "position": {"x": request.x, "y": request.y, "z": request.z},
                "color": {"r": request.r, "g": request.g, "b": request.b},
                "size": request.size,
            }
        )
        return AddPrimitiveResult(id=object_id, ok=True)

    async def update_primitive(self, request: UpdatePrimitiveRequest) -> MutationResult:
        arguments = request.model_dump(exclude_none=True)
        if request.obj_id not in self.objects:
            self.calls.append(("update_primitive", arguments))
            return MutationResult(ok=False, reason=f"no object {request.obj_id!r}")
        self.calls.append(("update_primitive", arguments))
        current = self.objects[request.obj_id].model_dump()
        for field_name in ("x", "y", "z"):
            if field_name in arguments:
                current["position"][field_name] = arguments[field_name]
        for field_name in ("r", "g", "b"):
            if field_name in arguments:
                current["color"][field_name] = arguments[field_name]
        if request.prim_type is not None:
            current["type"] = request.prim_type
        if request.size is not None:
            current["size"] = request.size
        self.objects[request.obj_id] = SceneObject.model_validate(current)
        return MutationResult(ok=True)

    async def remove_primitive(self, request: RemovePrimitiveRequest) -> MutationResult:
        self.calls.append(("remove_primitive", request.model_dump()))
        self.objects.pop(request.obj_id, None)
        return MutationResult(ok=True)

    async def get_scene_state(self, request: EmptyRequest) -> SceneState:
        self.calls.append(("get_scene_state", {}))
        return SceneState(objects=list(self.objects.values()))

    async def bind(self, builder: Builder) -> None:
        scene = self

        class Tracking:
            async def get_user_frame(self, request: HeadPoseRequest) -> SpatialFrame:
                scene.calls.append(("get_user_frame", {}))
                return SpatialFrame(
                    origin=scene.pose.position,
                    forward=scene.pose.forward,
                    right=scene.pose.right,
                    up=scene.pose.up,
                )

        await builder.add_function_group("scene_state", _EvalSceneStateConfig(scene=self))
        await builder.add_function_group("scene_updates", _EvalSceneUpdatesConfig(scene=self))
        await builder.add_function_group("scene_objects", _EvalSceneObjectsConfig(scene=self))
        await builder.add_function_group("tracking", _EvalTrackingConfig(tracking=Tracking()))
        await builder.add_function_group("spatial", SpatialMathFunctionsConfig())
        await builder.add_function_group("vision", _EvalVisionConfig(scene=self))
        await builder.add_function_group("conversations", _EvalConversationConfig(scene=self))


_MUTATING = frozenset({"add_primitive", "update_primitive", "remove_primitive"})

_SCENE_ARG_LOOKUP = {
    "x": ("pos", 0),
    "y": ("pos", 1),
    "z": ("pos", 2),
    "r": ("color", 0),
    "g": ("color", 1),
    "b": ("color", 2),
    "size": ("size", None),
    "prim_type": ("type", None),
}


def _resolve_arg(obj_id: str | None, key: str, scene: list[dict[str, Any]]) -> Any:
    lookup = _SCENE_ARG_LOOKUP.get(key)
    item = next((entry for entry in scene if entry.get("id") == obj_id), None)
    if lookup is None or item is None:
        return None
    source, index = lookup
    value = item.get(source)
    if index is None or value is None:
        return value
    return value[index] if index < len(value) else None


def _match_call(name: str, args: dict[str, Any], expect: dict[str, Any], scene: list[dict[str, Any]]) -> bool:
    if name != expect["tool"]:
        return False
    for key, want in expect.get("args", {}).items():
        got = args.get(key)
        # An update that omits a field keeps the object's initial value.
        if got is None and name == "update_primitive":
            got = _resolve_arg(args.get("obj_id"), key, scene)
        if got is None:
            return False
        if isinstance(want, tuple):
            try:
                got = float(got)
            except (TypeError, ValueError):
                return False
            low, high = want
            if not (low <= got <= high):
                return False
        elif got != want:
            return False
    return True


def check_corpus(calls: list[tuple[str, dict[str, Any]]], case: dict[str, Any]) -> tuple[bool, str]:
    """Score one corpus rollout: perception gating, then order-independent
    mutation matching with the duplicate-add rule."""
    names = [name for name, _args in calls]
    mutations = [(name, args) for name, args in calls if name in _MUTATING]
    if first_tool := case.get("must_call_first"):
        if first_tool not in names:
            return False, f"{first_tool} was never called"
        first_mutation = next((index for index, name in enumerate(names) if name in _MUTATING), None)
        if first_mutation is not None and names.index(first_tool) > first_mutation:
            return False, f"{first_tool} called after the first mutation"
    wanted = list(case.get("result") or ())
    if not wanted and not mutations:
        return False, f"no mutating calls: {names}"
    expected_adds = sum(1 for expect in wanted if expect["tool"] == "add_primitive")
    actual_adds = sum(1 for name, _args in mutations if name == "add_primitive")
    if wanted and actual_adds > expected_adds:
        return False, f"duplicate add: {actual_adds} add_primitive calls for {expected_adds} expected"
    scene = list(case.get("scene", ()))
    remaining = list(mutations)
    unmatched = []
    for expect in wanted:
        for index, (name, args) in enumerate(remaining):
            if _match_call(name, args, expect, scene):
                remaining.pop(index)
                break
        else:
            unmatched.append(expect)
    if unmatched:
        wanted_desc = "; ".join(f"{expect['tool']}({expect.get('args', {})})" for expect in unmatched)
        actual = [f"{name}({args})" for name, args in mutations]
        return False, f"unmatched: {wanted_desc} | actual mutations: {actual} | calls: {names}"
    if not case.get("ignore_extra", True) and remaining:
        return False, f"extra mutating calls: {[name for name, _args in remaining]}"
    if (predicate := case.get("predicate")) is not None:
        ok, message = predicate(mutations)
        if not ok:
            return False, f"predicate failed: {message}"
    return True, "ok"


async def run_corpus_case(case: dict[str, Any]) -> bool:
    scene = FakeScene.from_corpus_case(case)
    llm = make_llm(load_models_config(_CONFIG.models_yaml), "agent_llm")
    try:
        async with WorkflowBuilder() as builder:
            await scene.bind(builder)
            scene_group = await builder.get_function_group("scene_state")
            scene_functions = await scene_group.get_all_functions()
            tracking_group = await builder.get_function_group("tracking")
            tracking_functions = await tracking_group.get_all_functions()
            context = SceneContext(
                scene_functions["scene_state__get_scene_state"],
                tracking_functions["tracking__get_user_frame"],
            )
            context._recent_moves[_PARTICIPANT] = [
                f"{object_id}: previously at {before}, now at {after}"
                for object_id, before, after in case.get("recent_moves", ())
            ]
            supervisor = await scene_supervisor(
                builder=builder,
                llm=llm,
                context=context,
            )
            # A crashed rollout is a case result, not an eval abort: score
            # whatever calls it made before failing.
            try:
                reply = await supervisor.ainvoke(
                    SceneRequest(
                        transcript=case["user"],
                        participant_id=_PARTICIPANT,
                        timestamp_us=10_000_000,
                    )
                )
                response = reply.response
            except Exception as exc:
                response = f"<workflow error: {exc}>"
    finally:
        await llm.close()
    ok, why = check_corpus(scene.calls, case)
    status = "PASS" if ok else f"FAIL {why}"
    print(f"{status:32} {case['name']}: {response}", flush=True)
    return ok


# The basics battery: the most common utterances, their observed live STT
# corruptions, and history-bearing variants. Run after EVERY prompt or ops
# change (`uv run xr_render_demo_eval basics`); prompt edits keep breaking exactly these through
# example contamination, and the full corpus hides one-case damage inside its
# run-to-run variance. Utterances must stay disjoint from prompt examples.
_BASICS_SCENE = (
    {"id": "box-0", "type": "box", "position": {"x": 0.6, "y": 1.3, "z": -1.1},
     "color": {"r": 0, "g": 1, "b": 1}, "size": 0.1},
    {"id": "sphere-1", "type": "sphere", "position": {"x": -0.5, "y": 1.5, "z": -1.3},
     "color": {"r": 0, "g": 0.8, "b": 0}, "size": 0.1},
)
_BASICS_HISTORY = (
    ("Add a cyan cube.", "Added a cyan cube."),
    ("Make a green sphere.", "Created a green sphere."),
)
_NO_MUTATION = _MUTATING

BASICS = (
    Case(
        name="basics_create_cube",
        request="Make a red cube.",
        required_tools=frozenset({"add_primitive"}),
        expected_colors=(("box-0", (1.0, 0.0, 0.0)),),
        expected_positions=(("box-0", (0.0, 1.6, -1.5)),),
    ),
    Case(
        name="basics_create_sphere",
        request="Add a blue sphere.",
        required_tools=frozenset({"add_primitive"}),
        expected_colors=(("sphere-0", (0.0, 0.4, 1.0)),),
        expected_positions=(("sphere-0", (0.0, 1.6, -1.5)),),
    ),
    Case(
        name="basics_garbled_shape",
        request="Make a blue spear.",
        required_tools=frozenset({"add_primitive"}),
        expected_colors=(("sphere-0", (0.0, 0.4, 1.0)),),
        expected_positions=(("sphere-0", (0.0, 1.6, -1.5)),),
    ),
    Case(
        name="basics_garbled_shape_cute",
        request="Add a red cute.",
        required_tools=frozenset({"add_primitive"}),
        expected_colors=(("box-0", (1.0, 0.0, 0.0)),),
    ),
    Case(
        name="basics_garbled_color",
        request="Add a blew sphere.",
        required_tools=frozenset({"add_primitive"}),
        expected_colors=(("sphere-0", (0.0, 0.4, 1.0)),),
    ),
    Case(
        name="basics_stated_distance",
        request="Create a purple sphere two meters ahead of me.",
        required_tools=frozenset({"add_primitive"}),
        expected_positions=(("sphere-0", (0.0, 1.6, -2.0)),),
    ),
    Case(
        name="basics_create_cube_with_history",
        request="Make a red cube.",
        scene=_BASICS_SCENE,
        history=_BASICS_HISTORY,
        required_tools=frozenset({"add_primitive"}),
        forbidden_tools=frozenset({"update_primitive", "remove_primitive"}),
        expected_colors=(("box-1", (1.0, 0.0, 0.0)),),
        expected_positions=(("box-1", (0.0, 1.6, -1.5)),),
    ),
    Case(
        name="basics_garbled_shape_with_history",
        request="Make a blue spear.",
        scene=_BASICS_SCENE,
        history=_BASICS_HISTORY,
        required_tools=frozenset({"add_primitive"}),
        forbidden_tools=frozenset({"update_primitive", "remove_primitive"}),
        expected_colors=(("sphere-2", (0.0, 0.4, 1.0)),),
    ),
    Case(
        name="basics_recolor_with_history",
        request="Make the sphere red.",
        scene=_BASICS_SCENE,
        history=_BASICS_HISTORY,
        required_tools=frozenset({"update_primitive"}),
        forbidden_tools=frozenset({"add_primitive", "remove_primitive"}),
        expected_colors=(("sphere-1", (1.0, 0.0, 0.0)),),
    ),
    Case(
        name="basics_move_with_history",
        request="Move the cube to my left.",
        scene=_BASICS_SCENE,
        history=_BASICS_HISTORY,
        required_tools=frozenset({"update_primitive"}),
        forbidden_tools=frozenset({"add_primitive", "remove_primitive"}),
    ),
    Case(
        name="basics_remove_with_history",
        request="Remove the cube.",
        scene=_BASICS_SCENE,
        history=_BASICS_HISTORY,
        required_tools=frozenset({"remove_primitive"}),
        forbidden_tools=frozenset({"add_primitive"}),
    ),
    Case(
        name="basics_resize_with_history",
        request="Double the size of the sphere.",
        scene=_BASICS_SCENE,
        history=_BASICS_HISTORY,
        required_tools=frozenset({"update_primitive"}),
        forbidden_tools=frozenset({"add_primitive", "remove_primitive"}),
        expected_sizes=(("sphere-1", 0.2),),
    ),
    # Systematic perturbation classes, not observed-corruption harvesting:
    # dropped words, number homophones, filler injection, run-on merges.
    Case(
        name="basics_dropped_color",
        request="Make a, uh, sphere.",
        required_tools=frozenset({"add_primitive"}),
        expected_positions=(("sphere-0", (0.0, 1.6, -1.5)),),
    ),
    Case(
        name="basics_number_homophone",
        request="Create an orange sphere too meters ahead of me.",
        required_tools=frozenset({"add_primitive"}),
        expected_positions=(("sphere-0", (0.0, 1.6, -2.0)),),
    ),
    Case(
        name="basics_filler_heavy_create",
        request="Um, could you like, make a purple cube, please.",
        required_tools=frozenset({"add_primitive"}),
        expected_colors=(("box-0", (0.6, 0.0, 1.0)),),
    ),
    Case(
        name="basics_run_on_two_commands",
        request="Make a cyan sphere move it up a little.",
        required_tools=frozenset({"add_primitive", "update_primitive"}),
        forbidden_tools=frozenset({"remove_primitive"}),
        expected_call_counts=(("add_primitive", 1),),
    ),
    Case(
        name="basics_dropped_preposition_move",
        request="Put the sphere the box.",
        scene=_BASICS_SCENE,
        history=_BASICS_HISTORY,
        forbidden_tools=frozenset({"add_primitive", "remove_primitive"}),
    ),
    Case(
        name="basics_anchored_create_keeps_words",
        # Live flail: the supervisor misroutes an anchored create to
        # placement, gets it bounced, then re-delegates WITHOUT the user's
        # spatial words and patches with a move. One add, no update, and the
        # cube must land on the user's left of the anchor.
        request="Put a yellow cube to the left of the blue cube.",
        scene=(
            {"id": "box-0", "type": "box", "position": {"x": 0.5, "y": 1.4, "z": -1.2},
             "color": {"r": 0, "g": 0, "b": 1}, "size": 0.1},
        ),
        history=_BASICS_HISTORY,
        required_tools=frozenset({"add_primitive"}),
        forbidden_tools=frozenset({"update_primitive", "remove_primitive"}),
        expected_call_counts=(("add_primitive", 1),),
        expected_colors=(("box-1", (1.0, 1.0, 0.0)),),
    ),
    Case(
        name="basics_truncated_command",
        # VAD cuts the sentence mid-word; the only correct outcome is a
        # question. The failure mode is the verification pass nudging the
        # model into inventing the missing destination.
        request="Put the sphere on the",
        scene=_BASICS_SCENE,
        history=_BASICS_HISTORY,
        forbidden_tools=_NO_MUTATION,
    ),
    Case(
        name="basics_truncated_then_completed",
        # Follow-through: the user answers the truncation ask-back and the
        # deferred action completes against the conversation context.
        request="On the box.",
        scene=_BASICS_SCENE,
        history=(
            *_BASICS_HISTORY,
            ("Put the sphere on the", "I think I missed the end of that. On the what?"),
        ),
        required_tools=frozenset({"update_primitive"}),
        forbidden_tools=frozenset({"add_primitive", "remove_primitive"}),
    ),
    Case(
        name="basics_truncated_then_nevermind",
        request="Never mind.",
        scene=_BASICS_SCENE,
        history=(
            *_BASICS_HISTORY,
            ("Put the sphere on the", "I think I missed the end of that. On the what?"),
        ),
        forbidden_tools=_NO_MUTATION,
    ),
    Case(
        name="basics_truncated_then_full_command",
        request="Put the sphere on the box.",
        scene=_BASICS_SCENE,
        history=(
            *_BASICS_HISTORY,
            ("Put the sphere on the", "I think I missed the end of that. On the what?"),
        ),
        required_tools=frozenset({"update_primitive"}),
        forbidden_tools=frozenset({"add_primitive", "remove_primitive"}),
        expected_call_counts=(("update_primitive", 1),),
    ),
    Case(
        name="basics_truncated_then_bare_object",
        request="The box.",
        scene=_BASICS_SCENE,
        history=(
            *_BASICS_HISTORY,
            ("Put the sphere on the", "I think I missed the end of that. On the what?"),
        ),
        required_tools=frozenset({"update_primitive"}),
        forbidden_tools=frozenset({"add_primitive", "remove_primitive"}),
    ),
    Case(
        name="basics_fragment_with_history",
        request="Sounds good.",
        scene=_BASICS_SCENE,
        history=_BASICS_HISTORY,
        forbidden_tools=_NO_MUTATION,
    ),
    Case(
        name="basics_correction_with_history",
        request="That's the wrong sphere.",
        scene=_BASICS_SCENE,
        history=_BASICS_HISTORY,
        forbidden_tools=_NO_MUTATION,
    ),
)


async def run_case(case: Case) -> bool:
    scene = FakeScene.from_case(case)
    llm = make_llm(load_models_config(_CONFIG.models_yaml), "agent_llm")
    try:
        async with WorkflowBuilder() as builder:
            await scene.bind(builder)
            supervisor = await scene_supervisor(
                builder=builder,
                llm=llm,
            )
            try:
                reply = await supervisor.ainvoke(
                    SceneRequest(
                        transcript=case.request,
                        participant_id=_PARTICIPANT,
                        timestamp_us=10_000_000,
                    )
                )
                response = reply.response
            except Exception as exc:
                response = f"<workflow error: {exc}>"
    finally:
        await llm.close()
    called = {name for name, _arguments in scene.calls}
    missing = case.required_tools - called
    forbidden = case.forbidden_tools & called
    call_order = tuple(name for name, _arguments in scene.calls)
    wrong_call_counts = {
        name: call_order.count(name)
        for name, expected in case.expected_call_counts
        if call_order.count(name) != expected
    }
    positions = [call_order.index(name) for name in case.required_order if name in call_order]
    out_of_order = len(positions) != len(case.required_order) or positions != sorted(positions)
    wrong_sizes = {
        object_id: scene.objects[object_id].size if object_id in scene.objects else None
        for object_id, expected in case.expected_sizes
        if object_id not in scene.objects or scene.objects[object_id].size != expected
    }
    wrong_colors = {
        object_id: (
            tuple(scene.objects[object_id].color.model_dump().values())
            if object_id in scene.objects
            else None
        )
        for object_id, expected in case.expected_colors
        if object_id not in scene.objects
        or tuple(scene.objects[object_id].color.model_dump().values()) != expected
    }
    wrong_positions = {
        object_id: (
            tuple(scene.objects[object_id].position.model_dump().values())
            if object_id in scene.objects
            else None
        )
        for object_id, expected in case.expected_positions
        if object_id not in scene.objects
        or tuple(scene.objects[object_id].position.model_dump().values()) != expected
    }
    passed = (
        not missing
        and not forbidden
        and not out_of_order
        and not wrong_call_counts
        and not wrong_sizes
        and not wrong_colors
        and not wrong_positions
    )
    status = (
        "PASS"
        if passed
        else (
            f"FAIL missing={sorted(missing)} forbidden={sorted(forbidden)} order={call_order} "
            f"counts={wrong_call_counts} sizes={wrong_sizes} colors={wrong_colors} "
            f"positions={wrong_positions}"
        )
    )
    print(f"{status:32} {case.name}: {response}", flush=True)
    return passed


# Train/test separation audit. Prompt worked examples are what the model
# memorizes as templates; any overlap with case inputs means the eval scores
# recall, not behavior (proven live: a blue/green prompt example made the
# anchored-create case pass while the same request failed on other colors).
_EVAL_VOCAB_COLORS = ("red", "green", "blue", "yellow", "cyan", "orange", "purple", "white", "black")
_EVAL_VOCAB_SHAPES = ("sphere", "cube", "box", "ball")


def audit_prompts() -> None:
    import re
    from pathlib import Path

    worker = (Path(__file__).resolve().parent / "../../worker/xr_render_demo_worker").resolve()
    prompts = sorted(worker.rglob("*prompt*.txt"))
    utterances = [case["user"] for case in CORPUS_CASES if case.get("user")]
    utterances += [case.request for case in (*CASES, *BASICS)]
    fixture_ids = {
        item["id"]
        for case in CORPUS_CASES
        for item in case.get("scene", ())
    } | {item["id"] for case in (*CASES, *BASICS) for item in case.scene}
    # The other tiers score the same prompts; their inputs must stay
    # disjoint from prompt examples too.
    try:
        from . import subagents as eval_subagents
    except ImportError:
        pass
    else:
        utterances += [case.instruction for case in eval_subagents.CASES]
        fixture_ids |= {item["id"] for case in eval_subagents.CASES for item in case.scene}
    try:
        from . import supervisor as eval_supervisor
    except ImportError:
        pass
    else:
        utterances += [case.request for case in eval_supervisor.CASES]
        fixture_ids |= {item["id"] for case in eval_supervisor.CASES for item in case.scene}
    for prompt_path in prompts:
        label = str(prompt_path.relative_to(worker))
        text = prompt_path.read_text(encoding="utf-8")
        lowered = text.lower()
        for utterance in utterances:
            needle = utterance.lower().strip(".?! ")
            # Short fragments ("the box") appear in ordinary prose; only
            # utterances long enough to be templates count as overlap.
            if len(needle) >= 12 and needle in lowered:
                print(f"AUDIT WARNING {label}: contains case utterance {utterance!r}")
        for fixture_id in sorted(fixture_ids):
            if fixture_id.lower() in lowered:
                print(f"AUDIT WARNING {label}: contains case fixture id {fixture_id!r}")
        for quoted in re.findall(r'"([^"]{4,80})"', text):
            words = set(re.findall(r"[a-z]+", quoted.lower()))
            colors = words & set(_EVAL_VOCAB_COLORS)
            shapes = words & set(_EVAL_VOCAB_SHAPES)
            if colors and shapes:
                print(
                    f"AUDIT WARNING {label}: example {quoted!r} pairs eval vocabulary "
                    f"({', '.join(sorted(colors | shapes))}); use non-eval colors/shapes"
                )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", help="Case names; omit to run all cases")
    args = parser.parse_args()
    audit_prompts()
    wanted = set(args.cases)
    if wanted == {"basics"}:
        wanted = {case.name for case in BASICS}
    corpus = [case for case in CORPUS_CASES if not wanted or case["name"] in wanted]
    native = [case for case in CASES if not wanted or case.name in wanted]
    basics = [case for case in BASICS if not wanted or case.name in wanted]
    if not corpus and not native and not basics:
        raise SystemExit(f"unknown cases: {args.cases}")
    corpus_results = [await run_corpus_case(case) for case in corpus]
    native_results = [await run_case(case) for case in native]
    basics_results = [await run_case(case) for case in basics]
    if corpus_results:
        print(f"corpus: {sum(corpus_results)}/{len(corpus_results)} passed")
    if native_results:
        print(f"native: {sum(native_results)}/{len(native_results)} passed")
    if basics_results:
        print(f"basics: {sum(basics_results)}/{len(basics_results)} passed")
    if not all(corpus_results + native_results + basics_results):
        raise SystemExit(1)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
