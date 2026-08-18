# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code-composed movement and creation operations for the render subagents.

Every operation takes only semantic arguments (object ids, direction names,
metres) and resolves the user frame, anchor coordinates, geometry, and the
scene write in code. No pose or coordinate ever round-trips through the
agent LLM, which cannot relay them reliably.
"""

import asyncio
import difflib
import re
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field
from xr_ai_tools import Tool
from xr_ai_tools import spatial as _spatial
from xr_ai_tools.tracking import TrackingTools
from xr_ai_tools.types import SpatialFrame as _SpatialFrame
from xr_ai_tools.types import Vector3 as _Vector3
from xr_render_scene import (
    AddPrimitiveRequest,
    EmptyRequest,
    RemovePrimitiveRequest,
    SceneObject,
    SceneTools,
    UpdatePrimitiveRequest,
)

_UserDirection = Literal["front", "back", "left", "right", "above", "below"]
_AnchorRelation = Literal["toward_user", "away_from_user", "left_of", "right_of", "above", "below"]

_DEFAULT_COLOR = (0.2, 0.9, 1.0)

_COLOR_WORDS = {
    "red": (1, 0, 0), "green": (0, 0.8, 0), "blue": (0, 0.4, 1), "yellow": (1, 1, 0),
    "cyan": (0, 1, 1), "magenta": (1, 0, 1), "orange": (1, 0.5, 0), "purple": (0.6, 0, 1),
    "white": (1, 1, 1), "black": (0, 0, 0), "teal": (0, 0.8, 0.8), "turquoise": (0.2, 0.9, 1),
    "lavender": (0.6, 0.4, 1), "pink": (1, 0.5, 0.8), "gray": (0.5, 0.5, 0.5), "grey": (0.5, 0.5, 0.5),
}
_SHAPE_WORDS = {
    "box": "box", "cube": "box", "block": "box", "crate": "box",
    "sphere": "sphere", "ball": "sphere", "orb": "sphere",
    "cone": "cone", "cylinder": "cylinder", "capsule": "capsule",
    "ring": "ring", "pyramid": "pyramid", "torus": "torus", "donut": "torus",
}


class MovedObject(BaseModel):
    obj_id: str
    x: float
    y: float
    z: float


class SwappedObjects(BaseModel):
    first: MovedObject
    second: MovedObject


class CreatedObject(BaseModel):
    id: str
    x: float
    y: float
    z: float
    created_this_turn: int = Field(default=1, description="Objects created so far in this turn, including this one.")


class RecoloredObject(BaseModel):
    obj_id: str
    r: float
    g: float
    b: float


class RemovedObject(BaseModel):
    obj_id: str


class CreationLedger:
    """Suppress identical repeated mutations within one subagent turn."""

    def __init__(self) -> None:
        self._seen: dict[tuple, CreatedObject] = {}
        self.mutations: dict[tuple, MovedObject] = {}
        self.count = 0

    def reset(self) -> None:
        self._seen.clear()
        self.mutations.clear()
        self.count = 0

    def get(self, key: tuple) -> CreatedObject | None:
        return self._seen.get(key)

    def record(self, key: tuple, created: CreatedObject) -> None:
        self._seen[key] = created


class TurnGuard:
    """Block mutations of existing objects after a failed reference lookup."""

    def __init__(self) -> None:
        self.halted = False

    def reset(self) -> None:
        self.halted = False


class _Leaves:
    """Resolution helpers over the scene and tracking tools."""

    def __init__(
        self,
        scene: SceneTools,
        tracking: TrackingTools | None = None,
        ledger: CreationLedger | None = None,
        guard: TurnGuard | None = None,
    ) -> None:
        self._scene = scene
        self._tracking = tracking
        self.ledger = ledger
        self.guard = guard
        self._add_lock = asyncio.Lock()

    async def update(self, arguments: dict) -> None:
        await self._scene.update_primitive.execute(UpdatePrimitiveRequest.model_validate(arguments))

    async def remove(self, object_id: str) -> None:
        await self._scene.remove_primitive.execute(RemovePrimitiveRequest(obj_id=object_id))

    def check_writable(self) -> None:
        if self.guard is not None and self.guard.halted:
            raise ValueError(
                "An earlier object reference in this instruction could not be resolved; "
                "change nothing else and report that failure back."
            )

    async def user_frame(self) -> _SpatialFrame:
        if self._tracking is None:
            raise RuntimeError("tracking is not available")
        return await self._tracking.get_user_frame.execute(EmptyRequest())

    async def find(self, object_ref: str) -> SceneObject:
        wanted = "".join("-" if c in "‐‑‒–—−" else c
                         for c in object_ref).strip().lower()
        wanted = re.sub(r"[\s_]+", "-", wanted) if re.fullmatch(r"[A-Za-z]+[\s_-]+\d+", wanted) else wanted
        state = await self._scene.get_scene_state.execute(EmptyRequest())
        for item in state.objects:
            if item.id == wanted:
                return item
        known = sorted(item.id for item in state.objects)
        if re.fullmatch(r"[a-z]+-\d+", wanted):
            prefix, _, number = wanted.partition("-")
            synonym_id = f"{_SHAPE_WORDS.get(prefix, prefix)}-{number}"
            for item in state.objects:
                if item.id == synonym_id:
                    logger.debug("spatial op resolved {!r} -> {}", object_ref, item.id)
                    return item
            logger.debug("spatial op lookup failed: {!r} not in {}", object_ref, known)
            raise ValueError(f"No scene object with id {object_ref!r}; the scene has {known}")
        words = re.findall(r"[a-z]+", wanted)
        exact_shape = next((_SHAPE_WORDS[word] for word in words if word in _SHAPE_WORDS), None)
        color = next((_COLOR_WORDS[word] for word in words if word in _COLOR_WORDS), None)

        def select(shape: str | None) -> list[SceneObject]:
            pool = [item for item in state.objects if shape is None or item.type == shape]
            if color is None or not pool:
                return pool if shape is not None or color is not None else []
            def color_distance(item: SceneObject) -> float:
                return ((item.color.r - color[0]) ** 2 + (item.color.g - color[1]) ** 2
                        + (item.color.b - color[2]) ** 2)
            best = min(color_distance(item) for item in pool)
            if best > 0.4:
                return []
            return [item for item in pool if color_distance(item) - best < 0.05]

        candidates = select(exact_shape) if (exact_shape or color) else []
        if len(candidates) != 1 and exact_shape is None:
            for word in words:
                if word in _COLOR_WORDS:
                    continue
                close = difflib.get_close_matches(word, _SHAPE_WORDS, n=1, cutoff=0.6)
                if close:
                    fuzzy = select(_SHAPE_WORDS[close[0]])
                    if fuzzy and (not candidates or len(fuzzy) < len(candidates)):
                        candidates = fuzzy
                    break
        if len(candidates) == 1:
            logger.debug("spatial op resolved {!r} -> {}", object_ref, candidates[0].id)
            return candidates[0]
        if candidates:
            if self.guard is not None:
                self.guard.halted = True
            matches = ", ".join(item.id for item in candidates)
            raise ValueError(f"{object_ref!r} is ambiguous: it matches {matches}; ask which one is meant")
        logger.debug("spatial op lookup failed: {!r} not in {}", object_ref, known)
        if self.guard is not None:
            self.guard.halted = True
        raise ValueError(
            f"No scene object matches {object_ref!r}: nothing in the scene has that description. "
            "Never substitute a different object; make no further tool call and report this back."
        )

    def shape(self, shape_words: str) -> str:
        words = re.findall(r"[a-z]+", shape_words.lower())
        for word in words:
            if word in _SHAPE_WORDS:
                return _SHAPE_WORDS[word]
        for word in words:
            close = difflib.get_close_matches(word, _SHAPE_WORDS, n=1, cutoff=0.6)
            if close:
                logger.debug("shape words resolved {!r} -> {}", shape_words, _SHAPE_WORDS[close[0]])
                return _SHAPE_WORDS[close[0]]
        shapes = ", ".join(sorted(set(_SHAPE_WORDS.values())))
        raise ValueError(f"Unknown shape {shape_words!r}; the renderer draws: {shapes}")

    async def color(self, color_words: str) -> tuple[float, float, float]:
        if not re.search(r"[a-z]+-\d+", color_words.lower()):
            numbers = [float(v) for v in re.findall(r"-?\d*\.\d+|-?\d+", color_words)]
            if len(numbers) == 3 and all(0.0 <= v <= 1.0 for v in numbers):
                return (numbers[0], numbers[1], numbers[2])
        words = re.findall(r"[a-z]+", color_words.lower())
        if not words:
            return _DEFAULT_COLOR
        for word in words:
            if word in _COLOR_WORDS:
                return _COLOR_WORDS[word]
        halted = self.guard.halted if self.guard is not None else False
        try:
            source = await self.find(color_words)
            return (source.color.r, source.color.g, source.color.b)
        except Exception:
            if self.guard is not None:
                self.guard.halted = halted
        for word in words:
            close = difflib.get_close_matches(word, _COLOR_WORDS, n=1, cutoff=0.75)
            if close:
                logger.debug("color words resolved {!r} -> {}", color_words, close[0])
                return _COLOR_WORDS[close[0]]
        known = ", ".join(sorted(_COLOR_WORDS))
        raise ValueError(f"Unknown color {color_words!r}; use one of {known}, or name a scene object")

    async def spot(self, operation: str, arguments: dict) -> tuple[float, float, float]:
        if operation == "compute_user_relative_position":
            frame = _SpatialFrame.model_validate(arguments["user_frame"])
            result = _spatial.user_relative(frame, arguments["direction_from_user"], arguments["distance_meters"])
        elif operation == "offset_position_in_user_frame":
            frame = _SpatialFrame.model_validate(arguments["user_frame"])
            start = _Vector3.model_validate(arguments["start_position"])
            result = _spatial.offset_user_frame(
                frame, start,
                forward=arguments.get("forward_meters", 0.0),
                right=arguments.get("right_meters", 0.0),
                up=arguments.get("up_meters", 0.0),
            )
        elif operation == "compute_position_relative_to_anchor":
            frame = _SpatialFrame.model_validate(arguments["user_frame"])
            anchor = _Vector3.model_validate(arguments["anchor_position"])
            result = _spatial.anchor_relative(
                frame, anchor, arguments["relation_to_anchor"], arguments.get("distance_meters", 0.3)
            )
        elif operation == "compute_midpoint":
            first = _Vector3.model_validate(arguments["first_position"])
            second = _Vector3.model_validate(arguments["second_position"])
            result = _spatial.midpoint(first, second)
        elif operation == "compute_position_toward_or_away_from_reference":
            start = _Vector3.model_validate(arguments["start_position"])
            reference = _Vector3.model_validate(arguments["reference_position"])
            direction = arguments["movement_direction"]
            distance = arguments.get("distance_meters", 0.3)
            result = _spatial.toward(start, reference, distance if direction == "toward" else -distance)
        else:
            raise ValueError(f"unknown spatial operation: {operation!r}")
        return result.x, result.y, result.z

    async def write(self, object_id: str, position: tuple[float, float, float]) -> MovedObject:
        self.check_writable()
        x, y, z = position
        await self._scene.update_primitive.execute(
            UpdatePrimitiveRequest(obj_id=object_id, x=x, y=y, z=z)
        )
        return MovedObject(obj_id=object_id, x=x, y=y, z=z)

    async def add(
        self,
        prim_type: str,
        position: tuple[float, float, float],
        color: tuple[float, float, float],
        size: float,
    ) -> CreatedObject:
        x, y, z = position
        r, g, b = color
        key = (prim_type, round(x, 2), round(y, 2), round(z, 2), round(r, 2), round(g, 2), round(b, 2), round(size, 3))
        async with self._add_lock:
            if self.ledger is not None and (existing := self.ledger.get(key)) is not None:
                return existing
            result = await self._scene.add_primitive.execute(
                AddPrimitiveRequest(prim_type=prim_type, x=x, y=y, z=z, r=r, g=g, b=b, size=size)
            )
            created = CreatedObject(id=result.id, x=x, y=y, z=z)
            if self.ledger is not None:
                self.ledger.count += 1
                created.created_this_turn = self.ledger.count
                self.ledger.record(key, created)
            return created

    async def resize(self, obj: SceneObject, factor: float) -> MovedObject:
        self.check_writable()
        key = ("resize", obj.id, round(factor, 4))
        async with self._add_lock:
            if self.ledger is not None and (done := self.ledger.mutations.get(key)) is not None:
                return done
            await self._scene.update_primitive.execute(
                UpdatePrimitiveRequest(obj_id=obj.id, size=round(obj.size * factor, 4))
            )
            result = MovedObject(obj_id=obj.id, x=obj.position.x, y=obj.position.y, z=obj.position.z)
            if self.ledger is not None:
                self.ledger.mutations[key] = result
            return result


# ── Request models ────────────────────────────────────────────────────────────

class _ObjRequest(BaseModel):
    object_words: str = Field(
        description="The instruction's exact words for this object, copied verbatim (mangled nouns fine); "
                    "an id only when the instruction itself states that id."
    )


class _MoveUserRelativeRequest(_ObjRequest):
    direction: _UserDirection
    distance: float = Field(
        default=1.5, description="Distance from the user in metres; pass a stated distance exactly."
    )


class _NudgeRequest(_ObjRequest):
    forward: float = Field(default=0.0, description="Signed user-forward shift in metres.")
    right: float = Field(default=0.0, description="Signed user-right shift in metres.")
    up: float = Field(default=0.0, description="Signed world-up shift in metres.")


class _MoveObjectRelativeRequest(BaseModel):
    movee_words: str = Field(description="The instruction's exact words for the object to move.")
    anchor_words: str = Field(description="The instruction's exact words for the anchor object.")
    relation: _AnchorRelation
    distance: float = Field(default=0.3, description="Distance from the anchor in metres.")


class _MoveInsideRequest(BaseModel):
    movee_words: str = Field(description="The instruction's exact words for the object to move.")
    container_words: str = Field(description="The instruction's exact words for the container.")


class _MoveBetweenRequest(BaseModel):
    movee_words: str = Field(description="The instruction's exact words for the object to move.")
    first_anchor_words: str = Field(description="The instruction's exact words for the first anchor.")
    second_anchor_words: str = Field(description="The instruction's exact words for the second anchor.")


class _MoveTowardRequest(BaseModel):
    movee_words: str = Field(description="The instruction's exact words for the object to move.")
    target_words: str = Field(description="The instruction's exact words for the reference object.")
    direction: Literal["toward", "away"] = Field(default="toward", description="Move toward or away from the target.")
    distance: float = Field(default=0.5, ge=0, description="Non-negative travel distance in metres.")


class _MoveTowardUserRequest(_ObjRequest):
    direction: Literal["toward", "away"] = Field(default="toward", description="Move toward or away from the user.")
    distance: float = Field(default=0.5, ge=0, description="Non-negative travel distance in metres.")


class _SwapRequest(BaseModel):
    first_words: str = Field(description="The instruction's exact words for the first object.")
    second_words: str = Field(description="The instruction's exact words for the second object.")


class _MoveToRequest(_ObjRequest):
    x: float
    y: float
    z: float


class _RecolorRequest(_ObjRequest):
    color_words: str = Field(
        description="The instruction's exact color word(s), copied verbatim (mangled spellings fine), "
                    "or an object to copy the color from ('same as cone-7')."
    )


class _CreateUserRelativeRequest(BaseModel):
    prim_type: str = Field(description="The instruction's exact shape word, copied verbatim.")
    direction: _UserDirection
    color_words: str = Field(
        default="",
        description="The instruction's exact color word(s), copied verbatim, or empty when no color is stated."
    )
    distance: float = Field(default=1.5, description="Distance from the user in metres.")
    size: float = Field(default=0.1, description="Sphere radius or box half-edge in metres.")


class _CreateObjectRelativeRequest(BaseModel):
    prim_type: str = Field(description="The instruction's exact shape word, copied verbatim.")
    anchor_words: str = Field(description="The instruction's exact words for the first (or only) anchor object.")
    relation: _AnchorRelation = Field(default="above")
    second_anchor_words: str = Field(
        default="",
        description=(
            "The instruction's exact words for the second anchor object, only when the user says "
            "'between X and Y'. Leave empty for all other relations."
        ),
    )
    color_words: str = Field(
        default="",
        description=(
            "The instruction's exact color word(s), copied verbatim; include whenever the instruction "
            "names a color (e.g. 'blue square' → color_words='blue'). Leave empty only when truly unstated."
        ),
    )
    distance: float = Field(default=0.3, description="Distance from the anchor in metres.")
    size: float = Field(default=0.1, description="Sphere radius or box half-edge in metres.")


class _CreateAtRequest(BaseModel):
    prim_type: str = Field(description="The instruction's exact shape word, copied verbatim.")
    x: float
    y: float
    z: float
    color_words: str = Field(
        default="",
        description=(
            "The instruction's exact color word(s), copied verbatim; include whenever the instruction "
            "names a color. Leave empty only when truly unstated."
        ),
    )
    size: float = Field(default=0.1, description="Sphere radius or box half-edge in metres.")



class _ChangeShapeRequest(_ObjRequest):
    prim_type: str = Field(description="The instruction's exact shape word, copied verbatim.")


class _ResizeRequest(_ObjRequest):
    factor: float = Field(description="Multiplier applied to the current size.")


# ── Placement tools ───────────────────────────────────────────────────────────

def make_placement_tools(
    scene: SceneTools,
    tracking: TrackingTools,
    *,
    guard: TurnGuard | None = None,
) -> list[Tool]:
    """Return the placement tools for one subagent delegation."""
    leaves = _Leaves(scene, tracking, guard=guard)

    async def move_user_relative(req: _MoveUserRelativeRequest) -> MovedObject:
        target = await leaves.find(req.object_words)
        frame = await leaves.user_frame()
        spot = await leaves.spot("compute_user_relative_position",
            {"user_frame": frame.model_dump(), "direction_from_user": req.direction, "distance_meters": req.distance})
        return await leaves.write(target.id, spot)

    async def nudge(req: _NudgeRequest) -> MovedObject:
        current = await leaves.find(req.object_words)
        frame = await leaves.user_frame()
        spot = await leaves.spot("offset_position_in_user_frame", {
            "user_frame": frame.model_dump(), "start_position": current.position.model_dump(),
            "forward_meters": req.forward, "right_meters": req.right, "up_meters": req.up,
        })
        return await leaves.write(current.id, spot)

    async def move_object_relative(req: _MoveObjectRelativeRequest) -> MovedObject:
        logger.debug("move_object_relative movee={!r} anchor={!r} relation={}",
                     req.movee_words, req.anchor_words, req.relation)
        movee = await leaves.find(req.movee_words)
        anchor = await leaves.find(req.anchor_words)
        if movee.id == anchor.id:
            raise ValueError(
                f"{req.movee_words!r} and {req.anchor_words!r} are the same object ({movee.id}); "
                "an object cannot be placed relative to itself."
            )
        frame = await leaves.user_frame()
        spot = await leaves.spot("compute_position_relative_to_anchor", {
            "user_frame": frame.model_dump(), "anchor_position": anchor.position.model_dump(),
            "relation_to_anchor": req.relation, "distance_meters": req.distance,
        })
        return await leaves.write(movee.id, spot)

    async def move_inside(req: _MoveInsideRequest) -> MovedObject:
        movee = await leaves.find(req.movee_words)
        container = await leaves.find(req.container_words)
        return await leaves.write(movee.id, (container.position.x, container.position.y, container.position.z))

    async def move_between(req: _MoveBetweenRequest) -> MovedObject:
        movee = await leaves.find(req.movee_words)
        anchor_a = await leaves.find(req.first_anchor_words)
        anchor_b = await leaves.find(req.second_anchor_words)
        spot = await leaves.spot("compute_midpoint", {
            "first_position": anchor_a.position.model_dump(),
            "second_position": anchor_b.position.model_dump(),
        })
        return await leaves.write(movee.id, spot)

    async def move_toward(req: _MoveTowardRequest) -> MovedObject:
        movee = await leaves.find(req.movee_words)
        target = await leaves.find(req.target_words)
        spot = await leaves.spot("compute_position_toward_or_away_from_reference", {
            "start_position": movee.position.model_dump(),
            "reference_position": target.position.model_dump(),
            "movement_direction": req.direction, "distance_meters": req.distance,
        })
        return await leaves.write(movee.id, spot)

    async def move_toward_user(req: _MoveTowardUserRequest) -> MovedObject:
        movee = await leaves.find(req.object_words)
        frame = await leaves.user_frame()
        spot = await leaves.spot("compute_position_toward_or_away_from_reference", {
            "start_position": movee.position.model_dump(),
            "reference_position": frame.origin.model_dump(),
            "movement_direction": req.direction, "distance_meters": req.distance,
        })
        return await leaves.write(movee.id, spot)

    async def swap_positions(req: _SwapRequest) -> SwappedObjects:
        first = await leaves.find(req.first_words)
        second = await leaves.find(req.second_words)
        first_pos = (first.position.x, first.position.y, first.position.z)
        second_pos = (second.position.x, second.position.y, second.position.z)
        return SwappedObjects(
            first=await leaves.write(first.id, second_pos),
            second=await leaves.write(second.id, first_pos),
        )

    async def move_to(req: _MoveToRequest) -> MovedObject:
        target = await leaves.find(req.object_words)
        return await leaves.write(target.id, (req.x, req.y, req.z))

    return [
        Tool("move_user_relative", "Move an existing object to a point in a named direction from the user. "
             "Not for stated shifts like 'one metre to my left'; nudge does those.",
             _MoveUserRelativeRequest, MovedObject, move_user_relative),
        Tool("nudge", "Shift an existing object from its current position by signed user-frame offsets.",
             _NudgeRequest, MovedObject, nudge),
        Tool("move_object_relative", "Move an existing object to a point in a named relation to an anchor object.",
             _MoveObjectRelativeRequest, MovedObject, move_object_relative),
        Tool("move_inside", "Move an existing object into the center of a container object.",
             _MoveInsideRequest, MovedObject, move_inside),
        Tool("move_between", "Move an existing object to the midpoint between two anchor objects.",
             _MoveBetweenRequest, MovedObject, move_between),
        Tool("move_toward", "Move an existing object toward or away from another object.",
             _MoveTowardRequest, MovedObject, move_toward),
        Tool("move_toward_user", "Move an existing object toward or away from the user.",
             _MoveTowardUserRequest, MovedObject, move_toward_user),
        Tool("swap_positions", "Exchange the positions of two existing objects.",
             _SwapRequest, SwappedObjects, swap_positions),
        Tool("move_to", "Move an existing object to explicit world coordinates taken from the request, "
             "SCENE OBJECTS, or [Recent moves]; never invent coordinates.",
             _MoveToRequest, MovedObject, move_to),
    ]


# ── Appearance tools ──────────────────────────────────────────────────────────

def make_appearance_tools(
    scene: SceneTools,
    *,
    guard: TurnGuard | None = None,
) -> list[Tool]:
    leaves = _Leaves(scene, guard=guard)

    async def recolor(req: _RecolorRequest) -> RecoloredObject:
        leaves.check_writable()
        target = await leaves.find(req.object_words)
        r, g, b = await leaves.color(req.color_words)
        await leaves.update({"obj_id": target.id, "r": r, "g": g, "b": b})
        return RecoloredObject(obj_id=target.id, r=r, g=g, b=b)

    return [
        Tool("recolor", "Change an existing object's color, keeping position, type, and size.",
             _RecolorRequest, RecoloredObject, recolor),
    ]


# ── Object tools ──────────────────────────────────────────────────────────────

def make_object_tools(
    scene: SceneTools,
    tracking: TrackingTools,
    *,
    ledger: CreationLedger | None = None,
    guard: TurnGuard | None = None,
) -> list[Tool]:
    leaves = _Leaves(scene, tracking, ledger=ledger, guard=guard)

    async def create_user_relative(req: _CreateUserRelativeRequest) -> CreatedObject:
        prim = leaves.shape(req.prim_type)
        color = await leaves.color(req.color_words)
        frame = await leaves.user_frame()
        spot = await leaves.spot("compute_user_relative_position",
            {"user_frame": frame.model_dump(), "direction_from_user": req.direction, "distance_meters": req.distance})
        return await leaves.add(prim, spot, color, req.size)

    async def create_object_relative(req: _CreateObjectRelativeRequest) -> CreatedObject:
        prim = leaves.shape(req.prim_type)
        color = await leaves.color(req.color_words)
        try:
            anchor = await leaves.find(req.anchor_words)
        except ValueError as error:
            raise ValueError(
                f"{error}. If the instruction names no existing object to anchor on, this is a bare "
                "creation: call create_user_relative with direction front and distance 1.5 instead."
            ) from None
        if req.second_anchor_words:
            logger.debug("create_object_relative between={!r} and={!r}", req.anchor_words, req.second_anchor_words)
            anchor_b = await leaves.find(req.second_anchor_words)
            spot = await leaves.spot("compute_midpoint", {
                "first_position": anchor.position.model_dump(),
                "second_position": anchor_b.position.model_dump(),
            })
        else:
            logger.debug("create_object_relative anchor={!r} relation={} distance={}",
                         req.anchor_words, req.relation, req.distance)
            frame = await leaves.user_frame()
            spot = await leaves.spot("compute_position_relative_to_anchor", {
                "user_frame": frame.model_dump(), "anchor_position": anchor.position.model_dump(),
                "relation_to_anchor": req.relation, "distance_meters": req.distance,
            })
        return await leaves.add(prim, spot, color, req.size)

    async def create_at(req: _CreateAtRequest) -> CreatedObject:
        logger.debug("create_at ({}, {}, {})", req.x, req.y, req.z)
        prim = leaves.shape(req.prim_type)
        color = await leaves.color(req.color_words)
        return await leaves.add(prim, (req.x, req.y, req.z), color, req.size)

    async def change_shape(req: _ChangeShapeRequest) -> MovedObject:
        leaves.check_writable()
        prim = leaves.shape(req.prim_type)
        current = await leaves.find(req.object_words)
        await leaves.update({"obj_id": current.id, "prim_type": prim})
        return MovedObject(obj_id=current.id, x=current.position.x, y=current.position.y, z=current.position.z)

    async def resize_object(req: _ResizeRequest) -> MovedObject:
        current = await leaves.find(req.object_words)
        return await leaves.resize(current, req.factor)

    async def remove_object(req: _ObjRequest) -> RemovedObject:
        leaves.check_writable()
        target = await leaves.find(req.object_words)
        await leaves.remove(target.id)
        return RemovedObject(obj_id=target.id)

    return [
        Tool("create_user_relative", "Create a new object at a point in a named direction from the user.",
             _CreateUserRelativeRequest, CreatedObject, create_user_relative),
        Tool("create_object_relative",
             "Create a new object relative to one anchor object, or at the midpoint between two anchor objects "
             "(set second_anchor_words for 'between X and Y').",
             _CreateObjectRelativeRequest, CreatedObject, create_object_relative),
        Tool("create_at", "Create a new object at explicit world coordinates.",
             _CreateAtRequest, CreatedObject, create_at),
        Tool("change_shape",
             "Change an existing object into another primitive type, keeping position, color, and size.",
             _ChangeShapeRequest, MovedObject, change_shape),
        Tool("resize_object", "Multiply an existing object's size by a factor, keeping everything else.",
             _ResizeRequest, MovedObject, resize_object),
        Tool("remove_object", "Remove an existing object from the scene.",
             _ObjRequest, RemovedObject, remove_object),
    ]


__all__ = [
    "CreatedObject", "CreationLedger", "MovedObject", "RecoloredObject", "RemovedObject",
    "SwappedObjects", "TurnGuard",
    "make_appearance_tools", "make_object_tools", "make_placement_tools",
]
