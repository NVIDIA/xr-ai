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
    MutationResult,
    RemovePrimitiveRequest,
    SceneObject,
    SceneTools,
    UpdatePrimitiveRequest,
)

from ._trace import current_mutation_evidence


def _record(field: str) -> None:
    evidence = current_mutation_evidence.get()
    if evidence is not None:
        setattr(evidence, field, getattr(evidence, field) + 1)


_UserDirection = Literal["at", "front", "back", "left", "right", "above", "below"]
_AnchorRelation = Literal["toward_user", "away_from_user", "left_of", "right_of", "above", "below"]

_DEFAULT_COLOR = (0.2, 0.9, 1.0)

_COLOR_WORDS = {
    "red": (1, 0, 0),
    "green": (0, 0.8, 0),
    "blue": (0, 0.4, 1),
    "yellow": (1, 1, 0),
    "cyan": (0, 1, 1),
    "magenta": (1, 0, 1),
    "orange": (1, 0.5, 0),
    "brown": (0.5, 0.25, 0.1),
    "purple": (0.6, 0, 1),
    "white": (1, 1, 1),
    "black": (0, 0, 0),
    "teal": (0, 0.8, 0.8),
    "turquoise": (0.2, 0.9, 1),
    "lavender": (0.6, 0.4, 1),
    "pink": (1, 0.5, 0.8),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
}
_SHAPE_WORDS = {
    "box": "box",
    "cube": "box",
    "block": "box",
    "crate": "box",
    "sphere": "sphere",
    "ball": "sphere",
    "orb": "sphere",
    "cone": "cone",
    "cylinder": "cylinder",
    "capsule": "capsule",
    "ring": "ring",
    "pyramid": "pyramid",
    "torus": "torus",
    "donut": "torus",
}

# A discriminated color source: the subagent LLM picks the kind through the
# tool schema; the code dispatches on it without reinterpreting the phrase.
_ColorKind = Literal["literal", "scene_object", "physical"]

_COLOR_KIND_GUIDE = (
    "Which kind of color source the instruction names: literal for stated "
    "color words or numbers (mangled spellings fine), scene_object to copy "
    "an existing XR object, physical for any real-world thing (clothing, a "
    "held object, a room surface) observed through the camera."
)
_COLOR_VALUE_GUIDE = (
    "The instruction's exact words for that source, copied verbatim: the "
    "color word(s) for literal ('teel'), the XR object words or id for "
    "scene_object ('capsule-8'), or the whole real-world phrase for "
    "physical ('the crate I'm holding', 'the ceiling')."
)


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


class CreatedObjects(BaseModel):
    objects: list[CreatedObject]


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
        physical_color: Tool | None = None,
    ) -> None:
        self._scene = scene
        self._tracking = tracking
        self.ledger = ledger
        self.guard = guard
        self.physical_color = physical_color
        self._add_lock = asyncio.Lock()

    @staticmethod
    def _confirm(result: MutationResult) -> MutationResult:
        if not result.ok:
            raise ValueError(f"the scene rejected the change: {result.reason or 'unknown reason'}")
        _record("applied")
        return result

    async def update(self, arguments: dict) -> MutationResult:
        return self._confirm(
            await self._scene.update_primitive.execute(UpdatePrimitiveRequest.model_validate(arguments))
        )

    async def remove(self, object_id: str) -> None:
        self._confirm(await self._scene.remove_primitive.execute(RemovePrimitiveRequest(obj_id=object_id)))

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
        wanted = "".join("-" if c in "‐‑‒–—−" else c for c in object_ref).strip().lower()
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
                return (item.color.r - color[0]) ** 2 + (item.color.g - color[1]) ** 2 + (item.color.b - color[2]) ** 2

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
            if word in _COLOR_WORDS:
                raise ValueError(
                    f"{word!r} is a color, not a shape. Do not change the shape; "
                    "report the color change back as a recolor for appearance_agent."
                )
        for word in words:
            close = difflib.get_close_matches(word, _SHAPE_WORDS, n=1, cutoff=0.6)
            if close:
                logger.debug("shape words resolved {!r} -> {}", shape_words, _SHAPE_WORDS[close[0]])
                return _SHAPE_WORDS[close[0]]
        shapes = ", ".join(sorted(set(_SHAPE_WORDS.values())))
        raise ValueError(f"Unknown shape {shape_words!r}; the renderer draws: {shapes}")

    async def resolve_color(self, kind: _ColorKind, value: str) -> tuple[float, float, float]:
        if kind == "literal":
            numbers = [float(v) for v in re.findall(r"-?\d*\.\d+|-?\d+", value)]
            if len(numbers) == 3 and all(0.0 <= v <= 1.0 for v in numbers):
                return (numbers[0], numbers[1], numbers[2])
            words = re.findall(r"[a-z]+", value.lower())
            if len(numbers) == 3 and not words:
                raise ValueError(f"numeric color {value!r} is out of range; r, g, b must each be between 0 and 1")
            if not words:
                return _DEFAULT_COLOR
            for word in words:
                if word in _COLOR_WORDS:
                    return _COLOR_WORDS[word]
            for word in words:
                close = difflib.get_close_matches(word, _COLOR_WORDS, n=1, cutoff=0.75)
                if close:
                    logger.debug("color words resolved {!r} -> {}", value, close[0])
                    return _COLOR_WORDS[close[0]]
            known = ", ".join(sorted(_COLOR_WORDS))
            raise ValueError(
                f"Unknown color {value!r}; use one of {known}, copy a scene object, or observe a physical source"
            )
        if not value.strip():
            raise ValueError(f"color_value is required for color_kind {kind!r}")
        if kind == "scene_object":
            # A failed scene reference surfaces as its recoverable ask-back;
            # it never falls through to the camera.
            match = await self.find(value)
            return (match.color.r, match.color.g, match.color.b)
        if kind != "physical":
            raise ValueError(f"unknown color kind {kind!r}")
        if self.physical_color is None:
            raise ValueError(f"no camera is available to observe {value!r}")
        resolved = await self.physical_color.execute(self.physical_color.request_model(source_words=value))
        return (resolved.r, resolved.g, resolved.b)

    async def spot(self, operation: str, arguments: dict) -> tuple[float, float, float]:
        if operation == "compute_user_relative_position":
            frame = _SpatialFrame.model_validate(arguments["user_frame"])
            if arguments["direction_from_user"] == "at":
                result = frame.origin
            else:
                result = _spatial.user_relative(
                    frame,
                    arguments["direction_from_user"],
                    arguments["distance_meters"],
                )
        elif operation == "offset_position_in_user_frame":
            frame = _SpatialFrame.model_validate(arguments["user_frame"])
            start = _Vector3.model_validate(arguments["start_position"])
            result = _spatial.offset_user_frame(
                frame,
                start,
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
        self._confirm(
            await self._scene.update_primitive.execute(UpdatePrimitiveRequest(obj_id=object_id, x=x, y=y, z=z))
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
            if not result.ok:
                raise ValueError(f"the scene rejected the creation: {result.reason or 'unknown reason'}")
            _record("applied")
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
            self._confirm(
                await self._scene.update_primitive.execute(
                    UpdatePrimitiveRequest(obj_id=obj.id, size=round(obj.size * factor, 4))
                )
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


class _RepeatNudgeRequest(_NudgeRequest):
    repetitions: int = Field(ge=1, le=20, description="Number of complete out-and-back cycles.")


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
    color_kind: _ColorKind = Field(description=_COLOR_KIND_GUIDE)
    color_value: str = Field(description=_COLOR_VALUE_GUIDE)


class _CreateUserRelativeRequest(BaseModel):
    prim_type: str = Field(
        description="Only the instruction's shape noun, copied verbatim; never put a color word here."
    )
    direction: _UserDirection = Field(
        description=(
            "Requested direction from the user. Use front when no position is stated, below for "
            "the user's feet or floor, and at only for the user's exact current location."
        )
    )
    color_kind: _ColorKind = Field(default="literal", description=_COLOR_KIND_GUIDE)
    color_value: str = Field(
        default="", description=_COLOR_VALUE_GUIDE + " Leave empty when the instruction states no color."
    )
    distance_from_user_meters: float = Field(
        description="Position offset from the user; copy an explicitly stated spatial distance, otherwise 1.5."
    )
    object_radius_or_half_edge_meters: float = Field(
        default=0.1,
        description="Object dimension only; keep 0.1 unless dimensions are explicitly requested.",
    )


class _CreateLinearArrangementRequest(BaseModel):
    prim_type: str = Field(
        description="Only the instruction's shape noun, copied verbatim; never put a color word here."
    )
    count: int = Field(ge=2, le=20, description="Number of new objects in the arrangement.")
    layout: Literal["horizontal_row", "vertical_stack"]
    color_kind: _ColorKind = Field(default="literal", description=_COLOR_KIND_GUIDE)
    color_value: str = Field(
        default="", description=_COLOR_VALUE_GUIDE + " Leave empty when the instruction states no color."
    )
    distance: float = Field(default=1.5, description="Distance in front of the user to the stack base.")
    spacing: float = Field(
        default=0.0,
        ge=0.0,
        description="Requested center-to-center spacing in metres; zero uses touching objects.",
    )
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
    color_kind: _ColorKind = Field(default="literal", description=_COLOR_KIND_GUIDE)
    color_value: str = Field(
        default="",
        description=_COLOR_VALUE_GUIDE + " Include whenever the instruction names a "
        "color source; leave empty only when truly unstated.",
    )
    distance: float = Field(default=0.3, description="Distance from the anchor in metres.")
    size: float = Field(default=0.1, description="Sphere radius or box half-edge in metres.")


class _CreateAtRequest(BaseModel):
    prim_type: str = Field(description="The instruction's exact shape word, copied verbatim.")
    x: float
    y: float
    z: float
    color_kind: _ColorKind = Field(default="literal", description=_COLOR_KIND_GUIDE)
    color_value: str = Field(
        default="",
        description=_COLOR_VALUE_GUIDE + " Include whenever the instruction names a "
        "color source; leave empty only when truly unstated.",
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
        spot = await leaves.spot(
            "compute_user_relative_position",
            {"user_frame": frame.model_dump(), "direction_from_user": req.direction, "distance_meters": req.distance},
        )
        return await leaves.write(target.id, spot)

    async def nudge(req: _NudgeRequest) -> MovedObject:
        current = await leaves.find(req.object_words)
        frame = await leaves.user_frame()
        spot = await leaves.spot(
            "offset_position_in_user_frame",
            {
                "user_frame": frame.model_dump(),
                "start_position": current.position.model_dump(),
                "forward_meters": req.forward,
                "right_meters": req.right,
                "up_meters": req.up,
            },
        )
        return await leaves.write(current.id, spot)

    async def repeat_nudge(req: _RepeatNudgeRequest) -> MovedObject:
        current = await leaves.find(req.object_words)
        frame = await leaves.user_frame()
        position = current.position
        result: MovedObject | None = None
        for _ in range(req.repetitions):
            outward = await leaves.spot(
                "offset_position_in_user_frame",
                {
                    "user_frame": frame.model_dump(),
                    "start_position": position.model_dump(),
                    "forward_meters": req.forward,
                    "right_meters": req.right,
                    "up_meters": req.up,
                },
            )
            result = await leaves.write(current.id, outward)
            position = _Vector3(x=result.x, y=result.y, z=result.z)
            returned = await leaves.spot(
                "offset_position_in_user_frame",
                {
                    "user_frame": frame.model_dump(),
                    "start_position": position.model_dump(),
                    "forward_meters": -req.forward,
                    "right_meters": -req.right,
                    "up_meters": -req.up,
                },
            )
            result = await leaves.write(current.id, returned)
            position = _Vector3(x=result.x, y=result.y, z=result.z)
        assert result is not None
        return result

    async def move_object_relative(req: _MoveObjectRelativeRequest) -> MovedObject:
        logger.debug(
            "move_object_relative movee={!r} anchor={!r} relation={}", req.movee_words, req.anchor_words, req.relation
        )
        movee = await leaves.find(req.movee_words)
        anchor = await leaves.find(req.anchor_words)
        if movee.id == anchor.id:
            raise ValueError(
                f"{req.movee_words!r} and {req.anchor_words!r} are the same object ({movee.id}); "
                "an object cannot be placed relative to itself."
            )
        frame = await leaves.user_frame()
        spot = await leaves.spot(
            "compute_position_relative_to_anchor",
            {
                "user_frame": frame.model_dump(),
                "anchor_position": anchor.position.model_dump(),
                "relation_to_anchor": req.relation,
                "distance_meters": req.distance,
            },
        )
        return await leaves.write(movee.id, spot)

    async def move_inside(req: _MoveInsideRequest) -> MovedObject:
        movee = await leaves.find(req.movee_words)
        container = await leaves.find(req.container_words)
        return await leaves.write(movee.id, (container.position.x, container.position.y, container.position.z))

    async def move_between(req: _MoveBetweenRequest) -> MovedObject:
        movee = await leaves.find(req.movee_words)
        anchor_a = await leaves.find(req.first_anchor_words)
        anchor_b = await leaves.find(req.second_anchor_words)
        spot = await leaves.spot(
            "compute_midpoint",
            {
                "first_position": anchor_a.position.model_dump(),
                "second_position": anchor_b.position.model_dump(),
            },
        )
        return await leaves.write(movee.id, spot)

    async def move_toward(req: _MoveTowardRequest) -> MovedObject:
        movee = await leaves.find(req.movee_words)
        target = await leaves.find(req.target_words)
        spot = await leaves.spot(
            "compute_position_toward_or_away_from_reference",
            {
                "start_position": movee.position.model_dump(),
                "reference_position": target.position.model_dump(),
                "movement_direction": req.direction,
                "distance_meters": req.distance,
            },
        )
        return await leaves.write(movee.id, spot)

    async def move_toward_user(req: _MoveTowardUserRequest) -> MovedObject:
        movee = await leaves.find(req.object_words)
        frame = await leaves.user_frame()
        spot = await leaves.spot(
            "compute_position_toward_or_away_from_reference",
            {
                "start_position": movee.position.model_dump(),
                "reference_position": frame.origin.model_dump(),
                "movement_direction": req.direction,
                "distance_meters": req.distance,
            },
        )
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
        Tool(
            "move_user_relative",
            "Move an existing object to a point in a named direction from the user. "
            "Use direction at for the user's current location. Not for stated shifts like "
            "'one metre to my left'; nudge does those.",
            _MoveUserRelativeRequest,
            MovedObject,
            move_user_relative,
        ),
        Tool(
            "nudge",
            "Shift an existing object from its current position by signed user-frame offsets. "
            "Moving 'further away' along viewing depth is a positive forward offset, preserving "
            "the object's lateral and vertical coordinates. Use only for one shift, never for "
            "repeated, cyclic, or back-and-forth motion.",
            _NudgeRequest,
            MovedObject,
            nudge,
        ),
        Tool(
            "repeat_nudge",
            "Perform complete repeated out-and-back motion of one existing object. For N "
            "repetitions of two opposing phases, set the first-phase signed offset and "
            "repetitions=N; the tool performs both phases of every cycle.",
            _RepeatNudgeRequest,
            MovedObject,
            repeat_nudge,
            return_direct=True,
            render_result=lambda result: (
                f"Completed the requested repeated out-and-back motion for {result.obj_id}."
            ),
            examples=(
                "For two left-and-right cycles, set a negative right offset and repetitions=2.",
            ),
        ),
        Tool(
            "move_object_relative",
            "Move an existing object to a point in a named relation to an anchor object.",
            _MoveObjectRelativeRequest,
            MovedObject,
            move_object_relative,
        ),
        Tool(
            "move_inside",
            "Move an existing object into the center of a container object.",
            _MoveInsideRequest,
            MovedObject,
            move_inside,
        ),
        Tool(
            "move_between",
            "Move an existing object to the midpoint between two anchor objects.",
            _MoveBetweenRequest,
            MovedObject,
            move_between,
        ),
        Tool(
            "move_toward",
            "Move an existing object toward or away from another object.",
            _MoveTowardRequest,
            MovedObject,
            move_toward,
        ),
        Tool(
            "move_toward_user",
            "Move an existing object radially toward or away from the user's position. Use only "
            "for explicit 'toward me' or 'away from me', not 'further away' along viewing depth.",
            _MoveTowardUserRequest,
            MovedObject,
            move_toward_user,
        ),
        Tool(
            "swap_positions",
            "Exchange the positions of two existing objects.",
            _SwapRequest,
            SwappedObjects,
            swap_positions,
        ),
        Tool(
            "move_to",
            "Move an existing object to explicit world coordinates taken from the request, "
            "SCENE OBJECTS, or [Recent moves]; never invent coordinates.",
            _MoveToRequest,
            MovedObject,
            move_to,
            examples=(
                "To restore an object where it was before, copy its previous coordinates from "
                "[Recent moves] into one move_to call.",
            ),
        ),
    ]


# ── Appearance tools ──────────────────────────────────────────────────────────


def make_appearance_tools(
    scene: SceneTools,
    *,
    guard: TurnGuard | None = None,
    physical_color: Tool | None = None,
) -> list[Tool]:
    leaves = _Leaves(scene, guard=guard, physical_color=physical_color)

    async def recolor(req: _RecolorRequest) -> RecoloredObject:
        leaves.check_writable()
        target = await leaves.find(req.object_words)
        r, g, b = await leaves.resolve_color(req.color_kind, req.color_value)
        current = (target.color.r, target.color.g, target.color.b)
        if all(abs(have - want) < 1e-6 for have, want in zip(current, (r, g, b))):
            # Requested state already holds; record it so the supervisor's
            # success gate accepts an "already that color" reply.
            _record("satisfied")
            return RecoloredObject(obj_id=target.id, r=r, g=g, b=b)
        await leaves.update({"obj_id": target.id, "r": r, "g": g, "b": b})
        return RecoloredObject(obj_id=target.id, r=r, g=g, b=b)

    return [
        Tool(
            "recolor",
            "Change an existing object's color, keeping position, type, and size. A held, worn, "
            "or room object is a physical color source even when an XR object has the same shape; "
            "preserve that complete physical phrase in color_value.",
            _RecolorRequest,
            RecoloredObject,
            recolor,
            examples=(
                "To match an existing ring to a worn scarf, use color_kind physical and preserve "
                "the complete scarf phrase in color_value.",
            ),
        ),
    ]


# ── Object tools ──────────────────────────────────────────────────────────────


def make_object_tools(
    scene: SceneTools,
    tracking: TrackingTools,
    *,
    ledger: CreationLedger | None = None,
    guard: TurnGuard | None = None,
    physical_color: Tool | None = None,
) -> list[Tool]:
    leaves = _Leaves(scene, tracking, ledger=ledger, guard=guard, physical_color=physical_color)

    async def create_user_relative(req: _CreateUserRelativeRequest) -> CreatedObject:
        prim = leaves.shape(req.prim_type)
        color = await leaves.resolve_color(req.color_kind, req.color_value)
        frame = await leaves.user_frame()
        spot = await leaves.spot(
            "compute_user_relative_position",
            {
                "user_frame": frame.model_dump(),
                "direction_from_user": req.direction,
                "distance_meters": req.distance_from_user_meters,
            },
        )
        return await leaves.add(prim, spot, color, req.object_radius_or_half_edge_meters)

    async def create_linear_arrangement(req: _CreateLinearArrangementRequest) -> CreatedObjects:
        prim = leaves.shape(req.prim_type)
        color = await leaves.resolve_color(req.color_kind, req.color_value)
        frame = await leaves.user_frame()
        base = await leaves.spot(
            "compute_user_relative_position",
            {"user_frame": frame.model_dump(), "direction_from_user": "front", "distance_meters": req.distance},
        )
        gap = req.spacing or max(0.05, req.size * 2)
        center = (req.count - 1) / 2
        positions = []
        for index in range(req.count):
            offset = (index - center) * gap
            if req.layout == "vertical_stack":
                positions.append((base[0], base[1] + index * gap, base[2]))
            else:
                positions.append(
                    (
                        base[0] + frame.right.x * offset,
                        base[1] + frame.right.y * offset,
                        base[2] + frame.right.z * offset,
                    )
                )
        created = [await leaves.add(prim, position, color, req.size) for position in positions]
        return CreatedObjects(objects=created)

    async def create_object_relative(req: _CreateObjectRelativeRequest) -> CreatedObject:
        prim = leaves.shape(req.prim_type)
        color = await leaves.resolve_color(req.color_kind, req.color_value)
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
            spot = await leaves.spot(
                "compute_midpoint",
                {
                    "first_position": anchor.position.model_dump(),
                    "second_position": anchor_b.position.model_dump(),
                },
            )
        else:
            logger.debug(
                "create_object_relative anchor={!r} relation={} distance={}",
                req.anchor_words,
                req.relation,
                req.distance,
            )
            frame = await leaves.user_frame()
            spot = await leaves.spot(
                "compute_position_relative_to_anchor",
                {
                    "user_frame": frame.model_dump(),
                    "anchor_position": anchor.position.model_dump(),
                    "relation_to_anchor": req.relation,
                    "distance_meters": req.distance,
                },
            )
        return await leaves.add(prim, spot, color, req.size)

    async def create_at(req: _CreateAtRequest) -> CreatedObject:
        logger.debug("create_at ({}, {}, {})", req.x, req.y, req.z)
        prim = leaves.shape(req.prim_type)
        color = await leaves.resolve_color(req.color_kind, req.color_value)
        return await leaves.add(prim, (req.x, req.y, req.z), color, req.size)

    async def change_shape(req: _ChangeShapeRequest) -> MovedObject:
        leaves.check_writable()
        prim = leaves.shape(req.prim_type)
        current = await leaves.find(req.object_words)
        result = await leaves.update({"obj_id": current.id, "prim_type": prim})
        # A shape change replaces the object; the scene returns its new id.
        return MovedObject(
            obj_id=result.new_id or current.id, x=current.position.x, y=current.position.y, z=current.position.z
        )

    async def resize_object(req: _ResizeRequest) -> MovedObject:
        current = await leaves.find(req.object_words)
        return await leaves.resize(current, req.factor)

    async def remove_object(req: _ObjRequest) -> RemovedObject:
        leaves.check_writable()
        target = await leaves.find(req.object_words)
        await leaves.remove(target.id)
        return RemovedObject(obj_id=target.id)

    return [
        Tool(
            "create_linear_arrangement",
            "Create a requested horizontal row or vertical stack of new identical objects directly "
            "in its final aligned layout. Preserve a stated spacing; otherwise use touching objects. "
            "Use one call instead of separate default creations and later moves.",
            _CreateLinearArrangementRequest,
            CreatedObjects,
            create_linear_arrangement,
            return_direct=True,
            render_result=lambda result: (
                f"Created the requested arrangement of {len(result.objects)} new objects."
            ),
            examples=(
                "For a vertical stack of four new objects, use layout vertical_stack and count=4.",
                "For a row with a stated separation, use layout horizontal_row and copy that "
                "center-to-center spacing exactly.",
            ),
        ),
        Tool(
            "create_user_relative",
            "Default creation tool when the instruction names no anchor object or world coordinates, "
            "even if unrelated scene objects already exist. Create a new object at a point in a "
            "named direction from the user. The shape may be "
            "a speech-to-text-mangled word; pass it verbatim because this tool repairs it. One call "
            "creates exactly one object, so a requested count requires one call per object. With "
            "no stated position use direction front and distance 1.5, regardless of other scene "
            "objects. Use direction at only for the user's current location; use direction below "
            "and distance about 1.5 for the floor or the user's feet.",
            _CreateUserRelativeRequest,
            CreatedObject,
            create_user_relative,
            examples=(
                "When a possibly misspelled color adjective precedes a clear shape noun, put only "
                "the noun in prim_type and use color_kind literal with the adjective in color_value.",
                "A named color such as amber is literal, not a physical source.",
                "A stated user-relative distance belongs only in distance_from_user_meters; keep "
                "object_radius_or_half_edge_meters at its default unless dimensions are separately specified.",
                "For an ordinary object requested two metres forward, set distance_from_user_meters=2 "
                "and object_radius_or_half_edge_meters=0.1; direction or eye-height wording does not "
                "change its dimensions.",
                "For an object at the user's feet, set direction=below and "
                "distance_from_user_meters=1.5; never use direction=at.",
                "For an object one metre to the user's left, set direction=left and "
                "distance_from_user_meters=1; do not keep the default distance.",
                "With no position words, use the default front position even when another object "
                "already occupies it; do not invent a relation to that object.",
                "For a new object matching a held or worn item, use color_kind physical and keep "
                "the complete source phrase in color_value.",
            ),
        ),
        Tool(
            "create_object_relative",
            "Use only when the instruction explicitly names an existing anchor object in a spatial "
            "relationship. Create a new object relative to one anchor object, or at the midpoint "
            "between two anchor objects "
            "(set second_anchor_words for 'between X and Y'). Copy anchor wording and call the tool "
            "even when the scene uses a different spelling; its fuzzy resolver decides the match. "
            "In 'create NEW-TARGET above ANCHOR', prim_type and color describe the new target and "
            "anchor_words is everything after the relation; this is creation, never recoloring the anchor.",
            _CreateObjectRelativeRequest,
            CreatedObject,
            create_object_relative,
            examples=(
                "For creation above a possibly mangled anchor, call this tool and copy the anchor "
                "phrase unchanged; never decline based on its spelling or apparent absence.",
                "For a new maroon block above a cyan bawl, pass cyan bawl unchanged as anchor_words "
                "so the tool can resolve the spoken shape.",
            ),
        ),
        Tool(
            "create_at",
            "Create a new object at explicit world coordinates.",
            _CreateAtRequest,
            CreatedObject,
            create_at,
        ),
        Tool(
            "change_shape",
            "Change an existing object into another primitive type, keeping position, color, and "
            "size. This cannot apply a color named in the same instruction; after the call, report "
            "that color as a remaining separate operation.",
            _ChangeShapeRequest,
            MovedObject,
            change_shape,
            return_direct=True,
            render_result=lambda result: (
                f"Changed the shape of {result.obj_id}; its color was not changed."
            ),
        ),
        Tool(
            "resize_object",
            "Required for every size change to an existing object. Multiply its size by factor 2 "
            "for twice/double or 0.5 for half, keeping everything else unchanged. Invoke the tool; "
            "never print a hypothetical call as ordinary text.",
            _ResizeRequest,
            MovedObject,
            resize_object,
            examples=(
                "For an unquantified request to enlarge an existing object, call once with "
                "factor=1.5.",
            ),
        ),
        Tool(
            "remove_object",
            "Required exactly once for every focused remove, delete, or erase instruction about an "
            "existing object; execute it before replying.",
            _ObjRequest,
            RemovedObject,
            remove_object,
        ),
    ]


__all__ = [
    "CreatedObject",
    "CreationLedger",
    "MovedObject",
    "RecoloredObject",
    "RemovedObject",
    "SwappedObjects",
    "TurnGuard",
    "make_appearance_tools",
    "make_object_tools",
    "make_placement_tools",
]
