# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code-composed movement and creation operations for the render subagents.

Every operation takes only semantic arguments (object ids, direction names,
metres) and resolves the user frame, anchor coordinates, geometry, and the
scene write itself by composing the leaf NAT functions. No pose or coordinate
ever round-trips through the agent LLM, which cannot relay them reliably.
"""

import asyncio
import difflib
import re
from typing import Annotated, Literal

from loguru import logger
from nat.plugin_api import (
    Builder,
    Function,
    FunctionGroup,
    FunctionGroupBaseConfig,
    FunctionGroupRef,
    register_function_group,
)
from pydantic import BaseModel, ConfigDict, Field
from xr_render_scene import EmptyRequest, SceneObject

_UserDirection = Literal["front", "back", "left", "right", "above", "below"]
_AnchorRelation = Literal["toward_user", "away_from_user", "left_of", "right_of", "above", "below"]

_ObjectWords = Annotated[
    str,
    Field(description="The instruction's exact words for this object, copied verbatim (mangled nouns fine); "
                      "an id only when the instruction itself states that id."),
]

_AnchorWords = Annotated[
    str,
    Field(description="The instruction's exact words for the anchor object, copied verbatim; "
                      "never an id chosen from the scene, unless the instruction itself states the id."),
]

_ColorWords = Annotated[
    str,
    Field(default="",
          description="The instruction's exact color word(s), copied verbatim (mangled spellings fine), "
                      "or an object to copy the color from ('same as cone-7'); empty when no color is stated."),
]

_ShapeWords = Annotated[
    str,
    Field(description="The instruction's exact shape word, copied verbatim (mangled spellings fine)."),
]

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
    """Final position applied to one existing object."""

    obj_id: str
    x: float
    y: float
    z: float


class SwappedObjects(BaseModel):
    """Final positions applied by one swap."""

    first: MovedObject
    second: MovedObject


class CreatedObject(BaseModel):
    """Stable id and position of one newly created object."""

    id: str
    x: float
    y: float
    z: float
    created_this_turn: int = Field(default=1, description="Objects created so far in this turn, including this one.")


class RemovedObject(BaseModel):
    """Stable id of one removed object."""

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
    """Block mutations of existing objects after a failed reference lookup.

    A model that cannot resolve a reference substitutes a plausible scene id
    on retry; once any lookup fails within a delegation, moves and edits must
    stop and the failure must travel back to the user. Creations stay allowed
    so a failed anchor can still degrade into a bare create.
    """

    def __init__(self) -> None:
        self.halted = False

    def reset(self) -> None:
        self.halted = False


class _Leaves:
    """Resolution helpers over the leaf NAT functions."""

    def __init__(
        self,
        functions: dict[str, Function],
        ledger: CreationLedger | None = None,
        guard: TurnGuard | None = None,
    ) -> None:
        self._functions = functions
        self.ledger = ledger
        self.guard = guard
        self._add_lock = asyncio.Lock()

    async def update(self, arguments: dict) -> None:
        await self._functions["scene_updates__update_primitive"].ainvoke(arguments)

    async def remove(self, object_id: str) -> None:
        await self._functions["scene_objects__remove_primitive"].ainvoke({"obj_id": object_id})

    def check_writable(self) -> None:
        if self.guard is not None and self.guard.halted:
            raise ValueError(
                "An earlier object reference in this instruction could not be resolved; "
                "change nothing else and report that failure back."
            )

    async def user_frame(self):
        return await self._functions["tracking__get_user_frame"].ainvoke({})

    async def find(self, object_ref: str) -> SceneObject:
        # Models sometimes emit ids with unicode dashes or stray whitespace.
        wanted = "".join("-" if character in "\u2010\u2011\u2012\u2013\u2014\u2212" else character
                         for character in object_ref).strip().lower()
        wanted = re.sub(r"[\s_]+", "-", wanted) if re.fullmatch(r"[A-Za-z]+[\s_-]+\d+", wanted) else wanted
        state = await self._functions["scene_state__get_scene_state"].ainvoke(EmptyRequest())
        for item in state.objects:
            if item.id == wanted:
                return item
        known = sorted(item.id for item in state.objects)
        # An id-shaped miss is a typo, never a description; resolving it as
        # one would silently pick a sibling of the mistyped id. Models do
        # swap shape synonyms into id prefixes ("cube-39" for box-39), and
        # that mapping is exact, so it resolves.
        if re.fullmatch(r"[a-z]+-\d+", wanted):
            prefix, _, number = wanted.partition("-")
            synonym_id = f"{_SHAPE_WORDS.get(prefix, prefix)}-{number}"
            for item in state.objects:
                if item.id == synonym_id:
                    logger.debug("spatial op resolved {!r} -> {}", object_ref, item.id)
                    return item
            logger.debug("spatial op lookup failed: {!r} not in {}", object_ref, known)
            raise ValueError(f"No scene object with id {object_ref!r}; the scene has {known}")
        # Not an id: read it as the instruction's own words, possibly
        # mangled by speech transcription. Exact shape and color words
        # resolve first; the fuzzy shape match runs only when they yield
        # nothing, so a phrase carrying a color ("the red one") never trips
        # over everyday words that sound like shapes.
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
            # A stated color that matches nothing must report back, never
            # silently pick the least-wrong object (squared-RGB threshold).
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
        # A vision fact arrives as a numeric triple ("RGB (1.0, 0.5, 0.0)").
        # Ids also carry digits ("same as capsule-0"), so the numeric read
        # only applies when no id-shaped token is present.
        if not re.search(r"[a-z]+-\d+", color_words.lower()):
            numbers = [float(value) for value in re.findall(r"-?\d*\.\d+|-?\d+", color_words)]
            if len(numbers) == 3 and all(0.0 <= value <= 1.0 for value in numbers):
                return (numbers[0], numbers[1], numbers[2])
        words = re.findall(r"[a-z]+", color_words.lower())
        if not words:
            return _DEFAULT_COLOR
        for word in words:
            if word in _COLOR_WORDS:
                return _COLOR_WORDS[word]
        # The object-copy lookup is speculative, so a miss must not trip
        # the turn guard, and it runs before the fuzzy match because
        # everyday words sit within difflib range of color names
        # ("cone" -> "orange").
        halted = self.guard.halted if self.guard is not None else False
        try:
            source = await self.find(color_words)
            return (source.color.r, source.color.g, source.color.b)
        except Exception:
            if self.guard is not None:
                self.guard.halted = halted
        # 0.75 admits transcription slips ("blew") while rejecting unrelated
        # words that drift within 0.6 of a color name.
        for word in words:
            close = difflib.get_close_matches(word, _COLOR_WORDS, n=1, cutoff=0.75)
            if close:
                logger.debug("color words resolved {!r} -> {}", color_words, close[0])
                return _COLOR_WORDS[close[0]]
        known = ", ".join(sorted(_COLOR_WORDS))
        raise ValueError(f"Unknown color {color_words!r}; use one of {known}, or name a scene object")

    async def spot(self, operation: str, arguments: dict) -> tuple[float, float, float]:
        result = await self._functions[f"spatial__{operation}"].ainvoke(arguments)
        return result.x, result.y, result.z

    async def write(self, object_id: str, position: tuple[float, float, float]) -> MovedObject:
        self.check_writable()
        x, y, z = position
        await self._functions["scene_updates__update_primitive"].ainvoke({"obj_id": object_id, "x": x, "y": y, "z": z})
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
        # The lock covers the memo's check-then-act: identical creates
        # gathered in one model response must still dedupe.
        async with self._add_lock:
            if self.ledger is not None and (existing := self.ledger.get(key)) is not None:
                return existing
            result = await self._functions["scene_objects__add_primitive"].ainvoke(
                {"prim_type": prim_type, "x": x, "y": y, "z": z, "r": r, "g": g, "b": b, "size": size}
            )
            created = CreatedObject(id=result.id, x=x, y=y, z=z)
            if self.ledger is not None:
                self.ledger.count += 1
                created.created_this_turn = self.ledger.count
                self.ledger.record(key, created)
            return created


async def _leaf_functions(builder: Builder, refs: tuple[FunctionGroupRef, ...]) -> dict[str, Function]:
    functions: dict[str, Function] = {}
    for ref in refs:
        group = await builder.get_function_group(ref)
        functions.update(await group.get_all_functions())
    return functions


class PlacementOpsConfig(FunctionGroupBaseConfig, name="xr_render_placement_ops"):
    """Configure the composed movement operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    guard: TurnGuard | None = Field(default=None, exclude=True, repr=False)
    scene_state: FunctionGroupRef = FunctionGroupRef("scene_state")
    scene_updates: FunctionGroupRef = FunctionGroupRef("scene_updates")
    tracking: FunctionGroupRef = FunctionGroupRef("tracking")
    spatial: FunctionGroupRef = FunctionGroupRef("spatial")


@register_function_group(config_type=PlacementOpsConfig)
async def placement_ops(config: PlacementOpsConfig, builder: Builder):
    """Build movement operations that resolve geometry and write in code."""
    leaves = _Leaves(
        await _leaf_functions(
            builder,
            (config.scene_state, config.scene_updates, config.tracking, config.spatial),
        ),
        guard=config.guard,
    )
    group = FunctionGroup(config=config)

    async def move_user_relative(
        object_words: _ObjectWords,
        direction: _UserDirection,
        distance: Annotated[
            float, Field(description="Distance from the user in metres; pass a stated distance exactly.")
        ] = 1.5,
    ) -> MovedObject:
        target = await leaves.find(object_words)
        frame = await leaves.user_frame()
        spot = await leaves.spot(
            "compute_user_relative_position",
            {"user_frame": frame.model_dump(), "direction_from_user": direction, "distance_meters": distance},
        )
        return await leaves.write(target.id, spot)

    group.add_function(
        "move_user_relative",
        move_user_relative,
        description=(
            "Move an existing object to a point in a named direction from the user. Not for stated shifts "
            "like 'one metre to my left'; nudge does those."
        ),
    )

    async def nudge(
        object_words: _ObjectWords,
        forward: Annotated[float, Field(description="Signed user-forward shift in metres.")] = 0.0,
        right: Annotated[float, Field(description="Signed user-right shift in metres.")] = 0.0,
        up: Annotated[float, Field(description="Signed world-up shift in metres.")] = 0.0,
    ) -> MovedObject:
        current = await leaves.find(object_words)
        frame = await leaves.user_frame()
        spot = await leaves.spot(
            "offset_position_in_user_frame",
            {
                "user_frame": frame.model_dump(),
                "start_position": current.position.model_dump(),
                "forward_meters": forward,
                "right_meters": right,
                "up_meters": up,
            },
        )
        return await leaves.write(current.id, spot)

    group.add_function(
        "nudge",
        nudge,
        description="Shift an existing object from its current position by signed user-frame offsets.",
    )

    async def move_object_relative(
        movee_words: _ObjectWords,
        anchor_words: _AnchorWords,
        relation: _AnchorRelation,
        distance: Annotated[
            float, Field(description="Distance from the anchor in metres; pass a stated distance exactly.")
        ] = 0.3,
    ) -> MovedObject:
        logger.debug("move_object_relative movee={!r} anchor={!r} relation={} distance={}",
                     movee_words, anchor_words, relation, distance)
        movee = await leaves.find(movee_words)
        anchor = await leaves.find(anchor_words)
        if movee.id == anchor.id:
            raise ValueError(
                f"{movee_words!r} and {anchor_words!r} are the same object ({movee.id}); an object cannot "
                "be placed relative to itself. Make no other tool call and report the problem back."
            )
        frame = await leaves.user_frame()
        spot = await leaves.spot(
            "compute_position_relative_to_anchor",
            {
                "user_frame": frame.model_dump(),
                "anchor_position": anchor.position.model_dump(),
                "relation_to_anchor": relation,
                "distance_meters": distance,
            },
        )
        return await leaves.write(movee.id, spot)

    group.add_function(
        "move_object_relative",
        move_object_relative,
        description="Move an existing object to a point in a named relation to an anchor object.",
    )

    async def move_inside(movee_words: _ObjectWords, container_words: _ObjectWords) -> MovedObject:
        movee = await leaves.find(movee_words)
        container = await leaves.find(container_words)
        return await leaves.write(
            movee.id,
            (container.position.x, container.position.y, container.position.z),
        )

    group.add_function(
        "move_inside",
        move_inside,
        description="Move an existing object into the center of a container object.",
    )

    async def move_between(
        movee_words: _ObjectWords,
        first_anchor_words: _ObjectWords,
        second_anchor_words: _ObjectWords,
    ) -> MovedObject:
        movee = await leaves.find(movee_words)
        anchor_a = await leaves.find(first_anchor_words)
        anchor_b = await leaves.find(second_anchor_words)
        spot = await leaves.spot(
            "compute_midpoint",
            {"first_position": anchor_a.position.model_dump(), "second_position": anchor_b.position.model_dump()},
        )
        return await leaves.write(movee.id, spot)

    group.add_function(
        "move_between",
        move_between,
        description="Move an existing object to the midpoint between two anchor objects.",
    )

    async def move_toward(
        movee_words: _ObjectWords,
        target_words: _ObjectWords,
        direction: Annotated[
            Literal["toward", "away"], Field(description="Move toward or away from the target.")
        ] = "toward",
        distance: Annotated[float, Field(ge=0, description="Non-negative travel distance in metres.")] = 0.5,
    ) -> MovedObject:
        movee = await leaves.find(movee_words)
        target = await leaves.find(target_words)
        spot = await leaves.spot(
            "compute_position_toward_or_away_from_reference",
            {
                "start_position": movee.position.model_dump(),
                "reference_position": target.position.model_dump(),
                "movement_direction": direction,
                "distance_meters": distance,
            },
        )
        return await leaves.write(movee.id, spot)

    group.add_function(
        "move_toward",
        move_toward,
        description="Move an existing object toward or away from another object.",
    )

    async def move_toward_user(
        movee_words: _ObjectWords,
        direction: Annotated[
            Literal["toward", "away"], Field(description="Move toward or away from the user.")
        ] = "toward",
        distance: Annotated[float, Field(ge=0, description="Non-negative travel distance in metres.")] = 0.5,
    ) -> MovedObject:
        movee = await leaves.find(movee_words)
        frame = await leaves.user_frame()
        spot = await leaves.spot(
            "compute_position_toward_or_away_from_reference",
            {
                "start_position": movee.position.model_dump(),
                "reference_position": frame.origin.model_dump(),
                "movement_direction": direction,
                "distance_meters": distance,
            },
        )
        return await leaves.write(movee.id, spot)

    group.add_function(
        "move_toward_user",
        move_toward_user,
        description="Move an existing object toward or away from the user.",
    )

    async def swap_positions(first_words: _ObjectWords, second_words: _ObjectWords) -> SwappedObjects:
        first = await leaves.find(first_words)
        second = await leaves.find(second_words)
        first_position = (first.position.x, first.position.y, first.position.z)
        second_position = (second.position.x, second.position.y, second.position.z)
        return SwappedObjects(
            first=await leaves.write(first.id, second_position),
            second=await leaves.write(second.id, first_position),
        )

    group.add_function(
        "swap_positions",
        swap_positions,
        description="Exchange the positions of two existing objects.",
    )

    async def move_to(object_words: _ObjectWords, x: float, y: float, z: float) -> MovedObject:
        target = await leaves.find(object_words)
        return await leaves.write(target.id, (x, y, z))

    group.add_function(
        "move_to",
        move_to,
        description=(
            "Move an existing object to explicit world coordinates taken from the request, SCENE OBJECTS, or "
            "[Recent moves]; never invent coordinates."
        ),
    )

    yield group


class RecoloredObject(BaseModel):
    """Final color applied to one existing object."""

    obj_id: str
    r: float
    g: float
    b: float


class AppearanceOpsConfig(FunctionGroupBaseConfig, name="xr_render_appearance_ops"):
    """Configure the composed recolor operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    guard: TurnGuard | None = Field(default=None, exclude=True, repr=False)
    scene_state: FunctionGroupRef = FunctionGroupRef("scene_state")
    scene_updates: FunctionGroupRef = FunctionGroupRef("scene_updates")


@register_function_group(config_type=AppearanceOpsConfig)
async def appearance_ops(config: AppearanceOpsConfig, builder: Builder):
    """Build the recolor operation that resolves references and colors in code."""
    leaves = _Leaves(
        await _leaf_functions(builder, (config.scene_state, config.scene_updates)),
        guard=config.guard,
    )
    group = FunctionGroup(config=config)

    async def recolor(
        object_words: _ObjectWords,
        color_words: Annotated[
            str,
            Field(description="The instruction's exact color word(s), copied verbatim (mangled "
                              "spellings fine), or an object to copy the color from ('same as cone-7')."),
        ],
    ) -> RecoloredObject:
        leaves.check_writable()
        target = await leaves.find(object_words)
        r, g, b = await leaves.color(color_words)
        await leaves.update({"obj_id": target.id, "r": r, "g": g, "b": b})
        return RecoloredObject(obj_id=target.id, r=r, g=g, b=b)

    group.add_function(
        "recolor",
        recolor,
        description="Change an existing object's color, keeping position, type, and size.",
    )

    yield group


class ObjectOpsConfig(FunctionGroupBaseConfig, name="xr_render_object_ops"):
    """Configure the composed creation and reshape operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ledger: CreationLedger | None = Field(default=None, exclude=True, repr=False)
    guard: TurnGuard | None = Field(default=None, exclude=True, repr=False)
    scene_state: FunctionGroupRef = FunctionGroupRef("scene_state")
    scene_updates: FunctionGroupRef = FunctionGroupRef("scene_updates")
    scene_objects: FunctionGroupRef = FunctionGroupRef("scene_objects")
    tracking: FunctionGroupRef = FunctionGroupRef("tracking")
    spatial: FunctionGroupRef = FunctionGroupRef("spatial")


@register_function_group(config_type=ObjectOpsConfig)
async def object_ops(config: ObjectOpsConfig, builder: Builder):
    """Build creation and reshape operations that resolve geometry in code."""
    leaves = _Leaves(
        await _leaf_functions(
            builder,
            (config.scene_state, config.scene_updates, config.scene_objects, config.tracking, config.spatial),
        ),
        ledger=config.ledger,
        guard=config.guard,
    )
    group = FunctionGroup(config=config)

    async def create_user_relative(
        prim_type: _ShapeWords,
        direction: _UserDirection,
        color_words: _ColorWords = "",
        distance: Annotated[
            float, Field(description="Distance from the user in metres; pass a stated distance exactly.")
        ] = 1.5,
        size: Annotated[float, Field(description="Sphere radius or box half-edge in metres.")] = 0.1,
    ) -> CreatedObject:
        prim = leaves.shape(prim_type)
        color = await leaves.color(color_words)
        frame = await leaves.user_frame()
        spot = await leaves.spot(
            "compute_user_relative_position",
            {"user_frame": frame.model_dump(), "direction_from_user": direction, "distance_meters": distance},
        )
        return await leaves.add(prim, spot, color, size)

    group.add_function(
        "create_user_relative",
        create_user_relative,
        description="Create a new object at a point in a named direction from the user.",
    )

    async def create_object_relative(
        prim_type: _ShapeWords,
        anchor_words: _AnchorWords,
        relation: _AnchorRelation,
        color_words: _ColorWords = "",
        distance: Annotated[
            float, Field(description="Distance from the anchor in metres; pass a stated distance exactly.")
        ] = 0.3,
        size: Annotated[float, Field(description="Sphere radius or box half-edge in metres.")] = 0.1,
    ) -> CreatedObject:
        logger.debug("create_object_relative anchor={!r} relation={} distance={}", anchor_words, relation, distance)
        prim = leaves.shape(prim_type)
        color = await leaves.color(color_words)
        try:
            anchor = await leaves.find(anchor_words)
        except ValueError as error:
            raise ValueError(
                f"{error}. If the instruction names no existing object to anchor on, this is a bare "
                "creation: call create_user_relative with direction front and distance 1.5 instead."
            ) from None
        frame = await leaves.user_frame()
        spot = await leaves.spot(
            "compute_position_relative_to_anchor",
            {
                "user_frame": frame.model_dump(),
                "anchor_position": anchor.position.model_dump(),
                "relation_to_anchor": relation,
                "distance_meters": distance,
            },
        )
        return await leaves.add(prim, spot, color, size)

    group.add_function(
        "create_object_relative",
        create_object_relative,
        description="Create a new object at a point in a named relation to an anchor object.",
    )

    async def create_at(
        prim_type: _ShapeWords,
        x: float,
        y: float,
        z: float,
        color_words: _ColorWords = "",
        size: Annotated[float, Field(description="Sphere radius or box half-edge in metres.")] = 0.1,
    ) -> CreatedObject:
        logger.debug("create_at ({}, {}, {})", x, y, z)
        prim = leaves.shape(prim_type)
        color = await leaves.color(color_words)
        return await leaves.add(prim, (x, y, z), color, size)

    group.add_function(
        "create_at",
        create_at,
        description="Create a new object at explicit world coordinates.",
    )

    async def change_shape(object_words: _ObjectWords, prim_type: _ShapeWords) -> MovedObject:
        leaves.check_writable()
        prim = leaves.shape(prim_type)
        current = await leaves.find(object_words)
        await leaves.update({"obj_id": current.id, "prim_type": prim})
        return MovedObject(obj_id=current.id, x=current.position.x, y=current.position.y, z=current.position.z)

    group.add_function(
        "change_shape",
        change_shape,
        description="Change an existing object into another primitive type, keeping position, color, and size.",
    )

    async def resize_object(
        object_words: _ObjectWords,
        factor: Annotated[float, Field(description="Multiplier applied to the current size.")],
    ) -> MovedObject:
        leaves.check_writable()
        current = await leaves.find(object_words)
        # Model retries re-apply multiplicative resizes; one factor per
        # object per turn.
        key = ("resize", current.id, round(factor, 4))
        async with leaves._add_lock:
            if leaves.ledger is not None and (done := leaves.ledger.mutations.get(key)) is not None:
                return done
            await leaves.update({"obj_id": current.id, "size": round(current.size * factor, 4)})
            result = MovedObject(obj_id=current.id, x=current.position.x, y=current.position.y, z=current.position.z)
            if leaves.ledger is not None:
                leaves.ledger.mutations[key] = result
            return result

    group.add_function(
        "resize_object",
        resize_object,
        description="Multiply an existing object's size by a factor, keeping everything else.",
    )

    async def remove_object(object_words: _ObjectWords) -> RemovedObject:
        leaves.check_writable()
        target = await leaves.find(object_words)
        await leaves.remove(target.id)
        return RemovedObject(obj_id=target.id)

    group.add_function(
        "remove_object",
        remove_object,
        description="Remove an existing object from the scene.",
    )

    yield group


__all__ = ["AppearanceOpsConfig", "CreationLedger", "ObjectOpsConfig", "PlacementOpsConfig", "TurnGuard"]
