# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative scene state and movement history for the workflow."""

from __future__ import annotations

from nat.plugin_api import Function
from xr_ai_nat.functions.xr_tracking import HeadPoseRequest
from xr_render_scene import EmptyRequest, SceneState


class SceneContext:
    """Read authoritative scene state and retain the last movement for undo."""

    def __init__(self, get_scene_state: Function, get_user_frame: Function | None = None) -> None:
        self._get_scene_state = get_scene_state
        self._get_user_frame = get_user_frame
        self._recent_moves: dict[str, list[str]] = {}
        self._recent_moves_age: dict[str, int] = {}
        self._mutating_delegations: set[str] = set()
        self._any_delegations: set[str] = set()

    def mark_mutating(self, participant_id: str) -> None:
        """Flag that this turn delegated mutating work."""
        self._mutating_delegations.add(participant_id)
        self._any_delegations.add(participant_id)

    def mark_delegated(self, participant_id: str) -> None:
        """Flag that this turn delegated any subagent work."""
        self._any_delegations.add(participant_id)

    def take_delegated(self, participant_id: str) -> bool:
        """Consume and return whether the turn delegated at all."""
        if participant_id in self._any_delegations:
            self._any_delegations.discard(participant_id)
            return True
        return False

    def take_mutating(self, participant_id: str) -> bool:
        """Consume and return whether the turn delegated any mutating work."""
        if participant_id in self._mutating_delegations:
            self._mutating_delegations.discard(participant_id)
            return True
        return False

    async def snapshot(self) -> SceneState:
        """Return the renderer's current typed scene state."""
        return await self._get_scene_state.ainvoke(EmptyRequest())

    async def describe(self, participant_id: str, *, bearings: bool = False) -> str:
        """Format the current scene and participant's last movement for a subagent."""
        state = await self.snapshot()
        parts = [f"[SCENE OBJECTS]\n{state.model_dump_json()}"]
        if bearings and (computed := await self._bearings(state)):
            parts.append(
                "[Object bearings from the user] (computed; +right/-left, +ahead/-behind, +up/-down)\n" + computed
            )
        if moves := self._recent_moves.get(participant_id):
            parts.append("[Recent moves]\n" + "\n".join(moves))
        return "\n\n".join(parts)

    async def _bearings(self, state: SceneState) -> str:
        if self._get_user_frame is None or not state.objects:
            return ""
        try:
            frame = await self._get_user_frame.ainvoke(HeadPoseRequest())
        except Exception:
            return ""
        lines = []
        for item in state.objects:
            dx = item.position.x - frame.origin.x
            dy = item.position.y - frame.origin.y
            dz = item.position.z - frame.origin.z
            right = dx * frame.right.x + dy * frame.right.y + dz * frame.right.z
            ahead = dx * frame.forward.x + dy * frame.forward.y + dz * frame.forward.z
            lines.append(f"  {item.id}: {right:+.2f} right, {ahead:+.2f} ahead, {dy:+.2f} up")
        return "\n".join(lines)

    @staticmethod
    def changes(before: SceneState, after: SceneState) -> str:
        """Summarize what actually changed between two snapshots; empty if nothing."""
        old = {item.id: item for item in before.objects}
        new = {item.id: item for item in after.objects}
        lines = []
        for object_id in sorted(set(new) - set(old)):
            lines.append(f"added {object_id}")
        for object_id in sorted(set(old) - set(new)):
            lines.append(f"removed {object_id}")
        for object_id in sorted(set(old) & set(new)):
            if old[object_id] != new[object_id]:
                was, now = old[object_id], new[object_id]
                details = []
                if was.position != now.position:
                    details.append(
                        f"position ({was.position.x}, {was.position.y}, {was.position.z})"
                        f" -> ({now.position.x}, {now.position.y}, {now.position.z})"
                    )
                if was.color != now.color:
                    details.append("color changed")
                if was.size != now.size:
                    details.append(f"size {was.size} -> {now.size}")
                if was.type != now.type:
                    details.append(f"type {was.type} -> {now.type}")
                lines.append(f"changed {object_id}: " + ", ".join(details))
        return "; ".join(lines)

    @staticmethod
    def positions(state: SceneState) -> dict[str, tuple[float, float, float]]:
        return {item.id: (item.position.x, item.position.y, item.position.z) for item in state.objects}

    async def record_moves(self, participant_id: str, before: SceneState) -> None:
        """Record positions changed during one turn so a later undo can restore them."""
        before_positions = self.positions(before)
        after_positions = self.positions(await self.snapshot())
        moves = [
            f"{object_id}: previously at {before_positions[object_id]}, now at {position}"
            for object_id, position in after_positions.items()
            if object_id in before_positions and before_positions[object_id] != position
        ]
        # A turn that moved nothing keeps the previous turn's undo data
        # alive for one more turn; older records would restore stale
        # positions on a much later "put it back".
        if moves:
            self._recent_moves[participant_id] = moves
            self._recent_moves_age[participant_id] = 0
        elif participant_id in self._recent_moves:
            self._recent_moves_age[participant_id] = self._recent_moves_age.get(participant_id, 0) + 1
            if self._recent_moves_age[participant_id] > 1:
                del self._recent_moves[participant_id]
                del self._recent_moves_age[participant_id]


__all__ = ["SceneContext"]
