# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a file-backed task and own session-local participant progress."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TaskStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str
    instructions: str
    visual_completion_criteria: str
    knowledge_files: list[str] = Field(default_factory=list)


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str
    step_files: list[str] = Field(min_length=1)


class TaskProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str
    task_id: str
    state: Literal["not_started", "running", "completed"] = "not_started"
    revision: int = 0
    step_index: int = 0
    transitions: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    citation: str
    text: str
    tokens: frozenset[str]


class TaskStore:
    """Validate one task folder and keep code-owned progress in memory."""

    def __init__(self, task_directory: Path) -> None:
        self.task_directory = task_directory.resolve()
        self.definition = TaskDefinition.model_validate(
            yaml.safe_load((self.task_directory / "workflow.yaml").read_text(encoding="utf-8"))
        )
        self.steps = tuple(self._load_step(path) for path in self.definition.step_files)
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("task step IDs must be unique")
        self.knowledge = tuple(self._load_knowledge())
        self._progress: dict[str, TaskProgress] = {}
        self._lock = Lock()

    def _resolve_inside(self, relative: str) -> Path:
        path = (self.task_directory / relative).resolve()
        if not path.is_relative_to(self.task_directory):
            raise ValueError(f"task path escapes task directory: {relative}")
        return path

    def _load_step(self, relative: str) -> TaskStep:
        path = self._resolve_inside(relative)
        step = TaskStep.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        for knowledge in step.knowledge_files:
            candidate = self._resolve_inside(knowledge)
            if not candidate.is_file():
                raise ValueError(f"missing knowledge file: {knowledge}")
        return step

    @staticmethod
    def _tokens(text: str) -> frozenset[str]:
        return frozenset(re.findall(r"[a-z0-9]+", text.casefold()))

    def _load_knowledge(self):
        paths = {self._resolve_inside(relative) for step in self.steps for relative in step.knowledge_files}
        for path in sorted(paths):
            text = path.read_text(encoding="utf-8")
            text = re.sub(r"\A<!--.*?-->\s*", "", text, flags=re.DOTALL)
            for index, paragraph in enumerate(part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()):
                yield KnowledgeChunk(
                    citation=f"{path.relative_to(self.task_directory)}#chunk-{index + 1}",
                    text=paragraph,
                    tokens=self._tokens(paragraph),
                )

    def _load_progress_unlocked(self, participant_id: str) -> TaskProgress:
        return self._progress.get(
            participant_id,
            TaskProgress(participant_id=participant_id, task_id=self.definition.id),
        )

    def progress(self, participant_id: str) -> TaskProgress:
        with self._lock:
            return self._load_progress_unlocked(participant_id)

    def current_step(self, progress: TaskProgress) -> TaskStep | None:
        return None if progress.state == "completed" else self.steps[progress.step_index]

    def next_step(self, progress: TaskProgress) -> TaskStep | None:
        next_index = progress.step_index + 1
        return self.steps[next_index] if progress.state != "completed" and next_index < len(self.steps) else None

    def _save_unlocked(self, progress: TaskProgress) -> None:
        self._progress[progress.participant_id] = progress

    def start(self, participant_id: str) -> TaskProgress:
        with self._lock:
            progress = self._load_progress_unlocked(participant_id)
            if progress.state != "not_started":
                return progress
            progress = progress.model_copy(
                update={
                    "state": "running",
                    "revision": progress.revision + 1,
                    "transitions": [*progress.transitions, "start"],
                }
            )
            self._save_unlocked(progress)
            return progress

    def reset(self, participant_id: str) -> TaskProgress:
        with self._lock:
            progress = self._load_progress_unlocked(participant_id)
            progress = progress.model_copy(
                update={
                    "state": "not_started",
                    "revision": progress.revision + 1,
                    "step_index": 0,
                    "transitions": [*progress.transitions, "reset"],
                }
            )
            self._save_unlocked(progress)
            return progress

    def advance(self, participant_id: str) -> TaskProgress:
        with self._lock:
            progress = self._load_progress_unlocked(participant_id)
            if progress.state == "not_started":
                raise ValueError("task has not started")
            if progress.state == "completed":
                return progress
            next_index = progress.step_index + 1
            progress = progress.model_copy(
                update={
                    "state": "completed" if next_index >= len(self.steps) else "running",
                    "revision": progress.revision + 1,
                    "step_index": min(next_index, len(self.steps) - 1),
                    "transitions": [*progress.transitions, "advance"],
                }
            )
            self._save_unlocked(progress)
            return progress

    def search(self, query: str, *, limit: int) -> list[KnowledgeChunk]:
        terms = self._tokens(query)
        ranked = sorted(
            self.knowledge,
            key=lambda chunk: (len(terms & chunk.tokens), chunk.citation),
            reverse=True,
        )
        return [chunk for chunk in ranked if terms & chunk.tokens][:limit]


__all__ = [
    "KnowledgeChunk",
    "TaskDefinition",
    "TaskProgress",
    "TaskStep",
    "TaskStore",
]
