# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuously validate and content-address locally generated SOP guides."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ._workflow_spec import Workflow, load_workflow

_SUFFIXES = (".guide.yaml", ".guide.yml")
_MAX_GUIDE_BYTES = 1_000_000


@dataclass(frozen=True, slots=True)
class CatalogGuide:
    """One immutable catalog generation safe to pin into a running session."""

    path: Path
    relative_path: str
    sha256: str
    modified_at: str
    workflow: Workflow | None
    error: str | None

    @property
    def runnable(self) -> bool:
        return self.workflow is not None and self.workflow.runnable

    def as_index_item(self) -> dict[str, Any]:
        workflow = self.workflow
        return {
            "id": workflow.id if workflow is not None else None,
            "title": workflow.name if workflow is not None else self.path.stem,
            "version": workflow.version if workflow is not None else None,
            "status": workflow.status if workflow is not None else "invalid",
            "runnable": self.runnable,
            "path": self.relative_path,
            "sha256": self.sha256,
            "modified_at": self.modified_at,
            "error": self.error,
        }


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class GuideCatalog:
    """Maintain an atomically replaced snapshot of validated guides."""

    def __init__(self, guides_dir: Path, index_path: Path, *, interval_s: float) -> None:
        self._guides_dir = guides_dir.resolve()
        self._index_path = index_path
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self._guides: tuple[CatalogGuide, ...] = ()

    @property
    def items(self) -> tuple[dict[str, Any], ...]:
        return tuple(guide.as_index_item() for guide in self._guides)

    @property
    def guides(self) -> tuple[CatalogGuide, ...]:
        return self._guides

    def resolve(self, selector: str) -> CatalogGuide:
        """Resolve one exact id, title, or unambiguous id prefix."""

        normalized = selector.strip().casefold()
        if not normalized:
            raise ValueError("name a guide to start")
        exact = [
            guide
            for guide in self._guides
            if guide.workflow is not None
            and normalized in {guide.workflow.id.casefold(), guide.workflow.name.casefold()}
        ]
        matches = exact or [
            guide
            for guide in self._guides
            if guide.workflow is not None and guide.workflow.id.casefold().startswith(normalized)
        ]
        if not matches:
            raise ValueError(f"no guide matches {selector!r}")
        if len(matches) > 1:
            ids = [guide.workflow.id for guide in matches if guide.workflow]
            raise ValueError(f"guide name is ambiguous: {ids}")
        guide = matches[0]
        if guide.error is not None or guide.workflow is None:
            raise ValueError(f"guide is invalid: {guide.error}")
        if not guide.runnable:
            raise ValueError(f"guide {guide.workflow.id!r} is a draft; review it and set task.status to approved")
        return guide

    async def start(self) -> None:
        self._guides_dir.mkdir(parents=True, exist_ok=True)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        await self._scan()
        self._task = asyncio.create_task(
            self._watch(),
            name="workflow-guide-discovery",
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            try:
                await self._scan()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.opt(exception=True).warning("workflow guide discovery failed")

    async def _scan(self) -> None:
        guides = self._read_guides()
        old = tuple((item.relative_path, item.sha256, item.error) for item in self._guides)
        new = tuple((item.relative_path, item.sha256, item.error) for item in guides)
        if old == new and self._index_path.exists():
            return
        self._guides = guides
        _atomic_json(
            self._index_path,
            {
                "schema_version": 2,
                "scanned_at": datetime.now(UTC).isoformat(),
                "guides_dir": str(self._guides_dir),
                "guides": [guide.as_index_item() for guide in guides],
            },
        )
        invalid = sum(guide.error is not None for guide in guides)
        logger.info("discovered {} workflow guide(s), {} invalid", len(guides), invalid)

    def _read_guides(self) -> tuple[CatalogGuide, ...]:
        guides: list[CatalogGuide] = []
        for path in sorted(candidate for candidate in self._guides_dir.rglob("*") if candidate.is_file()):
            relative = path.relative_to(self._guides_dir).as_posix()
            if not relative.endswith(_SUFFIXES):
                continue
            stat = path.lstat()
            if path.is_symlink() or not path.resolve().is_relative_to(self._guides_dir):
                guides.append(
                    CatalogGuide(
                        path=path,
                        relative_path=relative,
                        sha256="",
                        modified_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                        workflow=None,
                        error="symbolic-link guides are not allowed",
                    )
                )
                continue
            if stat.st_size > _MAX_GUIDE_BYTES:
                guides.append(
                    CatalogGuide(
                        path=path,
                        relative_path=relative,
                        sha256="",
                        modified_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                        workflow=None,
                        error=f"guide exceeds {_MAX_GUIDE_BYTES} bytes",
                    )
                )
                continue
            content = path.read_bytes()
            workflow: Workflow | None = None
            error: str | None = None
            try:
                workflow = load_workflow(path)
            except Exception as exc:
                error = str(exc)
            guides.append(
                CatalogGuide(
                    path=path,
                    relative_path=relative,
                    sha256=hashlib.sha256(content).hexdigest(),
                    modified_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                    workflow=workflow,
                    error=error,
                )
            )
        valid_ids = [guide.workflow.id for guide in guides if guide.workflow is not None]
        duplicates = {guide_id for guide_id in valid_ids if valid_ids.count(guide_id) > 1}
        if duplicates:
            guides = [
                CatalogGuide(
                    path=guide.path,
                    relative_path=guide.relative_path,
                    sha256=guide.sha256,
                    modified_at=guide.modified_at,
                    workflow=None if guide.workflow is not None and guide.workflow.id in duplicates else guide.workflow,
                    error=(
                        f"duplicate task.id {guide.workflow.id!r}"
                        if guide.workflow is not None and guide.workflow.id in duplicates
                        else guide.error
                    ),
                )
                for guide in guides
            ]
        return tuple(guides)
