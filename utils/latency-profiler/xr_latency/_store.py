# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append-only JSONL storage for raw latency samples and run metadata."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunStore:
    """One benchmark run directory holding ``samples.jsonl`` and ``meta.json``."""

    root: Path
    _handle: Any = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, base: Path, label: str = "") -> RunStore:
        """Create ``base/<timestamp>[-label]`` and return its store."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        root = base / (f"{stamp}-{label}" if label else stamp)
        root.mkdir(parents=True, exist_ok=False)
        return cls(root)

    def write_meta(self, meta: dict[str, Any]) -> None:
        """Merge run metadata (stack, machine fingerprint, calibration) into ``meta.json`` next to the samples."""
        path = self.root / "meta.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        path.write_text(json.dumps({**current, **meta}, indent=2, sort_keys=True))

    def add(self, bench: str, case: dict[str, Any], **metrics: float | int | str | None) -> None:
        """Append one sample: ``bench`` name, the swept ``case`` parameters, and measured metrics."""
        if self._handle is None:
            self._handle = (self.root / "samples.jsonl").open("a", buffering=1)
        record = {"t": time.time(), "bench": bench, "case": case, **metrics}
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self) -> None:
        """Flush and close the sample file."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def read_meta(run_dir: Path) -> dict[str, Any]:
    """Return a run directory's metadata, or an empty dict when it has none."""
    path = run_dir / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def read_samples(run_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield every sample record from a run directory."""
    path = run_dir / "samples.jsonl"
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
