# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group raw samples into summaries and fitted curves."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._stats import LinearFit, Summary, fit_linear, summarize
from ._store import read_samples

# A sample taken while another client had requests running on the vLLM servers is not a clean measurement.


def load(run_dirs: Iterable[Path], *, drop_busy: bool = True) -> list[dict[str, Any]]:
    """Read every sample from ``run_dirs``; optionally drop samples recorded under heavy foreign GPU load."""
    rows = [row for run in run_dirs for row in read_samples(run)]
    if not drop_busy:
        return rows
    return [r for r in rows if not r["case"].get("foreign")]


def _key(case: dict[str, Any], by: Sequence[str]) -> tuple[Any, ...]:
    return tuple(case.get(k) for k in by)


def group(rows: Iterable[dict[str, Any]], bench: str, metric: str, by: Sequence[str],
          where: dict[str, Any] | None = None) -> dict[tuple[Any, ...], Summary]:
    """Summarize ``metric`` for rows of ``bench``, keyed by the case fields in ``by``."""
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row["bench"] != bench or row.get(metric) is None:
            continue
        if where and any(row["case"].get(k) != v for k, v in where.items()):
            continue
        buckets[_key(row["case"], by)].append(float(row[metric]))
    return {k: summarize(v) for k, v in sorted(buckets.items(), key=lambda kv: tuple(map(str, kv[0]))) if v}


@dataclass(frozen=True, slots=True)
class Curve:
    """A fitted latency curve ``y = intercept + slope * x`` over measured group medians."""

    name: str
    x_label: str
    fit: LinearFit
    points: tuple[tuple[float, float], ...]


def fit_curve(name: str, x_label: str, groups: dict[tuple[Any, ...], Summary]) -> Curve | None:
    """Fit group p50 against a numeric single-key x; returns None when there are too few distinct x values."""
    points = sorted((float(k[0]), s.p50) for k, s in groups.items() if k[0] is not None)
    if len({x for x, _ in points}) < 2:
        return None
    return Curve(name, x_label, fit_linear([x for x, _ in points], [y for _, y in points]), tuple(points))


def dump_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as indented JSON."""
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)))
