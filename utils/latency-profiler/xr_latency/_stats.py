# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summary statistics and simple linear fits for latency samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Summary:
    """Distribution summary of one latency series, in milliseconds."""

    n: int
    min: float
    p50: float
    p90: float
    p99: float
    max: float
    mean: float
    stdev: float


def summarize(values_ms: list[float]) -> Summary:
    """Summarize a non-empty series of millisecond values."""
    if not values_ms:
        raise ValueError("cannot summarize an empty series")
    arr = np.asarray(values_ms, dtype=float)
    p50, p90, p99 = np.percentile(arr, [50, 90, 99])
    return Summary(
        n=int(arr.size), min=float(arr.min()), p50=float(p50), p90=float(p90), p99=float(p99),
        max=float(arr.max()), mean=float(arr.mean()), stdev=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    )


@dataclass(frozen=True, slots=True)
class LinearFit:
    """``y = intercept + slope * x`` with its goodness of fit."""

    intercept: float
    slope: float
    r2: float
    n: int

    def predict(self, x: float) -> float:
        """Evaluate the fit at ``x``."""
        return self.intercept + self.slope * x


def fit_linear(x: list[float], y: list[float]) -> LinearFit:
    """Least-squares line through ``(x, y)``; needs at least two distinct x values."""
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xs.size < 2 or np.ptp(xs) == 0:
        raise ValueError("a linear fit needs at least two distinct x values")
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (intercept + slope * xs)
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else 1.0
    return LinearFit(float(intercept), float(slope), r2, int(xs.size))
