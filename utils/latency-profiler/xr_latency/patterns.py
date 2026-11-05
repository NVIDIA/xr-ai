# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common agent patterns priced from a fitted model.

A pattern is ``patterns/<name>.yaml``: a title, a description and a list of estimator stages. Patterns are generic
shapes (a plain voice question, a routed question, a tool loop), not shipped samples, so they stay valid as samples
change and let two machines be compared on the same turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .estimate import Result, evaluate

PATTERNS = Path(__file__).resolve().parents[1] / "patterns"
_THINKING_STAGES = {"llm", "tool_hop", "speak"}


@dataclass(frozen=True)
class Pattern:
    """One named turn shape."""

    name: str
    title: str
    description: str
    stages: list[dict[str, Any]]

    def with_thinking(self, thinking: bool) -> list[dict[str, Any]]:
        """The stages with LLM thinking forced on or off (``reasoning_tokens`` stages are left as written)."""
        return [{**s, "thinking": thinking} if s["kind"] in _THINKING_STAGES else s for s in self.stages]


def load_patterns(directory: Path = PATTERNS) -> dict[str, Pattern]:
    """Load every ``*.yaml`` in ``directory``, keyed by file stem."""
    out = {}
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        out[path.stem] = Pattern(path.stem, raw["title"], raw.get("description", ""), raw["stages"])
    return out


def price(model: dict[str, Any], pattern: Pattern, *, thinking: bool | None = None) -> Result:
    """Price ``pattern`` from ``model``. ``thinking`` overrides the pattern's own thinking flags when not None."""
    return evaluate(model, pattern.stages if thinking is None else pattern.with_thinking(thinking))


def price_all(model: dict[str, Any], *, thinking: bool | None = None) -> dict[str, Result | str]:
    """Price every pattern; a pattern the model cannot price (benchmark not run) maps to the reason."""
    out: dict[str, Result | str] = {}
    for name, pattern in load_patterns().items():
        try:
            out[name] = price(model, pattern, thinking=thinking)
        except KeyError as exc:
            out[name] = f"not priced: {exc.args[0]}"
    return out
