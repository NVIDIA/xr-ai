# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover sample-owned speech replay files and load their timed inputs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Scenario:
    sample: str
    duration_s: float
    events: list[dict]
    worker: dict = field(default_factory=dict)
    source: str = ""

    @property
    def sha256(self) -> str:
        data = asdict(self)
        data.pop("source")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    @property
    def texts(self) -> list[str]:
        return list(dict.fromkeys(event["say"] for event in self.events if "say" in event))

    @property
    def images(self) -> set[str]:
        return {event["scene"]["image"] for event in self.events if "image" in event.get("scene", {})}


def _discover(root: Path) -> dict[str, Path]:
    files = sorted(
        path for folder in ("eval", "evaluation") for path in (root / "agent-samples").glob(f"*/{folder}/speech/*.yaml")
    )
    found = {}
    for path in files:
        if path.stem in found:
            raise ValueError(f"duplicate speech replay {path.stem}: {found[path.stem]} and {path}")
        found[path.stem] = path
    return found


def _load_scenario(path: Path) -> _Scenario:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or set(data) != {"duration_s", "worker", "events"}:
        raise ValueError(f"{path}: expected duration_s, worker and events")
    duration, worker, events = data["duration_s"], data["worker"], data["events"]
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{path}: duration_s must be finite and positive")
    if (
        not isinstance(worker, dict)
        or set(worker) - {"package", "config", "launch"}
        or any(not isinstance(worker.get(key), str) or not worker[key].strip() for key in ("package", "config"))
        or not isinstance(worker.get("launch", {}), dict)
    ):
        raise ValueError(f"{path}: worker needs package, config and optional launch mapping")
    launch = worker.get("launch", {})
    if "launch" in worker and "mode" not in launch:
        raise ValueError(f"{path}: launch requires an explicit mode")
    if set(launch) - {"mode", "guide_path"} or any(
        not isinstance(value, str) or not value.strip() for value in launch.values()
    ):
        raise ValueError(f"{path}: launch supports native mode and guide_path strings")
    if "guide_path" in launch and not (path.parent / launch["guide_path"]).is_file():
        raise ValueError(f"{path}: launch guide_path does not exist")
    if not (path.parent / worker["config"]).is_file():
        raise ValueError(f"{path}: worker config does not exist: {worker['config']}")
    if not isinstance(events, list) or not events:
        raise ValueError(f"{path}: events must be a nonempty list")
    previous = -1
    for event in events:
        if not isinstance(event, dict) or set(event) - {"at_s", "say", "scene", "handler"}:
            raise ValueError(f"{path}: invalid replay event")
        at = event.get("at_s")
        if type(at) not in (int, float) or not math.isfinite(at) or not previous <= at < duration or at < 0:
            raise ValueError(f"{path}: event times must be ordered and inside the replay duration")
        if (
            not ({"say", "scene"} & event.keys())
            or ("say" in event and (not isinstance(event["say"], str) or not event["say"].strip()))
            or ("scene" in event and (not isinstance(event["scene"], dict) or not event["scene"]))
            or ("handler" in event and ("say" not in event or type(event["handler"]) is not bool))
        ):
            raise ValueError(f"{path}: events need speech or a scene and an optional boolean handler flag")
        scene = event.get("scene", {})
        if "image" in scene and (
            set(scene) != {"image"} or not isinstance(scene["image"], str) or not scene["image"].strip()
        ):
            raise ValueError(f"{path}: image scenes must name one supplied image asset")
        previous = at
    return _Scenario(path.stem, duration, events, worker, str(path.resolve()))


_SCENARIOS = _discover(_ROOT)
_SAMPLES = tuple(_SCENARIOS)


def _builtin(name: str) -> _Scenario:
    if name not in _SCENARIOS:
        raise ValueError(f"unknown replay {name!r}; choose one of: {', '.join(_SAMPLES)}")
    return _load_scenario(_SCENARIOS[name])
