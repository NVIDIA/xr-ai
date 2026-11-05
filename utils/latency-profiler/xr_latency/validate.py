# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the estimator against measurements it was not directly fitted to: composed first-audio and router paths."""

from __future__ import annotations

from typing import Any

from .estimate import PATHS, evaluate

_STRATEGY = {"S1_chat_batch_tts": "batch", "S2_chat_stream_tts": "stream_full", "S3_stream_sentence_tts": "sentence"}
# measured router path -> (estimator path, history bytes)
_ROUTER = {
    "single_candidate": ("single_candidate", 0), "focused_fast_path": ("focused", 0),
    "multi_route_accept": ("multi_route", 0), "multi_route_accept_ctx12k": ("multi_route", 12_000),
    "misroute_declined_once": ("misroute_declined_once", 0),
    "declines_exhausted_default": ("declines_exhausted_default", 0),
    "all_ineligible_clarify": ("clarify", 0),
}


def _row(table: list[dict[str, Any]], **keys: Any) -> dict[str, Any] | None:
    return next((r for r in table if all(r.get(k) == v for k, v in keys.items())), None)


def first_audio_checks(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Predicted vs measured time to first audio for each delivery strategy and answer length."""
    out = []
    for toks in model["streaming"]["completion_tokens"]:
        n = int(toks["sentences"])
        for key, strategy in _STRATEGY.items():
            measured = _row(model["streaming"]["first_audio"], sentences=n, strategy=key)
            if not measured:
                continue
            spec = {"kind": "speak", "prompt_tokens": 0, "out_tokens": toks["p50"],
                    "first_sentence_tokens": toks["p50"] / n, "strategy": strategy, "cached": False}
            predicted = evaluate(model, [spec]).first_audio_p50
            out.append({"case": f"{n} sentence(s), {strategy}", "predicted": predicted, "measured": measured["p50"]})
    return out


def router_checks(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Predicted vs measured routing time before the destination starts."""
    out = []
    for name, (path, history) in _ROUTER.items():
        measured = _row(model["router"]["total"], path=name)
        if not measured or path not in PATHS:
            continue
        spec = {"kind": "route", "path": path, "history_bytes": history}
        calibration = name in ("multi_route_accept", "multi_route_accept_ctx12k")
        out.append({"case": name + (" (calibration point)" if calibration else ""),
                    "predicted": evaluate(model, [spec]).done_p50, "measured": measured["p50"],
                    "calibration": calibration})
    return out


def independent(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Checks whose measurement was not used to calibrate the model."""
    return [c for c in checks if not c.get("calibration")]


def error_pct(row: dict[str, Any]) -> float:
    """Signed prediction error as a percentage of the measurement (small measurements use a 20 ms floor)."""
    return 100.0 * (row["predicted"] - row["measured"]) / max(row["measured"], 20.0)
