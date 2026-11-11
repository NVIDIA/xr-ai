# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit the latency model from raw samples: curves for p50 and p90 plus the measured lookup tables.

The output (``model.json``) is the only input to the estimator and the report, so every number a design
estimate uses can be traced back to a measured group here.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ._stats import Summary, fit_linear
from ._store import read_meta
from .analyze import group, load

# Chat-template and system overhead the server adds on top of the prompt text (measured: 182 - 128 at 128 tokens).
CHAT_OVERHEAD_TOKENS = 54


def _line(groups: dict[tuple[Any, ...], Summary], stat: str) -> dict[str, float] | None:
    points = sorted((float(k[0]), getattr(s, stat)) for k, s in groups.items() if k[0] is not None)
    if len({x for x, _ in points}) < 2:
        return None
    fit = fit_linear([x for x, _ in points], [y for _, y in points])
    return {"intercept": fit.intercept, "slope": fit.slope, "r2": fit.r2, "n_points": fit.n}


def _curve(groups: dict[tuple[Any, ...], Summary]) -> dict[str, Any] | None:
    """Linear fits of p50 and p90 against the single numeric key, plus the raw points."""
    p50, p90 = _line(groups, "p50"), _line(groups, "p90")
    if p50 is None or p90 is None:
        return None
    return {"p50": p50, "p90": p90,
            "points": [{"x": float(k[0]), "p50": s.p50, "p90": s.p90, "n": s.n} for k, s in groups.items()]}


def _table(groups: dict[tuple[Any, ...], Summary], names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{**dict(zip(names, k, strict=True)), "n": s.n, "p50": s.p50, "p90": s.p90, "min": s.min, "max": s.max,
             "mean": s.mean} for k, s in groups.items()]


def fit_all(run_dirs: Iterable[Path]) -> dict[str, Any]:
    """Build the model from every sample in ``run_dirs`` (samples taken under foreign load are dropped)."""
    run_dirs = list(run_dirs)
    rows = load(run_dirs)
    model: dict[str, Any] = {"sample_count": len(rows), "machine": _machine(run_dirs)}

    llm: dict[str, Any] = {"chat_overhead_tokens": CHAT_OVERHEAD_TOKENS}
    llm["ttft_cold_vs_ctx"] = _curve(group(rows, "llm.context.cold", "first_text", ["ctx_tokens"], {"mode": "stream"}))
    llm["ttft_warm_vs_ctx"] = _curve(group(rows, "llm.context.prefix", "first_text", ["ctx_tokens"], {"warm": True}))
    cold = group(rows, "llm.context.cold", "first_text", ["ctx_tokens"], {"mode": "stream"})
    llm["ttft_floor"] = (
        {"p50": min(v.p50 for v in cold.values()), "p90": min(v.p90 for v in cold.values())} if cold else None)
    llm["ttft_cold_table"] = _table(
        group(rows, "llm.context.cold", "first_text", ["ctx_tokens"], {"mode": "stream"}), ("ctx_tokens",))
    llm["ttft_warm_table"] = _table(
        group(rows, "llm.context.prefix", "first_text", ["ctx_tokens"], {"warm": True}), ("ctx_tokens",))
    llm["decode_total_vs_tokens_chat"] = _curve(
        group(rows, "llm.output", "total", ["max_tokens"], {"mode": "chat"}))
    llm["decode_total_vs_tokens_stream"] = _curve(
        group(rows, "llm.output", "total", ["max_tokens"], {"mode": "stream"}))
    llm["stream_first_text_vs_tokens"] = _table(
        group(rows, "llm.output", "first_text", ["max_tokens"], {"mode": "stream"}), ("max_tokens",))
    llm["thinking"] = {
        "first_text": _table(group(rows, "llm.thinking", "first_text", ["thinking"]), ("thinking",)),
        "first_reasoning": _table(group(rows, "llm.thinking", "first_reasoning", ["thinking"]), ("thinking",)),
        "reasoning_chunks": _table(group(rows, "llm.thinking", "reasoning_chunks", ["thinking"]), ("thinking",)),
        "total": _table(group(rows, "llm.thinking", "total", ["thinking"]), ("thinking",)),
    }
    llm["concurrency_ttft"] = _table(group(rows, "llm.concurrency", "first_text", ["width"]), ("width",))
    llm["concurrency_total"] = _table(group(rows, "llm.concurrency", "total", ["width"]), ("width",))
    llm["live_defaults"] = {
        m: _table(group(rows, "llm.live_defaults", m, ["call"]), ("call",))
        for m in ("first_text", "first_reasoning", "total", "completion_tokens")
    }
    llm["contention"] = {
        m: _table(group(rows, "contention", m, ["load"]), ("load",)) for m in ("first_text", "total")}
    model["llm"] = llm

    decision: dict[str, Any] = {}
    for labels in (2, 4, 8):
        decision[f"total_vs_history_bytes_labels{labels}"] = _curve(
            group(rows, "decision", "total", ["history_bytes"], {"labels": labels}))
    decision["table"] = _table(group(rows, "decision", "total", ["labels", "history_bytes"]),
                               ("labels", "history_bytes"))
    decision["concurrency"] = _table(group(rows, "decision.concurrency", "total", ["width", "rank"]),
                                     ("width", "rank"))
    decision["confidence"] = _table(group(rows, "decision", "confidence", ["history_bytes"]), ("history_bytes",))
    hits = [r for r in rows if r["bench"] == "decision"]
    decision["accuracy_on_synthetic"] = (
        sum(r["choice"] == r["expected"] for r in hits) / len(hits) if hits else None
    )
    model["decision"] = decision

    model["tts"] = {
        "batch_total_vs_chars": _curve(group(rows, "tts.batch", "total", ["chars"], {"warm": True})),
        "stream_total_vs_chars": _curve(group(rows, "tts.stream", "total", ["chars"], {"warm": True})),
        "stream_first_chunk": _table(group(rows, "tts.stream", "first_chunk", ["chars"], {"warm": True}), ("chars",)),
        "batch_table": _table(group(rows, "tts.batch", "total", ["chars"], {"warm": True}), ("chars",)),
        "audio_seconds": _table(group(rows, "tts.batch", "audio_s", ["chars"], {"warm": True}), ("chars",)),
        "chunks": _table(group(rows, "tts.stream", "chunks", ["chars"], {"warm": True}), ("chars",)),
    }

    vlm_prompt = group(rows, "vlm.chat", "prompt_tokens", ["res", "images"])
    model["vlm"] = {
        "chat_total": _table(group(rows, "vlm.chat", "total", ["res", "images"]), ("res", "images")),
        "stream_first_text": _table(group(rows, "vlm.stream", "first_text", ["res", "images"]), ("res", "images")),
        "cached_total": _table(group(rows, "vlm.cached", "total", ["res", "images"]), ("res", "images")),
        "prompt_tokens": _table(vlm_prompt, ("res", "images")),
        "first_text_vs_prompt_tokens": _vlm_ttft_curve(rows),
    }

    chat_total = group(rows, "vlm.chat", "total", ["res", "images"])
    first_text = group(rows, "vlm.stream", "first_text", ["res", "images"])
    out_tokens = group(rows, "vlm.chat", "completion_tokens", ["res", "images"])
    per_token = [((chat_total[k].p50 - first_text[k].p50) / out_tokens[k].p50,
                  (chat_total[k].p90 - first_text[k].p90) / out_tokens[k].p50)
                 for k in chat_total if k in first_text and k in out_tokens and out_tokens[k].p50 >= 5]
    if per_token:
        model["vlm"]["decode_ms_per_token"] = {
            "p50": max(sum(a for a, _ in per_token) / len(per_token), 0.0),
            "p90": max(sum(b for _, b in per_token) / len(per_token), 0.0), "cells": len(per_token)}

    stream: dict[str, Any] = {}
    for metric in ("first_text", "total"):
        stream[f"modes_{metric}"] = _table(group(rows, "stream.modes", metric, ["sentences", "mode"]),
                                           ("sentences", "mode"))
    ratios = sorted(r["text_chars"] / r["completion_tokens"] for r in rows
                    if r["bench"] == "stream.modes" and r["case"].get("mode") == "chat"
                    and r.get("completion_tokens") and r.get("text_chars"))
    stream["chars_per_token"] = ratios[len(ratios) // 2] if ratios else None
    stream["completion_tokens"] = _table(
        group(rows, "stream.modes", "completion_tokens", ["sentences"], {"mode": "chat"}), ("sentences",))
    stream["tools_total"] = _table(group(rows, "stream.tools", "total", ["mode"]), ("mode",))
    stream["short_total"] = _table(group(rows, "stream.short", "total", ["max_tokens", "mode"]), ("max_tokens", "mode"))
    stream["short_first"] = _table(group(rows, "stream.short", "first_text", ["max_tokens", "mode"]),
                                   ("max_tokens", "mode"))
    stream["first_audio"] = _table(group(rows, "stream.audio", "first_audio", ["sentences", "strategy"]),
                                   ("sentences", "strategy"))
    stream["stall"] = _table(group(rows, "stream.audio", "total_stall", ["sentences", "strategy"]),
                             ("sentences", "strategy"))
    stream["llm_done"] = _table(group(rows, "stream.audio", "llm_done", ["sentences", "strategy"]),
                                ("sentences", "strategy"))
    model["streaming"] = stream

    router: dict[str, Any] = {}
    for metric in ("total", "answer_started", "first_output", "decide_ms", "admit_ms"):
        router[metric] = _table(group(rows, "router.path", metric, ["path"]), ("path",))
    router["ack_first_output"] = _table(group(rows, "router.ack", "first_output", ["emit_ms", "answer_delay_ms"]),
                                        ("emit_ms", "answer_delay_ms"))
    router["ack_answer_started"] = _table(group(rows, "router.ack", "answer_started", ["emit_ms", "answer_delay_ms"]),
                                          ("emit_ms", "answer_delay_ms"))
    router["ack_chat_ms"] = _table(group(rows, "router.ack", "ack_chat_ms", ["emit_ms", "answer_delay_ms"]),
                                   ("emit_ms", "answer_delay_ms"))
    small, large = _row_of(router["total"], "multi_route_accept"), _row_of(router["total"], "multi_route_accept_ctx12k")
    if small and large:
        router["decide_curve"] = {
            stat: {"intercept": small[stat], "slope": (large[stat] - small[stat]) / 12_000, "r2": None, "n_points": 2}
            for stat in ("p50", "p90")}
        router["decide_curve"]["points"] = [{"x": 0.0, "p50": small["p50"], "p90": small["p90"], "n": small["n"]},
                                            {"x": 12_000.0, "p50": large["p50"], "p90": large["p90"], "n": large["n"]}]
    model["router"] = router

    stt: dict[str, Any] = {}
    for metric in ("end_to_final", "end_to_speech_stop", "speech_start", "first_partial", "rtt_mean", "rtt_p99",
                   "late_frames", "enroll_ms"):
        stt[metric] = _table(group(rows, "stt.speaker", metric, ["mode", "frame_ms", "utterance"]),
                             ("mode", "frame_ms", "utterance"))
    stt_rows = [r for r in rows if r["bench"] == "stt.speaker"]
    stt["empty_transcript_rate"] = _table(
        {k: Summary(len(v), min(v), 0, 0, 0, max(v), sum(v) / len(v), 0.0) for k, v in _empty(stt_rows).items()},
        ("mode", "utterance"))
    model["stt"] = stt

    vad: dict[str, Any] = {}
    for metric in ("start_lag", "end_to_final", "first_final_vs_speech_end", "n_finals", "utt_s", "cpu_ms_per_frame"):
        vad[metric] = _table(group(rows, "vad", metric, ["utterance", "silence_duration", "frame_ms"]),
                             ("utterance", "silence_duration", "frame_ms"))
    fired = group(rows, "vad", "fired", ["utterance", "silence_duration", "frame_ms"])
    vad["fired_rate"] = _table({k: v for k, v in fired.items()}, ("utterance", "silence_duration", "frame_ms"))
    model["vad"] = vad
    return model


def _machine(run_dirs: list[Path]) -> dict[str, Any] | None:
    """The machine fingerprint recorded by the first run that has one."""
    return next((m for run in run_dirs if (m := read_meta(run).get("machine"))), None)


def _row_of(table: list[dict[str, Any]], path: str) -> dict[str, Any] | None:
    return next((r for r in table if r.get("path") == path), None)


def _empty(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[float]]:
    out: dict[tuple[Any, ...], list[float]] = {}
    for r in rows:
        out.setdefault((r["case"].get("mode"), r["case"].get("utterance")), []).append(
            0.0 if r.get("transcript") else 1.0)
    return out


def _vlm_ttft_curve(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Streamed first-text time against prompt tokens, joining each (res, images) cell to its measured token count."""
    tokens = {k: s.p50 for k, s in group(rows, "vlm.chat", "prompt_tokens", ["res", "images"]).items()}
    cells = group(rows, "vlm.stream", "first_text", ["res", "images"])
    joined = {(tokens[k],): s for k, s in cells.items() if k in tokens}
    return _curve(joined)
