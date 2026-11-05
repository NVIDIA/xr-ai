# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Design-time latency estimator: a sequence of stages priced from the fitted model.

A design is a list of stages evaluated in order. Every stage has a ``p50`` and a conservative ``p90`` (the sum of
stage p90s overstates the true p90), and a ``basis`` saying whether its numbers are measured, fitted from
measurements, or an assumption you supplied. ``first_audio`` is the time until the user hears something;
``done`` is the time until the whole turn finishes. ``xr_latency/report/calc.js`` ports this arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PATHS = {
    # path -> (decide calls, admit calls, router-spoken control texts before the answer)
    "single_candidate": (0, 0, 0),
    "focused": (0, 1, 0),
    "multi_route": (1, 1, 0),
    "misroute_declined_once": (2, 2, 0),
    "release_with_text_then_reroute": (2, 2, 1),
    "declines_exhausted_default": (2, 2, 0),
    "clarify": (1, 0, 1),
}


@dataclass
class Est:
    """One stage's cost: p50 and p90 milliseconds plus how trustworthy the number is."""

    p50: float
    p90: float
    basis: str  # measured | fitted | assumed


@dataclass
class Stage:
    """A priced stage on the critical path."""

    name: str
    start_p50: float
    end_p50: float
    start_p90: float
    end_p90: float
    basis: str
    note: str = ""


@dataclass
class Result:
    """Priced design."""

    stages: list[Stage] = field(default_factory=list)
    first_audio_p50: float | None = None
    first_audio_p90: float | None = None
    done_p50: float = 0.0
    done_p90: float = 0.0


def _curve(c: dict[str, Any] | None, x: float, stat: str) -> float:
    if not c:
        raise KeyError("model has no data for this term; run the matching benchmark first")
    fit = c[stat]
    return max(fit["intercept"] + fit["slope"] * x, 0.0)


def _est(c: dict[str, Any] | None, x: float, basis: str = "fitted") -> Est:
    return Est(_curve(c, x, "p50"), _curve(c, x, "p90"), basis)


def _lookup(table: list[dict[str, Any]], **keys: Any) -> dict[str, Any] | None:
    for row in table:
        if all(row.get(k) == v for k, v in keys.items()):
            return row
    return None


def llm_ttft(model: dict[str, Any], prompt_tokens: float, cached: bool) -> Est:
    """Time to first visible token with thinking off; ``cached`` means the prefix is already in the server cache."""
    llm = model["llm"]
    e = _est(llm["ttft_warm_vs_ctx"], prompt_tokens) if cached else _est(llm["ttft_cold_vs_ctx"], prompt_tokens)
    floor = llm.get("ttft_floor")  # a straight line through large prompts predicts ~0 ms for tiny ones
    if floor:
        return Est(max(e.p50, floor["p50"]), max(e.p90, floor["p90"]), e.basis)
    return e


def llm_decode_ms_per_token(model: dict[str, Any]) -> Est:
    """Per-token decode time from the slope of total time against output tokens."""
    c = model["llm"]["decode_total_vs_tokens_chat"]
    return Est(max(c["p50"]["slope"], 0.0), max(c["p90"]["slope"], 0.0), "fitted")


def thinking_tax(model: dict[str, Any]) -> Est:
    """Extra time to first visible token when thinking is on, from the thinking-on vs off experiment."""
    rows = model["llm"]["thinking"]["first_text"]
    on, off = _lookup(rows, thinking=True), _lookup(rows, thinking=False)
    if not on or not off:
        raise KeyError("model has no thinking experiment")
    return Est(max(on["p50"] - off["p50"], 0.0), max(on["p90"] - off["p90"], 0.0), "measured")


def reasoning_cost(model: dict[str, Any], spec: dict[str, Any]) -> Est:
    """Cost of thinking before the first visible token: explicit ``reasoning_tokens``, else the measured tax."""
    if spec.get("reasoning_tokens"):
        n, dec = spec["reasoning_tokens"], llm_decode_ms_per_token(model)
        return Est(n * dec.p50, n * dec.p90, "fitted")
    return thinking_tax(model) if spec.get("thinking") else Est(0, 0, "measured")


def decide_ms(model: dict[str, Any], history_bytes: float, labels: int = 4) -> Est:
    """One routing decision call against the history it must read (calibrated on the router's own paths if present)."""
    if model.get("router", {}).get("decide_curve"):
        return _est(model["router"]["decide_curve"], history_bytes)
    c = model["decision"].get(f"total_vs_history_bytes_labels{labels}") or model["decision"][
        "total_vs_history_bytes_labels4"]
    return _est(c, history_bytes)


def tts_first(model: dict[str, Any], chars: float, stream: bool) -> Est:
    """Time until the first audio exists for ``chars`` of text."""
    tts = model["tts"]
    if stream:
        rows = tts["stream_first_chunk"]
        p50 = sum(r["p50"] for r in rows) / len(rows)
        p90 = sum(r["p90"] for r in rows) / len(rows)
        return Est(p50, p90, "measured")
    return _est(tts["batch_total_vs_chars"], chars)


def evaluate(model: dict[str, Any], design: list[dict[str, Any]]) -> Result:
    """Price ``design`` (a list of stage dicts) and return per-stage timing plus first-audio and done times."""
    result = Result()
    t50 = t90 = 0.0
    for spec in design:
        kind = spec["kind"]
        gets_audio, tail = False, 0.0
        if kind == "fixed":
            e = Est(spec["p50"], spec.get("p90", spec["p50"]), spec.get("basis", "assumed"))
            note = spec.get("note", "")
        elif kind == "endpoint":
            e, note = _endpoint(model, spec)
        elif kind == "route":
            e, note = _route(model, spec)
        elif kind == "llm":
            e, note = _llm(model, spec)
        elif kind == "tool_hop":
            e, note = _tool_hop(model, spec)
        elif kind == "vlm":
            e, note = _vlm(model, spec)
        elif kind == "tts":
            e, note, tail = _tts_only(model, spec)
            gets_audio = True
        elif kind == "vlm_decode":
            decode = model["vlm"]["decode_ms_per_token"]
            e = Est(spec["tokens"] * decode["p50"], spec["tokens"] * decode["p90"], "fitted")
            note = f"{spec['tokens']} more VLM output tokens"
        elif kind == "vlm_speak":
            e, note, tail = _vlm_speak(model, spec)
            gets_audio = True
        elif kind == "speak":
            e, note, tail = _speak(model, spec)
            gets_audio = True
        else:
            raise ValueError(f"unknown stage kind {kind!r}")
        result.stages.append(Stage(spec.get("name", kind), t50, t50 + e.p50, t90, t90 + e.p90, e.basis, note))
        t50 += e.p50
        t90 += e.p90
        if gets_audio and result.first_audio_p50 is None:
            result.first_audio_p50, result.first_audio_p90 = t50, t90
        t50 += tail
        t90 += tail
    result.done_p50, result.done_p90 = t50, t90
    return result


def _endpoint(model: dict[str, Any], spec: dict[str, Any]) -> tuple[Est, str]:
    """End of user speech until a final transcript, from the speaker-ASR measurements."""
    rows = model["stt"]["end_to_final"]
    pick = [r for r in rows if r["mode"] == spec.get("mode", "conversation") and r["utterance"] == spec.get(
        "utterance", "medium") and r["frame_ms"] == spec.get("frame_ms", 10)]
    if not pick:
        raise KeyError("model has no matching speaker-ASR measurement")
    return Est(pick[0]["p50"], pick[0]["p90"], "measured"), "speaker ASR end of speech to final"


def _route(model: dict[str, Any], spec: dict[str, Any]) -> tuple[Est, str]:
    decides, admits, controls = PATHS[spec.get("path", "multi_route")]
    d = decide_ms(model, spec.get("history_bytes", 3_000), spec.get("labels", 4))
    admit = spec.get("admit_ms", 0.0)
    ctrl = spec.get("control_speech_ms", 0.0) * controls
    return Est(decides * d.p50 + admits * admit + ctrl, decides * d.p90 + admits * admit + ctrl, "fitted"), (
        f"{decides} decide + {admits} admit + {controls} control speech")


def _llm(model: dict[str, Any], spec: dict[str, Any]) -> tuple[Est, str]:
    """Whole LLM call time (``done``). Streaming only matters to ``speak``, which needs the first-unit time."""
    ttft = llm_ttft(model, spec["prompt_tokens"] + model["llm"]["chat_overhead_tokens"], spec.get("cached", False))
    decode = llm_decode_ms_per_token(model)
    tax = reasoning_cost(model, spec)
    n = spec.get("out_tokens", 60)
    return Est(ttft.p50 + tax.p50 + n * decode.p50, ttft.p90 + tax.p90 + n * decode.p90, "fitted"), (
        f"ttft + {n} tokens decode" + (" + thinking" if spec.get("thinking") else ""))


def _tool_hop(model: dict[str, Any], spec: dict[str, Any]) -> tuple[Est, str]:
    """``hops`` serial non-streaming LLM calls each followed by a tool of ``tool_ms``; context grows each hop."""
    p50 = p90 = 0.0
    prompt = spec["prompt_tokens"]
    for _ in range(spec.get("hops", 1)):
        e, _ = _llm(model, {**spec, "prompt_tokens": prompt, "out_tokens": spec.get("hop_out_tokens", 40)})
        p50 += e.p50 + spec.get("tool_ms", 0.0)
        p90 += e.p90 + spec.get("tool_ms", 0.0)
        prompt += spec.get("hop_growth_tokens", 300)
    return Est(p50, p90, "fitted"), f"{spec.get('hops', 1)} serial tool hops (tool_ms is your assumption)"


def _vlm(model: dict[str, Any], spec: dict[str, Any]) -> tuple[Est, str]:
    """A VLM call by its measured (resolution, image count) cell; ``cached`` uses the repeat-query measurement."""
    vlm = model["vlm"]
    table = vlm["cached_total"] if spec.get("cached") else vlm["chat_total"]
    row = _lookup(table, res=spec["res"], images=spec["images"])
    if row is None:
        raise KeyError("no measured VLM cell for that resolution and image count")
    return Est(row["p50"], row["p90"], "measured"), f"{spec['images']} image(s) at {spec['res']}"


def _chars_per_token(model: dict[str, Any]) -> float:
    return model.get("streaming", {}).get("chars_per_token") or 4.0


def _audio_ms_per_char(model: dict[str, Any]) -> float:
    rows = model["tts"]["audio_seconds"]
    return sum(r["p50"] * 1000.0 / r["chars"] for r in rows) / len(rows)


def _speak(model: dict[str, Any], spec: dict[str, Any]) -> tuple[Est, str, float]:
    """From LLM start to first audio, under a delivery strategy (the LLM stage should not be listed separately).

    Also returns the remaining time until the whole answer has finished playing (speech is slower than generation).

    ``batch``: whole text then batch TTS. ``stream_full``: whole text then streamed TTS. ``sentence``: streamed text
    chunked at the first sentence, then streamed TTS for it. Adds ``playback_ms`` (transport + jitter buffer).
    """
    cached = spec.get("cached", False)
    ttft = llm_ttft(model, spec["prompt_tokens"] + model["llm"]["chat_overhead_tokens"], cached)
    decode = llm_decode_ms_per_token(model)
    tax = reasoning_cost(model, spec)
    total_tokens, first_tokens = spec.get("out_tokens", 60), spec.get("first_sentence_tokens", 20)
    chars_per_token = _chars_per_token(model)
    strategy = spec.get("strategy", "sentence")
    if strategy == "sentence":
        gen = Est(ttft.p50 + tax.p50 + first_tokens * decode.p50,
                  ttft.p90 + tax.p90 + first_tokens * decode.p90, "fitted")
        speak = tts_first(model, first_tokens * chars_per_token, True)
    else:
        gen = Est(ttft.p50 + tax.p50 + total_tokens * decode.p50,
                  ttft.p90 + tax.p90 + total_tokens * decode.p90, "fitted")
        speak = tts_first(model, total_tokens * chars_per_token, strategy == "stream_full")
    play = spec.get("playback_ms", 0.0)
    tail = total_tokens * chars_per_token * _audio_ms_per_char(model)
    return Est(gen.p50 + speak.p50 + play, gen.p90 + speak.p90 + play, "fitted"), f"{strategy} delivery", tail


def _vlm_speak(model: dict[str, Any], spec: dict[str, Any]) -> tuple[Est, str, float]:
    """A streamed VLM answer into sentence-chunked streamed TTS (the simple-vlm-example path).

    Time to first audio = VLM first text + decode of the first sentence + TTS first chunk + ``playback_ms``.
    Python-only: the page calculator does not offer this stage.
    """
    vlm = model["vlm"]
    row = _lookup(vlm["stream_first_text"], res=spec["res"], images=spec["images"])
    if row is None:
        raise KeyError("no measured VLM streaming cell for that resolution and image count")
    decode = vlm["decode_ms_per_token"]
    first_tokens, total_tokens = spec.get("first_sentence_tokens", 20), spec.get("out_tokens", 30)
    first = tts_first(model, first_tokens * _chars_per_token(model), True)
    play = spec.get("playback_ms", 0.0)
    e = Est(row["p50"] + first_tokens * decode["p50"] + first.p50 + play,
            row["p90"] + first_tokens * decode["p90"] + first.p90 + play, "fitted")
    tail = total_tokens * _chars_per_token(model) * _audio_ms_per_char(model)
    return e, f"{spec['images']} image(s) at {spec['res']}, streamed", tail


def _tts_only(model: dict[str, Any], spec: dict[str, Any]) -> tuple[Est, str, float]:
    """Speak text that already exists (an alert, a templated reply): first audio plus the time to play it all."""
    chars = spec["chars"]
    first = tts_first(model, chars, spec.get("stream", True))
    play = spec.get("playback_ms", 0.0)
    tail = chars * _audio_ms_per_char(model)
    return Est(first.p50 + play, first.p90 + play, first.basis), "speak existing text", tail
