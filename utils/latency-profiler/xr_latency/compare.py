# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two fitted models, typically the same stack measured on two machines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import estimate as est
from ._machine import label
from .patterns import price_all


def _terms(model: dict[str, Any]) -> dict[str, Callable[[], est.Est]]:
    """Headline component costs, each a callable that raises KeyError when the model lacks the measurement."""
    return {
        "LLM first token, cold 4k prompt": lambda: est.llm_ttft(model, 4_000, cached=False),
        "LLM first token, cached 4k prompt": lambda: est.llm_ttft(model, 4_000, cached=True),
        "LLM decode, ms per token": lambda: est.llm_decode_ms_per_token(model),
        "Thinking tax on first visible token": lambda: est.thinking_tax(model),
        "Router decision, 6 KB history": lambda: est.decide_ms(model, 6_000),
        "VLM answer, 1 image 1280x720": lambda: est._vlm(model, {"res": "1280x720", "images": 1})[0],
        "TTS first audio, streamed": lambda: est.tts_first(model, 80, True),
        "Speaker ASR, end of speech to final": lambda: est._endpoint(model, {})[0],
    }


def _value(thunk: Callable[[], est.Est]) -> float | None:
    try:
        return thunk().p50
    except (KeyError, IndexError, ZeroDivisionError):
        return None


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def _ratio(a: float | None, b: float | None) -> str:
    return "" if not a or not b else f"{b / a:.2f}x"


def compare(a: dict[str, Any], b: dict[str, Any], *, thinking: bool | None = None) -> str:
    """A text table of component costs and pattern first-audio times for ``a`` against ``b`` (p50, ms)."""
    lines = [f"A = {label(a.get('machine'))}", f"B = {label(b.get('machine'))}", "",
             f"{'component (p50 ms)':44s}{'A':>10s}{'B':>10s}{'B/A':>8s}"]
    ta, tb = _terms(a), _terms(b)
    for name in ta:
        va, vb = _value(ta[name]), _value(tb[name])
        lines.append(f"{name:44s}{_fmt(va):>10s}{_fmt(vb):>10s}{_ratio(va, vb):>8s}")
    pa, pb = price_all(a, thinking=thinking), price_all(b, thinking=thinking)
    lines += ["", f"{'pattern, time to first audio (p50 ms)':44s}{'A':>10s}{'B':>10s}{'B/A':>8s}"]
    for name in pa:
        va = getattr(pa[name], "first_audio_p50", None)
        vb = getattr(pb.get(name), "first_audio_p50", None)
        lines.append(f"{name:44s}{_fmt(va):>10s}{_fmt(vb):>10s}{_ratio(va, vb):>8s}")
    return "\n".join(lines)
