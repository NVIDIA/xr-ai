# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The benchmark registry and the suites that run several benchmarks in order.

Adding a benchmark is one module in ``bench/``, one ``Bench`` entry here, and one block in ``fit.py``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from ._stack import Stack
from ._store import RunStore

P = TypeVar("P")


@dataclass(frozen=True)
class Opts:
    """Per-run overrides. ``samples=None`` keeps each benchmark's own default sample count."""

    samples: int | None = None
    skip: frozenset[str] = frozenset()
    only: frozenset[str] = frozenset()
    stt_only_short: bool = False


def _plan(cls: type[P], opts: Opts, **extra: object) -> P:
    """Build ``cls(...)`` with ``samples`` only when the caller gave one."""
    return cls(**({"samples": opts.samples} if opts.samples is not None else {}), **extra)  # type: ignore[call-arg]


async def _llm(stack: Stack, store: RunStore, opts: Opts) -> None:
    from .bench import llm

    await llm.run(stack.llm("llm_raw"), store, _plan(llm.LlmPlan, opts, skip=set(opts.skip)), live=stack.llm())


async def _decision(stack: Stack, store: RunStore, opts: Opts) -> None:
    from .bench import decision

    await decision.run(stack.decision(), store, _plan(decision.DecisionPlan, opts))


async def _tts(stack: Stack, store: RunStore, opts: Opts) -> None:
    from .bench import tts

    await tts.run(stack.tts(), store, _plan(tts.TtsPlan, opts))


async def _vlm(stack: Stack, store: RunStore, opts: Opts) -> None:
    from .bench import vlm

    await vlm.run(stack.vlm(), store, _plan(vlm.VlmPlan, opts))


async def _streaming(stack: Stack, store: RunStore, opts: Opts) -> None:
    from .bench import streaming

    await streaming.run(stack.llm("llm_raw"), stack.tts(), store, _plan(streaming.StreamingPlan, opts, skip=opts.skip))


async def _router(stack: Stack, store: RunStore, opts: Opts) -> None:
    from .bench import router_paths

    await router_paths.run(stack.decision(), stack.llm(), store, _plan(router_paths.RouterPlan, opts, only=opts.only))


async def _contention(stack: Stack, store: RunStore, opts: Opts) -> None:
    from .bench import contention

    await contention.run(stack.llm("llm_raw"), store, _plan(contention.ContentionPlan, opts))


async def _stt(stack: Stack, store: RunStore, opts: Opts) -> None:
    from .bench import stt

    extra = {"base_url": url} if (url := stack.service_url("speaker_stt")) else {}
    await stt.run(stack.tts(), store, _plan(stt.SttPlan, opts, short_set_only=opts.stt_only_short, **extra))


async def _vad(stack: Stack, store: RunStore, opts: Opts) -> None:
    from .bench import vad

    await vad.run(stack.tts(), store, _plan(vad.VadPlan, opts))


@dataclass(frozen=True)
class Bench:
    """A runnable benchmark: what it measures and which profile entries it needs."""

    name: str
    summary: str
    needs: tuple[str, ...]  # profile roles, or ``services.<name>`` entries
    run: Callable[[Stack, RunStore, Opts], Awaitable[None]]


BENCHES: dict[str, Bench] = {b.name: b for b in (
    Bench("llm", "time to first token vs prompt size, cache, decode rate, thinking, concurrency",
          ("llm", "llm_raw"), _llm),
    Bench("decision", "router decision call vs history size and label count", ("router",), _decision),
    Bench("router", "every AgentRouter path with synthetic destinations, plus the acknowledgment race",
          ("router", "llm"), _router),
    Bench("streaming", "chat vs stream vs stream_events, tool turns, time to first audio per delivery strategy",
          ("llm_raw", "tts"), _streaming),
    Bench("vlm", "VLM latency vs image count and resolution, streamed and cached", ("vlm",), _vlm),
    Bench("tts", "batch and streamed speech vs text length", ("tts",), _tts),
    Bench("stt", "speaker-ASR end of speech to final at real-time pace", ("tts", "services.speaker_stt"), _stt),
    Bench("vad", "Silero VAD firing time vs frame size (CPU only)", ("tts",), _vad),
    Bench("contention", "LLM latency while other load runs on the same server", ("llm_raw",), _contention),
)}


@dataclass(frozen=True)
class Step:
    """One benchmark inside a suite, with its overrides."""

    bench: str
    opts: Opts = field(default_factory=Opts)


# ``quick`` feeds every estimator term and every pattern with small sample counts, for comparing machines fast.
# ``full`` is each benchmark's own default plan, including the router acknowledgment race, VAD and contention.
SUITES: dict[str, tuple[Step, ...]] = {
    "quick": (
        Step("llm", Opts(samples=5, skip=frozenset({"concurrency"}))),
        Step("decision", Opts(samples=8)),
        Step("tts", Opts(samples=6)),
        Step("vlm", Opts(samples=4)),
        Step("streaming", Opts(samples=4, skip=frozenset({"tools", "short"}))),
        Step("router", Opts(samples=5)),
        Step("stt", Opts(samples=2)),
    ),
    "full": tuple(Step(name) for name in ("decision", "tts", "vlm", "llm", "streaming", "router", "stt", "vad",
                                          "contention")),
}


def missing(bench: Bench, stack: Stack) -> list[str]:
    """Profile entries ``bench`` needs that ``stack`` does not declare."""
    absent = []
    for need in bench.needs:
        if need.startswith("services."):
            if not stack.service_url(need.removeprefix("services.")):
                absent.append(need)
        elif need not in stack.raw["models"]:
            absent.append(need)
    return absent
