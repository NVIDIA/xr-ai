# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``xr-latency`` command line."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ._gpu import configure_metrics, gpu_utilization, wait_until_quiet
from ._machine import fingerprint
from ._stack import Stack
from ._store import RunStore
from .suites import BENCHES, SUITES, Bench, Opts, Step, missing

RESULTS = Path(__file__).resolve().parents[1] / "results"
DEFAULT_STACK = "pr630-live"


def _service(stack: Stack, role: str):  # noqa: ANN202 - one of four typed services, chosen by role name
    """The typed service for ``role`` (``llm_raw`` is an ``llm``), or None for roles with no health check."""
    factories = {"llm": stack.llm, "vlm": stack.vlm, "router": stack.decision, "tts": stack.tts}
    factory = factories.get(role) or factories.get(role.split("_")[0])
    return factory(role) if factory else None


async def _probe(stack: Stack) -> int:
    """Health-check every role without generating anything."""
    ok = True
    for role in stack.roles():
        if (service := _service(stack, role)) is None:
            print(f"{role}: skipped")
            continue
        healthy = await service.health()
        ok &= healthy
        print(f"{role}: {'ready' if healthy else 'NOT READY'}")
    print(f"gpu utilization %: {gpu_utilization()}")
    return 0 if ok else 1


async def _down(bench: Bench, stack: Stack) -> list[str]:
    """Roles ``bench`` needs whose servers do not answer their health check."""
    down = []
    for role in bench.needs:
        service = _service(stack, role) if role in stack.raw["models"] else None
        if service is not None and not await service.health():
            down.append(role)
    return down


async def _run(args: argparse.Namespace) -> int:
    stack = Stack.load(args.stack, args.host)
    configure_metrics(stack.endpoints().values())
    if args.target in SUITES:
        steps = list(SUITES[args.target])
    else:
        steps = [Step(args.target, Opts(args.samples, frozenset(args.skip or ()), frozenset(args.only or ()),
                                        "short_set" in (args.only or ())))]
    if not args.no_quiet_wait and not await wait_until_quiet():
        print("GPUs stayed busy; pass --no-quiet-wait to run anyway")
        return 2
    results = Path(args.results_dir)
    machine = fingerprint(stack.endpoints())
    failed = 0
    for step in steps:
        bench = BENCHES[step.bench]
        if absent := missing(bench, stack):
            print(f"skip {bench.name}: the stack does not declare {', '.join(absent)}")
            continue
        if down := await _down(bench, stack):
            print(f"skip {bench.name}: not ready: {', '.join(down)}")
            continue
        opts = Opts(args.samples, step.opts.skip, step.opts.only, step.opts.stt_only_short) if (
            args.samples is not None and args.target in SUITES) else step.opts
        store = RunStore.create(results, f"{args.label}-{bench.name}" if args.label else bench.name)
        store.write_meta({"stack": args.stack, "bench": bench.name, "machine": machine, "gpu_util": gpu_utilization()})
        print(f"== {bench.name}: {bench.summary}")
        try:
            await bench.run(stack, store, opts)
        except Exception as exc:  # noqa: BLE001 - one failed benchmark must not discard the rest of a suite
            failed += 1
            print(f"FAILED {bench.name}: {type(exc).__name__}: {exc}")
        finally:
            store.close()
        print(f"   results: {store.root}")
    return 1 if failed else 0


def _model(path: Path) -> dict:
    return json.loads(path.read_text())


def _estimate(args: argparse.Namespace) -> int:
    import yaml

    from .estimate import evaluate
    from .patterns import load_patterns

    if args.design.exists():
        stages = yaml.safe_load(args.design.read_text())["stages"]
    else:
        stages = load_patterns()[args.design.stem].stages
    result = evaluate(_model(args.model), stages)
    for s in result.stages:
        print(f"{s.name:36s} +{s.end_p50 - s.start_p50:8.0f} ms p50  +{s.end_p90 - s.start_p90:8.0f} ms p90  "
              f"[{s.basis}] {s.note}")
    if result.first_audio_p50 is not None:
        print(f"time to first audio: p50 {result.first_audio_p50:.0f} ms, p90 {result.first_audio_p90:.0f} ms")
    print(f"turn done:           p50 {result.done_p50:.0f} ms, p90 {result.done_p90:.0f} ms")
    return 0


def _patterns(args: argparse.Namespace) -> int:
    from .patterns import load_patterns, price_all

    model = _model(args.model)
    patterns = load_patterns()
    print(f"{'pattern':22s}{'first audio p50':>17s}{'p90':>8s}{'playback end ms':>16s}  title")
    for name, result in price_all(model, thinking=True if args.thinking else None).items():
        if isinstance(result, str):
            print(f"{name:22s}  {result}")
            continue
        first = f"{result.first_audio_p50:.0f} ms" if result.first_audio_p50 is not None else "n/a"
        p90 = f"{result.first_audio_p90:.0f}" if result.first_audio_p90 is not None else "-"
        print(f"{name:22s}{first:>17s}{p90:>8s}{result.done_p50:>16.0f}  {patterns[name].title}")
    return 0


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(prog="xr-latency")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_stack(p: argparse.ArgumentParser) -> None:
        p.add_argument("--stack", default=DEFAULT_STACK, help="profile name in stacks/ or a path")
        p.add_argument("--host", help="replace localhost in the profile, to profile a stack served by another machine")

    probe = sub.add_parser("probe", help="health-check the services in a stack; generates nothing")
    add_stack(probe)
    run = sub.add_parser("run", help="run one benchmark or a suite (quick, full)")
    run.add_argument("target", choices=[*BENCHES, *SUITES])
    add_stack(run)
    run.add_argument("--samples", type=int, help="override the sample count (default: each benchmark's own)")
    run.add_argument("--skip", nargs="*", help="skip sweeps inside a benchmark (llm, streaming)")
    run.add_argument("--only", nargs="*", help="router: restrict to these path names (and 'ack'); stt: 'short_set'")
    run.add_argument("--label")
    run.add_argument("--results-dir", default=str(RESULTS))
    run.add_argument("--no-quiet-wait", action="store_true")
    report = sub.add_parser("report", help="fit model.json from runs and render report.html")
    report.add_argument("runs", nargs="+", type=Path)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--findings", type=Path, help="HTML fragment with a findings section")
    est = sub.add_parser("estimate", help="price a stage list (YAML file) or a pattern name from a fitted model.json")
    est.add_argument("design", type=Path, help="a design YAML, or the name of a file in patterns/")
    est.add_argument("--model", type=Path, required=True)
    pat = sub.add_parser("patterns", help="price every pattern from a fitted model.json")
    pat.add_argument("--model", type=Path, required=True)
    pat.add_argument("--thinking", action="store_true", help="price with LLM thinking on")
    cmp = sub.add_parser("compare", help="compare two fitted models, e.g. one stack on two machines")
    cmp.add_argument("a", type=Path)
    cmp.add_argument("b", type=Path)
    cmp.add_argument("--thinking", action="store_true", help="price patterns with LLM thinking on")
    args = parser.parse_args()
    if args.cmd == "estimate":
        raise SystemExit(_estimate(args))
    if args.cmd == "patterns":
        raise SystemExit(_patterns(args))
    if args.cmd == "compare":
        from .compare import compare

        print(compare(_model(args.a), _model(args.b), thinking=True if args.thinking else None))
        raise SystemExit(0)
    if args.cmd == "report":
        from .report.build import make_report

        print(make_report(args.runs, args.out, args.findings))
        raise SystemExit(0)
    if args.cmd == "probe":
        raise SystemExit(asyncio.run(_probe(Stack.load(args.stack, args.host))))
    raise SystemExit(asyncio.run(_run(args)))
