# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeatable GPU-memory measurement and reservation-profile generation.

Run through an environment that contains ``xr-ai-launcher``::

    python -m xr_ai_launcher.vram_measure measure --label omni \
      --output omni.measurement.json --gpu 1 -- <service command>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ._gpu import GPUDevice, query_gpu_inventory


def _output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL,
        ).strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _measurement_signature(command: list[str]) -> dict:
    config_hashes: dict[str, str] = {}
    for value in command:
        path = Path(value)
        if path.suffix.lower() not in {".json", ".yaml", ".yml"} or not path.is_file():
            continue
        try:
            config_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    return {
        "git_commit": _output(["git", "rev-parse", "HEAD"]),
        "driver_version": _output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
        ]),
        "python": sys.version.split()[0],
        "command": command,
        "config_sha256": config_hashes,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _sample(start: float, devices: tuple[GPUDevice, ...]) -> dict:
    return {
        "elapsed_s": round(time.monotonic() - start, 3),
        "gpus": {
            str(gpu.index): {
                "used_gib": round(gpu.used_memory_gib, 4),
                "free_gib": round(gpu.free_memory_gib, 4),
            }
            for gpu in devices
        },
    }


def _summarize(
    baseline: tuple[GPUDevice, ...], samples: list[dict], stable_window_s: float,
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    final_elapsed = samples[-1]["elapsed_s"] if samples else 0.0
    stable_start = max(0.0, final_elapsed - stable_window_s)
    for gpu in baseline:
        values = [sample["gpus"][str(gpu.index)]["used_gib"] for sample in samples]
        stable = [
            sample["gpus"][str(gpu.index)]["used_gib"]
            for sample in samples if sample["elapsed_s"] >= stable_start
        ]
        peak = max(values, default=gpu.used_memory_gib)
        peak_delta = max(0.0, peak - gpu.used_memory_gib)
        recommended = math.ceil((peak_delta * 1.10 + 1.0) * 10) / 10
        result[str(gpu.index)] = {
            "name": gpu.name,
            "uuid": gpu.uuid,
            "total_gib": round(gpu.total_memory_gib, 4),
            "baseline_used_gib": round(gpu.used_memory_gib, 4),
            "observed_peak_used_gib": round(peak, 4),
            "observed_peak_delta_gib": round(peak_delta, 4),
            "stable_median_used_gib": round(statistics.median(stable), 4),
            "stable_p95_used_gib": round(_percentile(stable, 0.95), 4),
            "recommended_reservation_gib": recommended,
        }
    return result


def measure(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("measure requires a command after --")

    baseline = query_gpu_inventory()
    indexes = {gpu.index for gpu in baseline}
    if args.gpu is not None and args.gpu not in indexes:
        raise SystemExit(f"GPU {args.gpu} is not present")

    print(f"Measuring VRAM every {args.interval:.2f}s: {' '.join(command)}", flush=True)
    process = subprocess.Popen(command)
    start = time.monotonic()
    samples: list[dict] = []
    interrupted = False
    try:
        while process.poll() is None:
            samples.append(_sample(start, query_gpu_inventory()))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        interrupted = True
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=20)

    if not interrupted:
        deadline = time.monotonic() + args.stable_seconds
        print(f"Command exited; sampling stable state for {args.stable_seconds:.0f}s", flush=True)
        while time.monotonic() < deadline:
            samples.append(_sample(start, query_gpu_inventory()))
            time.sleep(args.interval)
    if not samples:
        samples.append(_sample(start, query_gpu_inventory()))

    summary = _summarize(baseline, samples, args.stable_window)
    if args.gpu is not None:
        summary = {str(args.gpu): summary[str(args.gpu)]}
    artifact = {
        "schema_version": 1,
        "kind": "xr-ai-vram-measurement",
        "label": args.label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "measurement_signature": _measurement_signature(command),
        "sampling_interval_s": args.interval,
        "stable_window_s": args.stable_window,
        "margin_policy": {
            "variable_percent": 10,
            "fixed_gib": 1.0,
            "formula": "ceil_0.1(peak_delta * 1.10 + 1.0)",
        },
        "summary": summary,
        "samples": samples,
    }
    output = Path(args.output)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"VRAM measurement written to {output}", flush=True)
    for index, result in summary.items():
        print(
            f"GPU {index}: peak delta {result['observed_peak_delta_gib']:.1f} GiB, "
            f"recommended reservation {result['recommended_reservation_gib']:.1f} GiB",
            flush=True,
        )
    return process.returncode or 0


def _service_spec(value: str) -> tuple[str, int, bool, Path]:
    try:
        name, gpu_text, runtime, path_text = value.split(":", maxsplit=3)
        gpu = int(gpu_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "service must be NAME:GPU:vllm|other:MEASUREMENT.json"
        ) from exc
    if runtime not in {"vllm", "other"}:
        raise argparse.ArgumentTypeError("service runtime must be vllm or other")
    return name, gpu, runtime == "vllm", Path(path_text)


def certify(args: argparse.Namespace) -> int:
    grouped: dict[str, list[tuple[int, bool, Path, dict]]] = {}
    for name, gpu, is_vllm, path in args.service:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("kind") != "xr-ai-vram-measurement":
            raise SystemExit(f"{path} is not an XR-AI VRAM measurement")
        grouped.setdefault(name, []).append((gpu, is_vllm, path, raw))

    services: dict[str, dict] = {}
    sources: list[str] = []
    for name, runs in grouped.items():
        if len(runs) < args.minimum_runs:
            raise SystemExit(
                f"{name} has {len(runs)} measurement run(s); "
                f"certification requires {args.minimum_runs}"
            )
        layouts = {(gpu, is_vllm) for gpu, is_vllm, _path, _raw in runs}
        if len(layouts) != 1:
            raise SystemExit(f"{name} measurement runs disagree on GPU or runtime")
        signatures = [raw.get("measurement_signature") for _, _, _, raw in runs]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise SystemExit(f"{name} measurement signatures do not match")
        gpu, is_vllm = next(iter(layouts))
        reservations: list[float] = []
        measurements: list[str] = []
        for _, _, path, raw in runs:
            try:
                reservations.append(float(
                    raw["summary"][str(gpu)]["recommended_reservation_gib"]
                ))
            except KeyError as exc:
                raise SystemExit(f"{path} has no measurement for GPU {gpu}") from exc
            measurements.append(str(path))
            sources.append(str(path))
        services[name] = {
            "gpu": gpu,
            "reservation_gib": max(reservations),
            "vllm": is_vllm,
            "measurements": measurements,
            "measurement_signature": signatures[0],
        }

    profile = {
        "schema_version": 1,
        "hardware_profile": args.hardware_profile,
        "stack": args.stack,
        "status": "certified",
        "device_safety_reserve_gib": args.device_safety_reserve,
        "source": f"generated from {len(sources)} measurement artifact(s)",
        "services": services,
    }
    output = Path(args.output)
    output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(f"Certified VRAM profile written to {output}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure and certify XR-AI VRAM profiles")
    subparsers = parser.add_subparsers(dest="action", required=True)

    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--label", required=True)
    measure_parser.add_argument("--output", required=True)
    measure_parser.add_argument("--gpu", type=int)
    measure_parser.add_argument("--interval", type=float, default=0.25)
    measure_parser.add_argument("--stable-seconds", type=float, default=30.0)
    measure_parser.add_argument("--stable-window", type=float, default=15.0)
    measure_parser.add_argument("command", nargs=argparse.REMAINDER)
    measure_parser.set_defaults(func=measure)

    certify_parser = subparsers.add_parser("certify")
    certify_parser.add_argument("--hardware-profile", required=True)
    certify_parser.add_argument("--stack", required=True)
    certify_parser.add_argument("--output", required=True)
    certify_parser.add_argument("--device-safety-reserve", type=float, default=2.0)
    certify_parser.add_argument("--minimum-runs", type=int, default=3)
    certify_parser.add_argument(
        "--service", action="append", required=True, type=_service_spec,
        metavar="NAME:GPU:vllm|other:MEASUREMENT.json",
    )
    certify_parser.set_defaults(func=certify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
