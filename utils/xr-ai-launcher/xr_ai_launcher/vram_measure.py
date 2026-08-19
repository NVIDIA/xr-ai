# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeatable GPU-memory measurement and service-YAML certification.

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
from ._vram import service_config_fingerprint


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
    driver_output = _output([
        "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
    ])
    driver = (
        ",".join(dict.fromkeys(driver_output.splitlines()))
        if driver_output else None
    )
    return {
        "git_commit": _output(["git", "rev-parse", "HEAD"]),
        "driver_version": driver,
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


def _yaml_value(value: str | float) -> str:
    return str(value) if isinstance(value, float) else json.dumps(value)


def _set_yaml_scalars(path: Path, values: dict[str, str | float]) -> None:
    """Replace or append top-level scalar keys without requiring PyYAML."""
    lines = path.read_text(encoding="utf-8").splitlines()
    remaining = dict(values)
    updated: list[str] = []
    for line in lines:
        key = line.split(":", 1)[0].strip() if ":" in line else ""
        if not line.startswith((" ", "\t")) and key in remaining:
            updated.append(f"{key}: {_yaml_value(remaining.pop(key))}")
        else:
            updated.append(line)
    if remaining:
        if updated and updated[-1]:
            updated.append("")
        updated.extend(f"{key}: {_yaml_value(value)}" for key, value in remaining.items())
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def certify(args: argparse.Namespace) -> int:
    runs: list[tuple[Path, dict]] = []
    for path_text in args.measurement:
        path = Path(path_text)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("kind") != "xr-ai-vram-measurement":
            raise SystemExit(f"{path} is not an XR-AI VRAM measurement")
        runs.append((path, raw))
    if len(runs) < args.minimum_runs:
        raise SystemExit(
            f"got {len(runs)} measurement run(s); certification requires "
            f"{args.minimum_runs}"
        )
    signatures = [raw.get("measurement_signature") for _, raw in runs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise SystemExit("measurement signatures do not match")
    signature = signatures[0] or {}
    driver = signature.get("driver_version")
    git_commit = signature.get("git_commit")
    if not driver or not git_commit:
        raise SystemExit("measurement signature lacks driver or git commit")
    reservations: list[float] = []
    for path, raw in runs:
        try:
            reservations.append(float(
                raw["summary"][str(args.gpu)]["recommended_reservation_gib"]
            ))
        except KeyError as exc:
            raise SystemExit(f"{path} has no measurement for GPU {args.gpu}") from exc

    config = Path(args.config)
    _set_yaml_scalars(config, {
        "gpu_memory_reservation_gib": max(reservations),
        "gpu_memory_reservation_status": "certified",
        "gpu_memory_certification_driver": driver,
        "gpu_memory_certification_git": git_commit,
    })
    _set_yaml_scalars(config, {
        "gpu_memory_certification_sha256": service_config_fingerprint(config),
    })
    print(f"Certified VRAM reservation written to {config}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure and certify XR-AI VRAM reservations")
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
    certify_parser.add_argument("--config", required=True)
    certify_parser.add_argument("--gpu", required=True, type=int)
    certify_parser.add_argument("--minimum-runs", type=int, default=3)
    certify_parser.add_argument(
        "--measurement", action="append", required=True,
        metavar="MEASUREMENT.json",
    )
    certify_parser.set_defaults(func=certify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
