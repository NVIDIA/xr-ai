# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only sample discovery and host preflight, not runtime readiness."""

import argparse
import ctypes
import ctypes.util
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from xr_ai_launcher import GPUDevice, GPUInventoryError, detect_gpu_config, query_gpu_inventory


def run(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True, stderr=subprocess.PIPE, timeout=8).strip()


def major(value: str) -> int:
    match = re.match(r"^\d+", value)
    return int(match[0]) if match else 0


def library_available(name: str) -> bool:
    if ctypes.util.find_library(name):
        return True
    try:
        ctypes.CDLL(f"lib{name}.so.1")
    except OSError:
        return False
    return True


def encoder_support(devices: tuple[GPUDevice, ...]) -> dict[int, dict]:
    """Bound native-library failures and CUDA context lifetime to each child."""
    support = {}
    for gpu in devices:
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("_nvenc_probe.py"))],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu.uuid},
                capture_output=True, text=True, timeout=15, check=True,
            )
            value = json.loads(result.stdout.strip().splitlines()[-1])
            if (not isinstance(value, dict) or "nvenc" not in value
                    or not any(value["nvenc"] is state for state in (True, False, None))
                    or not isinstance(value.get("nvenc_error"), str)):
                raise ValueError("invalid encoder probe response")
            support[gpu.index] = value
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
            support[gpu.index] = {"nvenc": None, "nvenc_error": detail}
    return support


def catalog(root: Path) -> dict:
    root = root.resolve()
    common, gpus = [], []
    driver, profile = "", ""
    if not sys.platform.startswith("linux"):
        common.append("Linux is required")
    else:
        try:
            devices = query_gpu_inventory()
        except GPUInventoryError as exc:
            common.append(str(exc))
        else:
            encoders = encoder_support(devices)
            gpus = [{
                "index": gpu.index, "name": gpu.name, "compute": gpu.compute_capability,
                "memory_mib": None if gpu.total_memory_gib is None else round(gpu.total_memory_gib * 1024),
                **encoders[gpu.index],
            } for gpu in devices]
            if not any(gpu["nvenc"] is True for gpu in gpus):
                common.append("NVENC support could not be verified" if any(gpu["nvenc"] is None for gpu in gpus)
                              else "at least one NVENC-capable GPU is required")
            try:
                driver = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[0]
            except (OSError, subprocess.SubprocessError, IndexError) as exc:
                common.append(f"NVIDIA driver version query failed: {exc}")
            else:
                if major(driver) < 580:
                    common.append("NVIDIA driver 580+ is required")
            try:
                profile = detect_gpu_config(devices)
            except GPUInventoryError as exc:
                common.append(str(exc))
        for argv, error, valid in (
            (["docker", "version", "--format", "{{.Server.Version}}"], "Docker 24+ is required",
             lambda value: major(value) >= 24),
            (["docker", "info", "--format", "{{json .Runtimes}}"], "Docker NVIDIA runtime is unavailable",
             lambda value: '"nvidia"' in value),
        ):
            try:
                value = run(argv)
            except (OSError, subprocess.SubprocessError) as exc:
                common.append(f"{error}: {exc}")
            else:
                if not valid(value):
                    common.append(error)
        if not all(library_available(name) for name in ("nvidia-encode", "nvcuvid")):
            common.append("NVENC/NVDEC driver libraries are unavailable")

    demos = []
    for pyproject in sorted((root / "agent-samples").glob("*/pyproject.toml")):
        item = {"id": pyproject.parent.name, "description": "", "command": "",
                "project": pyproject.parent.relative_to(root).as_posix(), "blockers": list(common)}
        try:
            project = tomllib.loads(pyproject.read_text()).get("project", {})
            if not isinstance(project, dict):
                raise ValueError("project must be a table")
            dependencies = project.get("dependencies", [])
            if not isinstance(dependencies, list):
                raise ValueError("project.dependencies must be an array")
            if not any(str(value).startswith("xr-ai-launcher") for value in dependencies):
                continue
            for key in ("name", "description"):
                if not isinstance(project.get(key), str) or not project[key].strip():
                    raise ValueError(f"project.{key} must be a non-empty string")
            item["id"], item["description"] = project["name"], project["description"]
            scripts = project.get("scripts", {})
            if not isinstance(scripts, dict) or len(scripts) != 1:
                raise ValueError("inventory requires exactly one sample console command")
            command = next(iter(scripts))
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", command):
                raise ValueError("invalid sample console command")
            item["command"] = command
        except (OSError, ValueError) as exc:
            item["blockers"].append(f"{pyproject.relative_to(root)}: {exc}")
        if profile and not (pyproject.parent / "model-servers/yaml" / f"{profile}.yaml").is_file():
            item["blockers"].append(f"No matching GPU profile: {profile}")
        item["runnable"] = not item["blockers"]
        demos.append(item)
    try:
        commit = run(["git", "-C", str(root), "rev-parse", "HEAD"])
    except (OSError, subprocess.SubprocessError):
        commit = ""  # An exported source tree need not include Git metadata.
    return {"commit": commit, "driver_version": driver, "gpus": gpus, "demos": demos}


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="XR AI source directory")
    parser.add_argument("--resolve", metavar="SAMPLE", help="resolve an explicit sample project name")
    options = parser.parse_args(args)
    result = catalog(options.root)
    if options.resolve is None:
        print(json.dumps(result, separators=(",", ":")))
        return
    selected = next((item for item in result["demos"] if item["id"] == options.resolve), None)
    if selected is None:
        parser.error(f"unknown sample: {options.resolve}")
    if not selected["runnable"]:
        print("; ".join(selected["blockers"]), file=sys.stderr)
        raise SystemExit(1)
    print(f"{selected['id']}\t{selected['project']}\t{selected['command']}")


if __name__ == "__main__":
    main()
