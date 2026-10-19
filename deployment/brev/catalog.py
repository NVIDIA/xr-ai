# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own sample inventory and read-only host checks for deployment tools."""

import ctypes.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path


def catalog(root: Path) -> dict:
    root = root.resolve()

    def run(argv: list[str], timeout: int = 8) -> str:
        try:
            return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL, timeout=timeout).strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    def major(value: str) -> int:
        match = re.match(r"\s*(\d+)", value or "")
        return int(match.group(1)) if match else 0

    raw = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    encoder_raw = run(["nvidia-smi", "--query-gpu=index,encoder.stats.sessionCount", "--format=csv,noheader,nounits"])
    encoders = {}
    for line in encoder_raw.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2:
            encoders[parts[0]] = parts[1].lower() not in {"", "n/a", "[n/a]", "not supported", "[not supported]"}
    gpus = []
    driver = ""
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, compute, memory, gpu_driver = parts
        try:
            memory_mib = None if memory in {"N/A", "[N/A]", "Not Supported", "[Not Supported]"} else int(float(memory))
            gpu = {
                "index": int(index),
                "name": name,
                "compute": float(compute),
                "memory_mib": memory_mib,
                "nvenc": bool(encoders.get(index)),
            }
        except ValueError:
            continue
        driver = driver or gpu_driver
        gpus.append(gpu)

    docker_version = run(["docker", "version", "--format", "{{.Server.Version}}"])
    docker_runtimes = run(["docker", "info", "--format", "{{json .Runtimes}}"])
    codec_libraries = bool(ctypes.util.find_library("nvidia-encode")) and bool(ctypes.util.find_library("nvcuvid"))
    nvenc = [gpu for gpu in gpus if gpu["nvenc"]]
    common = []
    if not sys.platform.startswith("linux"):
        common.append("Linux is required")
    if major(driver) < 580:
        common.append("NVIDIA driver 580+ is required")
    if major(docker_version) < 24:
        common.append("Docker 24+ is required")
    if '"nvidia"' not in docker_runtimes:
        common.append("Docker NVIDIA runtime is unavailable")
    if not nvenc:
        common.append("at least one NVENC-capable GPU is required")
    if not codec_libraries:
        common.append("NVENC/NVDEC driver libraries are unavailable")
    sys.path.insert(0, str(root / "utils/xr-ai-launcher"))
    try:
        from xr_ai_launcher import detect_gpu_config

        profile = detect_gpu_config()
    except (ImportError, RuntimeError) as exc:
        profile = ""
        common.append(str(exc))

    def demo(demo_id, name, description, project, command, fits=True, blockers=()):
        all_blockers = list(common) + list(blockers)
        value = {
            "id": demo_id,
            "name": name,
            "description": description,
            "project": project,
            "command": command,
            "runnable": bool(fits and not all_blockers),
            "blockers": list(dict.fromkeys(all_blockers)),
        }
        return value

    demos = []

    for pyproject in sorted((root / "agent-samples").glob("*/pyproject.toml")):
        relative = pyproject.parent.relative_to(root).as_posix()
        project = tomllib.loads(pyproject.read_text()).get("project", {})
        scripts = project.get("scripts", {})
        if not scripts or not any(str(value).startswith("xr-ai-launcher") for value in project.get("dependencies", ())):
            continue
        if len(scripts) != 1:
            raise ValueError(f"{pyproject}: expected exactly one sample console command")
        project_name = project["name"]
        command = next(iter(scripts))
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", command):
            raise ValueError(f"{pyproject}: invalid sample console command")
        models = pyproject.parent / "model-servers"
        fits = bool(profile and (models / "yaml" / f"{profile}.yaml").is_file())
        blockers = [] if fits else ["No matching GPU profile"]
        item = demo(project_name, project_name, project["description"], relative, command, fits, blockers)
        demos.append(item)

    commit = run(["git", "-C", str(root), "rev-parse", "HEAD"])
    result = {"commit": commit, "driver_version": driver, "gpus": gpus, "demos": demos}
    recommended = next((item["id"] for item in demos if item["runnable"]), "")
    result["recommended"] = recommended

    return result


def main(root: Path, args: list[str]) -> None:
    result = catalog(root)
    demos, recommended = result["demos"], result["recommended"]
    option = args[0] if args else ""
    value = args[1] if len(args) > 1 else ""
    if option == "--resolve":
        if value == "auto":
            value = recommended
            if not value:
                print("no compatible demo fits this VM", file=sys.stderr)
                raise SystemExit(1)
        selected = next((item for item in demos if item["id"] == value), None)
        if selected is None:
            print("unknown demo", file=sys.stderr)
            raise SystemExit(2)
        if not selected["runnable"]:
            print("; ".join(selected["blockers"]), file=sys.stderr)
            raise SystemExit(1)
        print(f"{selected['id']}\t{selected['project']}\t{selected['command']}")
        return
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
