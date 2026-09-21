# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch this sample's Riva NIMs with persistent compiled repositories."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from xr_ai_logging import setup_logging
from xr_ai_vllm import _docker, load_config
from xr_ai_vllm._nim import build_nim_run_argv

_SCRIPT = Path(__file__).with_name("repository.py")
_CONTAINER_SCRIPT = "/opt/xr-ai/riva_repository.py"
_RESERVED_ENV = {
    "NIM_CACHE_PATH", "NIM_EXPORT_PATH", "NIM_DISABLE_MODEL_DOWNLOAD", "NIM_WORKSPACE",
    "XR_AI_RIVA_CONTRACT", "XR_AI_RIVA_COMMAND", "XR_AI_RIVA_BOOTSTRAP",
}


def _image_info(image: str) -> dict:
    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    if result.returncode:
        _docker._maybe_ngc_login(image)
        subprocess.run(["docker", "pull", image], check=True)
        result = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, text=True, check=True,
        )
    return json.loads(result.stdout)[0]


def _launch_args(cfg: dict, cache: Path, image: dict) -> list[str]:
    env = {str(k): str(v) for k, v in cfg.get("env", {}).items()}
    reserved = env.keys() & _RESERVED_ENV
    if reserved:
        raise ValueError(f"Riva repository caching manages these environment settings: {sorted(reserved)}")
    command = (image["Config"].get("Entrypoint") or []) + (image["Config"].get("Cmd") or [])
    if not command:
        raise ValueError("the Riva image must declare its startup command")
    # Hash configured settings rather than storing possible credentials in the
    # export manifest. The image ID also invalidates exports for mutable tags.
    contract = {
        "image": image["Id"],
        "settings": hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest(),
        # Bump only when build/export semantics make existing engines incompatible.
        "build_format": 1,
    }
    env.update({
        "XR_AI_RIVA_CONTRACT": json.dumps(contract, sort_keys=True),
        "XR_AI_RIVA_COMMAND": json.dumps(command),
        # Apply bootstrap fixes to containers without invalidating compiled engines.
        "XR_AI_RIVA_BOOTSTRAP": hashlib.sha256(_SCRIPT.read_bytes()).hexdigest(),
    })
    args = build_nim_run_argv(
        image=image["Id"], container_name=str(cfg["container_name"]),
        http_port=int(cfg["http_port"]), grpc_port=int(cfg["grpc_port"]),
        nim_cache=cache, cuda_visible_devices=str(cfg.get("cuda_visible_devices", "all")),
        extra_env=env,
    )
    # The shared lifecycle owns this container during both compilation and
    # serving, including --stop before a health endpoint is available.
    if args[-1] != image["Id"]:
        raise RuntimeError("NIM launch arguments must end with the image before adding the Riva entrypoint")
    return args[:-1] + [
        "--init", "--entrypoint", "python3",
        "--mount", f"type=bind,src={_SCRIPT},dst={_CONTAINER_SCRIPT},readonly",
        args[-1], _CONTAINER_SCRIPT,
    ]


def run() -> None:
    """Read the speech server YAML and start a managed build-and-serve container."""
    setup_logging("riva-server")
    cfg, yaml_dir, ready_file = load_config()
    for key in ("image", "container_name", "http_port", "grpc_port"):
        if not cfg.get(key):
            raise ValueError(f"{key!r} is required in the Riva server config")
    os.environ["NGC_API_KEY"] = os.environ.get("NGC_API_KEY", "").strip()
    if not os.environ["NGC_API_KEY"]:
        raise ValueError("NGC_API_KEY is required to launch the Riva NIM")
    cache = (yaml_dir / cfg.get("nim_cache", "../../models/nim")).resolve() / cfg["container_name"]
    cache.mkdir(parents=True, exist_ok=True)
    try:
        cache.chmod(0o777)
    except PermissionError:
        if cache.stat().st_mode & 0o777 != 0o777:
            raise
    image = _image_info(str(cfg["image"]))
    args = _launch_args(cfg, cache, image)
    port = int(cfg["http_port"])
    name = str(cfg["container_name"])
    _docker.run_container(
        argv=args, image=str(cfg["image"]), container_name=name, log_prefix=name,
        port=port, health_url=f"http://127.0.0.1:{port}/v1/health/ready",
        launch_banner=f"Launching Riva; compiled repositories persist under {cache / 'riva-repositories'}",
        reuse_banner=f"Riva already serving on port {port}, reusing",
        ready_banner=f"Ready → Riva on port {port}", ready_file=ready_file,
    )


if __name__ == "__main__":
    run()
