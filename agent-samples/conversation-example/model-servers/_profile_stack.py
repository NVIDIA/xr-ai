# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-owned Gemma 4/D3 profile composition using existing launcher APIs."""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from xr_ai_gpu import GPUDevice, query_gpu_inventory
from xr_ai_launcher import Process, pid_on_port_checked, require_credentials, run_stack
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config
from xr_ai_vllm import stop_persistent_servers

_logger = logging.getLogger(__name__)

_SERVICES = {
    "speaker-stt": ("speaker-stt", "speaker_stt"),
    "tts": ("pocket-tts", "pocket_tts_server"),
    "gemma4": ("gemma4-server", "gemma4_server"),
    "d3": ("d3-server", "d3_server"),
}


def _read(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: could not read YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def _configs(base: Path, profile: Path) -> dict[str, dict]:
    overrides = _read(profile)["servers"]
    if not isinstance(overrides, dict) or set(overrides) != set(_SERVICES):
        raise ValueError(f"{profile}: servers must contain exactly {', '.join(_SERVICES)}")
    portable_path = base / "yaml" / "gemma4.yaml"
    portable = _read(portable_path)
    if set(portable) != set(_SERVICES):
        raise ValueError(f"{portable_path}: must contain exactly {', '.join(_SERVICES)}")
    configs = {}
    for name, (project, command) in _SERVICES.items():
        defaults_path = base.parents[2] / "services" / project / f"{command}.yaml"
        config = _read(defaults_path)
        cache_source = defaults_path
        for source, override in ((portable_path, portable[name]), (profile, overrides[name])):
            if not isinstance(override, dict):
                raise ValueError(f"{source}: {name} must be a mapping")
            config.update(override)
            if "model_cache" in override:
                cache_source = source
        if "cuda_visible_devices" in config:
            raise ValueError(f"{name}: set CUDA_VISIBLE_DEVICES to restrict GPU selection, "
                             "rather than per-service cuda_visible_devices")
        if name in {"gemma4", "d3"} and config.get("tensor_parallel_size", 1) != 1:
            raise ValueError(f"{name}: the portable profile uses one GPU; tensor_parallel_size must be 1")
        cache = config.get("model_cache")
        if not isinstance(cache, str) or not cache:
            raise ValueError(f"{name}: model_cache must be a non-empty path")
        config["model_cache"] = str((cache_source.parent / cache).resolve())
        port = config.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"{name}: port must be an integer between 1 and 65535")
        configs[name] = config
    if len({cfg["port"] for cfg in configs.values()}) != len(configs):
        raise ValueError("model services must use distinct ports")

    models = load_models_config(profile)
    for role, service in (("llm", "gemma4"), ("vlm", "gemma4"), ("router", "d3")):
        spec = models.entries[role]
        server = configs[service]
        if spec.model_name != server["served_model_name"] or urlsplit(spec.base_url).port != server["port"]:
            raise ValueError(f"{profile}: {role} endpoint and model alias must match the {service} server")
    return configs


def _required_free_memory(profile: Path) -> float:
    value = _read(profile).get("min_free_memory_gib")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{profile}: min_free_memory_gib must be a finite positive number")
    return float(value)


def _select_gpu(required_gib: float) -> GPUDevice:
    devices = query_gpu_inventory()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = visible.split(",")
        selected = []
        for token in tokens:
            matches = [gpu for gpu in devices if token.strip() in (str(gpu.index), gpu.uuid)]
            if len(matches) != 1:
                raise ValueError("CUDA_VISIBLE_DEVICES must list visible driver indices or full GPU UUIDs")
            selected.extend(matches)
        devices = tuple(selected)
    candidates = [
        gpu for gpu in devices
        if gpu.compute_capability >= 8.0
        and gpu.free_memory_gib is not None and math.isfinite(gpu.free_memory_gib)
        and gpu.total_memory_gib is not None and math.isfinite(gpu.total_memory_gib)
        and gpu.total_memory_gib > 0 and gpu.free_memory_gib >= required_gib
    ]
    if not candidates:
        memory = ", ".join(
            f"GPU {gpu.index}: {gpu.free_memory_gib:.1f} GiB free" if gpu.free_memory_gib is not None
            else f"GPU {gpu.index}: free memory unavailable"
            for gpu in devices
        ) or "no visible GPUs"
        raise RuntimeError(
            f"Gemma stack needs one BF16-capable GPU (SM80+) with {required_gib:g} GiB free; {memory}. "
            "Stop the current model stack before relaunching or changing placement."
        )
    return max(candidates, key=lambda gpu: gpu.free_memory_gib)


def _place_on_gpu(configs: dict[str, dict], gpu: GPUDevice) -> None:
    for config in configs.values():
        config["cuda_visible_devices"] = gpu.uuid
    # Explicit KV caps own the vLLM cache size. Its separate fractional startup
    # check must not demand half of a large GPU after sibling services load;
    # aggregate model memory was checked against the profile's GiB estimate.
    total_bytes = gpu.total_memory_gib * 1024**3
    for name in ("gemma4", "d3"):
        config = configs[name]
        cache = config["kv_cache_memory_bytes"]
        if isinstance(cache, bool) or not isinstance(cache, int) or not 0 < cache < total_bytes:
            raise ValueError(f"{name}: kv_cache_memory_bytes must be positive and smaller than GPU memory")
        config["gpu_memory_utilization"] = min(float(config["gpu_memory_utilization"]), cache / total_bytes)


def _require_free_ports(configs: dict[str, dict]) -> None:
    # Placement is decided for a fresh stack, never inferred from occupied
    # memory or used to relocate a running model as a side effect of startup.
    for name, config in configs.items():
        _, checked, listening = pid_on_port_checked(config["port"])
        if not checked:
            raise RuntimeError(f"cannot inspect {name} port {config['port']}")
        if listening:
            raise RuntimeError(
                f"{name} port {config['port']} is occupied; stop the current model stack with --stop "
                "before launching the Gemma stack."
            )


def _run_selected_profile(base: Path, *, stop: bool, gpu_profile: str | None) -> bool:
    worker_yaml = base.parent / "yaml" / "conversation_example_worker.yaml"
    selected = _read(worker_yaml).get("models_config", "models.json")
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError(f"{worker_yaml}: models_config must be a non-empty path")
    profile = (worker_yaml.parent / selected).resolve()
    if "servers" not in _read(profile):
        return False

    if gpu_profile is not None:
        raise ValueError("--gpu-profile applies to the Omni stack; Gemma selects by free memory. "
                         "Use CUDA_VISIBLE_DEVICES to restrict GPU selection.")
    configs = _configs(base, profile)
    required_gib = _required_free_memory(profile)
    if stop:
        if not stop_persistent_servers([(name, cfg["port"]) for name, cfg in configs.items()]):
            raise RuntimeError("one or more configured model servers could not be stopped")
        return True

    _require_free_ports(configs)
    gpu = _select_gpu(required_gib)
    _place_on_gpu(configs, gpu)
    require_credentials("HF_TOKEN")
    log_file = setup_logging("orchestrator", namespace="conversation-example-models")
    _logger.info("Gemma stack on GPU %s (%s): %.1f GiB free, %.1f GiB required",
                 gpu.index, gpu.uuid, gpu.free_memory_gib, required_gib)
    directory = Path(tempfile.mkdtemp(prefix="model-configs-", dir=log_file.parent))
    try:
        processes = []
        for name, (project, command) in _SERVICES.items():
            path = directory / f"{command}.yaml"
            path.write_text(yaml.safe_dump(configs[name], sort_keys=False), encoding="utf-8")
            path.chmod(0o600)
            processes.append(Process(
                name, base.parents[2] / "services" / project, command,
                config=path, launch_mode="persist", port=configs[name]["port"],
            ))
        run_stack(processes, base, exit_after_ready=True)
    except BaseException:
        # Retain diagnostics on failure without persisting inline credentials.
        for path in directory.glob("*.yaml"):
            config = _read(path)
            if config.get("hf_token"):
                path.write_text(yaml.safe_dump({**config, "hf_token": ""}), encoding="utf-8")
        raise
    else:
        shutil.rmtree(directory)
    return True
