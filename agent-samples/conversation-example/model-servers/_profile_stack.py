# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-owned Gemma 4/D3 profile composition using existing launcher APIs."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from xr_ai_launcher import Process, detect_gpu_config, require_credentials, run_stack
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config
from xr_ai_vllm import stop_persistent_servers

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


def _configs(base: Path, profile: Path, gpu_profile: str | None) -> dict[str, dict]:
    overrides = _read(profile)["servers"]
    if not isinstance(overrides, dict) or set(overrides) != set(_SERVICES):
        raise ValueError(f"{profile}: servers must contain exactly {', '.join(_SERVICES)}")
    hardware = {}
    hardware_path = base / "yaml" / "gemma4" / f"{gpu_profile}.yaml"
    if gpu_profile is not None:
        available = {path.stem for path in hardware_path.parent.glob("*.yaml")}
        if gpu_profile not in available:
            raise ValueError(f"unknown GPU profile {gpu_profile!r}; available: {', '.join(sorted(available))}")
        hardware = _read(hardware_path)
        if set(hardware) != set(_SERVICES):
            raise ValueError(f"{hardware_path}: must contain exactly {', '.join(_SERVICES)}")
    configs = {}
    for name, (project, command) in _SERVICES.items():
        defaults_path = base.parents[2] / "services" / project / f"{command}.yaml"
        config = _read(defaults_path)
        cache_source = defaults_path
        for source, override in ((hardware_path, hardware.get(name, {})), (profile, overrides[name])):
            if not isinstance(override, dict):
                raise ValueError(f"{source}: {name} must be a mapping")
            config.update(override)
            if "model_cache" in override:
                cache_source = source
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


def _run_selected_profile(base: Path, *, stop: bool, gpu_profile: str | None) -> bool:
    worker_yaml = base.parent / "yaml" / "conversation_example_worker.yaml"
    selected = _read(worker_yaml).get("models_config", "models.json")
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError(f"{worker_yaml}: models_config must be a non-empty path")
    profile = (worker_yaml.parent / selected).resolve()
    if "servers" not in _read(profile):
        return False

    log_file = setup_logging("orchestrator", namespace="conversation-example-models")
    configs = _configs(base, profile, gpu_profile if stop else gpu_profile or detect_gpu_config())
    if stop:
        if not stop_persistent_servers([(name, cfg["port"]) for name, cfg in configs.items()]):
            raise RuntimeError("one or more configured model servers could not be stopped")
        return True

    require_credentials("HF_TOKEN")
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
