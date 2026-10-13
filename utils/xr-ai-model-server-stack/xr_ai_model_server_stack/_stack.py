# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose sample configs and delegate lifecycle to existing server utilities."""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from xr_ai_launcher import Process, detect_gpu_config, require_credentials, run_stack
from xr_ai_logging import setup_logging
from xr_ai_vllm import stop_persistent_servers

_COMMON = Path(__file__).resolve().parents[1] / "yaml"
# The explicit order leaves Omni's compilation ahead of Cosmos initialization.
_SERVICES = {
    "stt": ("stt-server", "stt_server", "stt_server"),
    "tts": ("pocket-tts", "pocket_tts_server", "pocket_tts_server"),
    "omni": ("nemotron-omni-llm", "nemotron_omni_llm_server", "nemotron_omni_llm_server"),
    "vlm": ("vlm-server", "vlm_server", "vlm_server"),
    "embedding": ("embedding-server", "embedding_server", "embedding_server"),
}


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: could not read YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def _configs(base: Path, services: tuple[str, ...], profile: str) -> dict[str, dict]:
    profiles = {path.stem for path in (base / "yaml").glob("*.yaml")}
    if profile not in profiles:
        raise ValueError(f"unknown GPU profile {profile!r}; available: {', '.join(sorted(profiles))}")
    overrides_path = base / "yaml" / f"{profile}.yaml"
    overrides = _read_yaml(overrides_path)
    if set(overrides) != set(services):
        raise ValueError(f"{overrides_path}: must contain exactly these servers: {', '.join(services)}")
    configs = {}
    for service in services:
        if service not in _SERVICES:
            raise ValueError(f"unknown model service {service!r}")
        common_path = _COMMON / f"{_SERVICES[service][2]}.yaml"
        config = _read_yaml(common_path)
        override = overrides[service]
        if not isinstance(override, dict):
            raise ValueError(f"{overrides_path}: {service} must be a mapping")
        config.update(override)
        # Servers resolve relative cache paths against the supplied YAML. Resolve
        # before materializing in a temporary directory; overrides belong to the sample.
        cache_source = overrides_path if "model_cache" in override else common_path
        cache_value = config.get("model_cache")
        if not isinstance(cache_value, str) or not cache_value:
            raise ValueError(f"{service}: model_cache must be a non-empty path")
        cache = Path(cache_value)
        config["model_cache"] = str((cache_source.parent / cache).resolve())
        port = config.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"{service}: port must be an integer between 1 and 65535")
        configs[service] = config
    if len({config["port"] for config in configs.values()}) != len(configs):
        raise ValueError("model services must use distinct ports")
    return configs


def _processes(configs: dict[str, dict], directory: Path, root: Path) -> list[Process]:
    processes = []
    for service, (project, command, basename) in _SERVICES.items():
        if service not in configs:
            continue
        config = configs[service]
        path = directory / f"{basename}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        processes.append(Process(
            service, str(root / "services" / project), command,
            config=path, launch_mode="persist", port=config["port"],
        ))
    return processes


def _run(base: Path, services: tuple[str, ...], *, stop: bool, gpu_profile: str | None) -> None:
    setup_logging("orchestrator", namespace=f"{base.parent.name}-models")
    configs = _configs(base, services, gpu_profile or detect_gpu_config())
    if stop:
        if not stop_persistent_servers([(name, config["port"]) for name, config in configs.items()]):
            raise RuntimeError("one or more configured model servers could not be stopped")
        return
    require_credentials("HF_TOKEN")
    # Configs stay available through child initialization and readiness. Services
    # read their config at startup; persistent servers do not reopen it afterward.
    with tempfile.TemporaryDirectory(prefix=f"{base.parent.name}-models-") as directory:
        processes = _processes(configs, Path(directory), base.parents[2])
        run_stack(processes, base, exit_after_ready=True)
