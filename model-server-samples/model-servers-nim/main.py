# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start a persistent NIM model stack, with hardware-specific local fallbacks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from xr_ai_launcher import (
    GPUInventoryError,
    Process,
    detect_gpu_config,
    load_deployment_profile,
    read_config_scalar,
    require_credentials,
    run_stack,
)
from xr_ai_logging import setup_logging
from xr_ai_vllm import stop_persistent_servers
from xr_ai_vllm._docker import has_xr_ai_ownership_marker, pid_on_port_checked

_BASE = Path(__file__).resolve().parent

# Fixed-size services start before the two language models so their runtime
# allocations are visible to the language models' memory profilers.
_SERVICES = {
    "stt": ("../../services/stt-server", "stt_server", "stt_server.yaml"),
    "stt-nim": ("riva-server", "nim_riva_server", "nim_stt_server.yaml"),
    "stt-adapter": ("compatibility-adapter", "nim_model_adapter", "stt_adapter.yaml"),
    "tts-nim": ("riva-server", "nim_riva_server", "nim_tts_server.yaml"),
    "tts-adapter": ("../../services/magpie-nim-tts", "magpie_nim_tts", "tts_adapter.yaml"),
    "embedding-nim": (
        "../../services/nim-server", "nim_server", "nim_embedding_server.yaml",
    ),
    "embedding": ("compatibility-adapter", "nim_model_adapter", "embedding_adapter.yaml"),
    "llm-nim": ("../../services/nim-server", "nim_server", "nim_llm_server.yaml"),
    "llm-adapter": ("compatibility-adapter", "nim_model_adapter", "llm_adapter.yaml"),
    "vlm-nim": ("../../services/nim-server", "nim_server", "nim_vlm_server.yaml"),
    "vlm-adapter": ("compatibility-adapter", "nim_model_adapter", "vlm_adapter.yaml"),
}
_BACKENDS = {
    "stt-adapter": "stt-nim", "tts-adapter": "tts-nim",
    "llm-adapter": "llm-nim", "vlm-adapter": "vlm-nim", "embedding": "embedding-nim",
}


def _gpu_profile_name(value: str) -> str:
    names = sorted(path.parent.name for path in (_BASE / "yaml").glob("*/models.json"))
    if value not in names:
        raise argparse.ArgumentTypeError(f"unknown GPU profile {value!r}; available profiles: {', '.join(names)}")
    return value


def _port(config: Path) -> int:
    raw = read_config_scalar(config, "http_port") or read_config_scalar(config, "port")
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"{config}: an integer http_port or port is required") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{config}: port must be between 1 and 65535")
    return port


def _build_processes(
    gpu_profile: str, models: Path | None = None,
) -> tuple[list[Process], tuple[str, ...], Path]:
    config_dir = _BASE / "yaml" / gpu_profile
    profile = models or config_dir / "models.json"
    deployment = load_deployment_profile(profile)
    unknown = deployment.services.keys() - _SERVICES.keys()
    if unknown:
        raise ValueError(f"unknown model services: {sorted(unknown)}")
    managed = {service for service in deployment.services if deployment.launch_mode(service) == "own"}
    managed.update(
        _BACKENDS[service] for service in tuple(managed)
        if service in _BACKENDS and deployment.launch_mode(_BACKENDS[service]) != "reuse"
    )
    processes = []
    ports: set[int] = set()
    for service, (project, command, filename) in _SERVICES.items():
        if service not in managed:
            continue
        config = config_dir / filename
        if not config.is_file():
            raise ValueError(f"GPU profile {gpu_profile!r} is missing {config}")
        port = _port(config)
        if port in ports:
            raise ValueError(f"multiple services use port {port}")
        ports.add(port)
        processes.append(Process(
            service, project, command, config=config, port=port,
            launch_mode="persist",
        ))
    return processes, deployment.required_credentials, profile


def _export_models(profile: Path, destination: Path) -> None:
    """Write a client profile without giving a consumer ownership of servers."""
    if destination.resolve() == profile.resolve():
        raise ValueError("the client export must not overwrite the deployment profile")
    data = json.loads(profile.read_text(encoding="utf-8"))
    for model in data["models"].values():
        deployment = model["deployment"]
        if deployment.get("ownership", "external") == "managed":
            deployment["ownership"] = "reused"
        # NGC and HF credentials belong to the server process, not its clients.
        deployment.pop("credentials", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _known_ports() -> list[tuple[str, int]]:
    candidates: dict[int, set[str]] = {}
    for service, (_, _, filename) in _SERVICES.items():
        for config in (_BASE / "yaml").glob(f"*/{filename}"):
            candidates.setdefault(_port(config), set()).add(service)
    targets = []
    for port, services in candidates.items():
        if "stt" in services and "stt-adapter" in services:
            # Spark's unmarked STT process is recognized by its command line;
            # the HTTP adapter must instead use managed-port ownership markers.
            pid, _, _ = pid_on_port_checked(port)
            service = "stt-adapter" if pid and has_xr_ai_ownership_marker(pid, port) else "stt"
        else:
            service = sorted(services)[0]
        targets.append((service, port))
    return sorted(targets)


def _stop_unselected_services(processes: list[Process], profile: Path) -> None:
    """Free omitted services while preserving explicitly reused endpoints and dependencies."""
    deployment = load_deployment_profile(profile)
    selected_services = {name for name, mode in deployment.services.items() if mode == "reuse"}
    # A reused adapter still needs its backend even though neither is launched.
    selected_services.update(_BACKENDS[name] for name in tuple(selected_services) if name in _BACKENDS)
    selected_ports = {process.port for process in processes}
    # Explicit local endpoints can be external to this launcher's ownership.
    # They must survive cleanup even when they use one of our known ports.
    models = json.loads(profile.read_text(encoding="utf-8"))["models"]
    for model in models.values():
        if url := model["endpoint"].get("base_url"):
            endpoint = urlsplit(url if "://" in url else "//" + url)
            if endpoint.hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} and endpoint.port:
                selected_ports.add(endpoint.port)
    unselected = [(service, port) for service, port in _known_ports()
                  if service not in selected_services and port not in selected_ports]
    if not stop_persistent_servers(unselected):
        raise RuntimeError("could not stop persistent model servers outside the selected profile")


def run() -> None:
    """Launch, inspect, export client settings, or stop the shared model stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--stop", action="store_true", help="Stop this stack's model ports.")
    actions.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print the launch plan without credentials, Docker, or model downloads.",
    )
    actions.add_argument(
        "--export-models", type=Path, metavar="PATH",
        help="Write a reusable client models JSON and exit without launching servers.",
    )
    parser.add_argument(
        "--gpu-profile", metavar="NAME", type=_gpu_profile_name,
        help="Select a hardware profile; otherwise detect Blackwell, dual Ada, or Spark.",
    )
    parser.add_argument(
        "--models", type=Path, metavar="PATH",
        help="Custom deployment JSON; defaults to yaml/<gpu-profile>/models.json.",
    )
    parser.add_argument(
        "--allow-anonymous", action="store_true",
        help="Allow Hugging Face downloads without HF_TOKEN for local fallback services.",
    )
    args = parser.parse_args()
    try:
        if args.stop:
            if not stop_persistent_servers(_known_ports()):
                raise RuntimeError("one or more persistent model servers are still running")
            return
        gpu_profile = args.gpu_profile or detect_gpu_config()
        processes, credentials, profile = _build_processes(gpu_profile, args.models)
        if args.export_models:
            _export_models(profile, args.export_models)
            print(f"Client models: {args.export_models.resolve()}")
            return
        print(f"Hardware: {gpu_profile}; models: {profile}")
        for process in processes:
            print(f"  {process.name}: port {process.port}, config {process.config}")
        if args.dry_run:
            return
        if not processes:
            raise ValueError("the deployment has no managed services to start")
        setup_logging("orchestrator", namespace="model-servers-nim")
        for credential in credentials:
            require_credentials(
                credential,
                allow_missing=credential == "HF_TOKEN" and args.allow_anonymous,
            )
        _stop_unselected_services(processes, profile)
        run_stack(processes, _BASE, exit_after_ready=True)
    except (GPUInventoryError, OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    run()
