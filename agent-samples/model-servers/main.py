# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
model-servers orchestrator — starts one shared AI inference stack and exits.

All servers are launch_mode="persist" so they keep running after this
process exits.  Model weights stay hot across stack restarts.

Which servers start is set by a deployment profile (``--models``): every
entry whose deployment is ``managed`` launches as the named service, so one
profile can mix local in-process servers, vLLM servers, and self-hosted NIM
containers. Shipped profiles (yaml/models.<name>.json):

  default
    stt        — nvidia/parakeet-tdt-0.6b-v3        port 8103  (NeMo ASR)
    tts        — en_US-lessac-medium                 port 8105  (Piper; CPU)
    omni       — Nemotron-3-Nano-Omni-30B-A3B       port 8108  (vLLM; llm + agent_llm)
    vlm        — nvidia/Cosmos3-Nano Reasoner       port 8100  (vLLM)
    embedding  — nvidia/llama-nemotron-embed-1b-v2  port 8109  (vLLM)

  vlm_llm_nim
    stt + tts + embedding local; the LLM and VLM as self-hosted NIM containers
    (Nemotron-3-Nano-Omni port 8110, Cosmos3-Nano Reasoner port 8100).
    Requires docker + NGC_API_KEY. Samples may reuse these endpoints.

Per-service placement (GPUs, ports, KV budgets) lives in the per-GPU-profile
YAML directory; a service may ship a profile-specific config variant named
``<config>_<profile>.yaml`` (e.g. ``embedding_server_omni.yaml``). Variants
key off the profile filename stem, so custom profiles use the service
defaults.

How to run:
    uv run --project agent-samples/model-servers model_servers
    uv run --project agent-samples/model-servers model_servers --models vlm_llm_nim

To stop all model servers:
    uv run --project agent-samples/model-servers model_servers --stop
"""
import argparse
from pathlib import Path

from xr_ai_launcher import (
    GPUInventoryError,
    Process,
    detect_gpu_config,
    load_deployment_profile,
    read_config_scalar,
    require_credentials,
    run_stack,
)
from xr_ai_launcher._config import _resolve_config_variant
from xr_ai_logging import setup_logging
from xr_ai_vllm import stop_persistent_servers

_BASE = Path(__file__).resolve().parent


def _gpu_profile_names() -> tuple[str, ...]:
    root = _BASE / "yaml"
    if not root.is_dir():
        return ()
    return tuple(path.name for path in sorted(root.iterdir()) if path.is_dir())


def _gpu_profile_name(value: str) -> str:
    names = _gpu_profile_names()
    if value not in names:
        available = ", ".join(names) or "none"
        raise argparse.ArgumentTypeError(
            f"unknown GPU profile {value!r}; available profiles: {available}"
        )
    return value

# service → (project, command, config basename). Order is launch
# order: NIM containers precede local servers (speech NIMs allocate fixed
# VRAM while LLM/VLM NIMs grab most of their GPU's free VRAM for KV cache);
# agent-llm precedes the VLM so its FlashInfer MoE JIT compilation runs with
# the full GPU free on single-GPU profiles.
_MODEL_SERVICES: dict[str, tuple[str, str, str]] = {
    "stt-nim":   ("../../services/nim-server", "nim_server", "nim_stt_server"),
    "tts-nim":   ("../../services/nim-server", "nim_server", "nim_tts_server"),
    "llm-nim":   ("../../services/nim-server", "nim_server", "nim_llm_server"),
    "vlm-nim":   ("../../services/nim-server", "nim_server", "nim_vlm_server"),
    "stt":       ("../../services/stt-server", "stt_server", "stt_server"),
    "tts":       ("../../services/piper-tts", "piper_tts_server", "piper_tts_server"),
    "agent-llm": (
        "../../services/nemotron3-nano-llm",
        "nemotron3_nano_llm_server",
        "nemotron3_nano_llm_server",
    ),
    "omni": (
        "../../services/nemotron-omni-llm",
        "nemotron_omni_llm_server",
        "nemotron_omni_llm_server",
    ),
    "vlm":       ("../../services/vlm-server", "vlm_server", "vlm_server"),
    "embedding": (
        "../../services/embedding-server",
        "embedding_server",
        "embedding_server",
    ),
}


def _profile_path(selection: str) -> Path:
    if "/" in selection or selection.endswith(".json"):
        return Path(selection)
    return _BASE / "yaml" / f"models.{selection}.json"


def _read_service_port(path: Path) -> int | None:
    """Read and validate the HTTP port used by one model-server config."""
    raw = read_config_scalar(path, "port") or read_config_scalar(path, "http_port")
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: port must be an integer, got {raw!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{path}: port must be between 1 and 65535, got {port}")
    return port


def _build_processes(
    selection: str, gpu_profile: str | None = None,
) -> tuple[list[Process], tuple[str, ...]]:
    profile_path = _profile_path(selection)
    deployment = load_deployment_profile(profile_path)
    unknown = deployment.services.keys() - _MODEL_SERVICES.keys()
    if unknown:
        raise ValueError(f"model profile declares unknown services: {sorted(unknown)}")

    profile_name = gpu_profile or detect_gpu_config()
    config_dir = _BASE / "yaml" / profile_name
    profile_key = profile_path.stem.removeprefix("models.")
    processes: list[Process] = []
    for service, (project, command, config_base) in _MODEL_SERVICES.items():
        if deployment.launch_mode(service) != "own":
            continue
        config = _resolve_config_variant(config_dir, config_base, profile_key)
        if not config.is_file():
            raise ValueError(
                f"GPU profile {profile_name!r} is incomplete: missing {config} "
                f"for service {service!r}"
            )
        port = _read_service_port(config)
        if port is None:
            raise ValueError(f"{config}: service config must declare port or http_port")
        processes.append(Process(
            service, project, command, config=config,
            launch_mode="persist", port=port,
        ))
    return processes, deployment.required_credentials


def _known_service_ports() -> list[tuple[str, int]]:
    """Discover cleanup targets from the YAML files that own their ports."""
    targets: set[tuple[str, int]] = set()
    for service, (project, _, config_base) in _MODEL_SERVICES.items():
        configs = list((_BASE / "yaml").glob(f"*/{config_base}*.yaml"))
        configs.append((_BASE / project / f"{config_base}.yaml").resolve())
        for config in configs:
            if config.is_file() and (port := _read_service_port(config)) is not None:
                targets.add((service, port))
    return sorted(targets)


def _stop_models() -> None:
    # Surface docker/ss/lsof failures so operators see why --stop aborted
    # instead of a silent traceback exit.
    try:
        if not stop_persistent_servers(_known_service_ports()):
            raise RuntimeError("one or more persistent servers are still running")
    except Exception as exc:
        raise SystemExit(
            f"model-servers: failed to stop persistent servers: {exc}"
        ) from exc


def _stop_unselected_services(processes: list[Process]) -> None:
    """Free capacity held by services outside the selected profile.

    Stops by port, keeping any port the profile uses: a persistent server
    already holding a selected port is reused (or evicted by the incoming
    wrapper when a different container owns it).
    """
    selected_ports = {process.port for process in processes}
    unselected = [
        (service, port)
        for service, port in _known_service_ports()
        if port not in selected_ports
    ]
    if not stop_persistent_servers(unselected):
        raise RuntimeError("could not stop persistent servers outside the profile")


def run() -> None:
    setup_logging("orchestrator", namespace="model-servers")

    p = argparse.ArgumentParser(add_help=False)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--stop", action="store_true",
        help="Stop every persisted model-server stack and exit.",
    )
    mode.add_argument(
        "--models", dest="models", metavar="NAME_OR_PATH",
        help="Deployment profile to start: a shipped name (default, "
             "vlm_llm_nim) or a path to a profile JSON.",
    )
    p.set_defaults(models="default")
    p.add_argument("--allow-anonymous", action="store_true",
                   help="Start without HF_TOKEN (unauthenticated downloads "
                        "of the multi-GB checkpoints may stall indefinitely).")
    p.add_argument(
        "--gpu-profile", metavar="NAME", type=_gpu_profile_name,
        help="Use a named YAML GPU profile instead of automatic detection. "
             "Intended for explicitly reviewed custom hardware profiles.",
    )
    ns, _ = p.parse_known_args()

    if ns.stop:
        _stop_models()
        return

    try:
        processes, credentials = _build_processes(ns.models, ns.gpu_profile)
    except GPUInventoryError as exc:
        p.error(
            f"{exc}\nUse --gpu-profile NAME to select an explicitly reviewed "
            "custom YAML profile."
        )
    except ValueError as exc:
        p.error(str(exc))

    # A missing HF_TOKEN silently stalls the multi-GB first-run download; see
    # docs/source/getting_started/credentials.md.
    require_credentials("HF_TOKEN", allow_missing=ns.allow_anonymous)
    for credential in credentials:
        require_credentials(credential)
    _stop_unselected_services(processes)
    run_stack(processes, _BASE, exit_after_ready=True)


if __name__ == "__main__":
    run()
