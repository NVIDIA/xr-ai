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
    omni       — Nemotron-3-Nano-Omni-30B-A3B       port 8108  (vLLM; llm + agent_llm)
    vlm        — nvidia/Cosmos3-Nano Reasoner       port 8100  (vLLM)
    embedding  — nvidia/llama-nemotron-embed-1b-v2  port 8109  (vLLM)

  vlm_llm_nim
    stt + embedding local; the LLM and VLM as self-hosted NIM containers
    (Nemotron-3-Nano port 8110, Cosmos-Reason1-7B port 8100). Requires
    docker + NGC_API_KEY. Pairs with the samples' models.vlm_llm_nim.json.

  vlm_speech_nim
    Riva speech NIM containers (gRPC 50051/50052) + Cosmos NIM + local
    embedding. Pairs with simple-vlm-example's models.vlm_speech_nim.json.
    Mutually exclusive with vlm_llm_nim on 2x48 GB.

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
import urllib.request
from dataclasses import replace
from pathlib import Path

from xr_ai_launcher import (
    VRAM_UTILIZATION_ENV,
    Process,
    detect_gpu_config,
    format_vram_preflight,
    load_deployment_profile,
    load_vram_profile,
    preflight_vram,
    query_gpu_inventory,
    require_credentials,
    require_vram_preflight,
    run_stack,
    utilization_overrides,
    validate_vram_certification,
)
from xr_ai_logging import setup_logging
from xr_ai_vllm import stop_persistent_servers

_BASE = Path(__file__).resolve().parent

# service → (project, command, config basename, port). Order is launch
# order: NIM containers precede local servers (speech NIMs allocate fixed
# VRAM while LLM/VLM NIMs grab most of their GPU's free VRAM for KV cache);
# agent-llm precedes the VLM so its FlashInfer MoE JIT compilation runs with
# the full GPU free on single-GPU profiles.
_MODEL_SERVICES: dict[str, tuple[str, str, str, int]] = {
    "stt-nim":   ("../../services/nim-server", "nim_server", "nim_stt_server", 9010),
    "tts-nim":   ("../../services/nim-server", "nim_server", "nim_tts_server", 9011),
    "llm-nim":   ("../../services/nim-server", "nim_server", "nim_llm_server", 8110),
    "vlm-nim":   ("../../services/nim-server", "nim_server", "nim_vlm_server", 8100),
    "stt":       ("../../services/stt-server", "stt_server", "stt_server", 8103),
    "agent-llm": (
        "../../services/nemotron3-nano-llm",
        "nemotron3_nano_llm_server",
        "nemotron3_nano_llm_server",
        8107,
    ),
    "omni": (
        "../../services/nemotron-omni-llm",
        "nemotron_omni_llm_server",
        "nemotron_omni_llm_server",
        8108,
    ),
    "vlm":       ("../../services/vlm-server", "vlm_server", "vlm_server", 8100),
    "embedding": (
        "../../services/embedding-server",
        "embedding_server",
        "embedding_server",
        8109,
    ),
}


def _profile_path(selection: str) -> Path:
    if "/" in selection or selection.endswith(".json"):
        return Path(selection)
    return _BASE / "yaml" / f"models.{selection}.json"


def _service_config(gpu_dir: str, config_base: str, profile_key: str) -> str:
    """Per-profile config variant when present, the service default otherwise."""
    variant = f"{gpu_dir}/{config_base}_{profile_key}.yaml"
    if (_BASE / variant).exists():
        return variant
    return f"{gpu_dir}/{config_base}.yaml"


def _build_processes(
    selection: str, gpu_profile: str | None = None,
) -> tuple[list[Process], tuple[str, ...]]:
    profile_path = _profile_path(selection)
    deployment = load_deployment_profile(profile_path)
    unknown = deployment.services.keys() - _MODEL_SERVICES.keys()
    if unknown:
        raise ValueError(f"model profile declares unknown services: {sorted(unknown)}")

    gpu_dir = f"yaml/{gpu_profile or detect_gpu_config()}"
    profile_key = profile_path.stem.removeprefix("models.")
    processes = [
        Process(
            service, project, command,
            config=_service_config(gpu_dir, config_base, profile_key),
            launch_mode="persist", port=port,
        )
        for service, (project, command, config_base, port) in _MODEL_SERVICES.items()
        if deployment.launch_mode(service) == "own"
    ]
    return processes, deployment.required_credentials


def _vram_profile_path(gpu_profile: str, selection: str) -> Path:
    profile_key = _profile_path(selection).stem.removeprefix("models.")
    return _BASE / "yaml" / gpu_profile / f"vram.{profile_key}.json"


def _service_is_ready(port: int) -> bool:
    for path in ("/health", "/v1/health/ready"):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=0.5,
            ) as response:
                if response.status == 200:
                    return True
        except Exception:
            continue
    return False


def _apply_vram_plan(
    processes: list[Process], *, selection: str, gpu_profile: str,
) -> list[Process]:
    """Preflight the complete stack and inject derived vLLM utilization."""
    profile = load_vram_profile(_vram_profile_path(gpu_profile, selection))
    selected = {process.name for process in processes}
    reserved = {item.service for item in profile.services}
    if selected != reserved:
        raise ValueError(
            f"VRAM profile services do not match deployment: "
            f"missing={sorted(selected - reserved)}, extra={sorted(reserved - selected)}"
        )
    validate_vram_certification(profile, {
        process.name: (_BASE / process.config).resolve()
        for process in processes if process.config is not None
    })

    inventory = query_gpu_inventory()
    active = frozenset(
        process.name for process in processes
        if process.port is not None and _service_is_ready(process.port)
    )
    results = preflight_vram(profile, inventory, active_services=active)
    print(format_vram_preflight(profile, results), flush=True)
    require_vram_preflight(results)

    overrides = utilization_overrides(profile, inventory)
    return [
        replace(
            process,
            environment=process.environment + (
                (VRAM_UTILIZATION_ENV, overrides[process.name]),
            ),
        )
        if process.name in overrides else process
        for process in processes
    ]


def _stop_models() -> None:
    # Surface docker/ss/lsof failures so operators see why --stop aborted
    # instead of a silent traceback exit.
    try:
        stop_persistent_servers([
            (service, port)
            for service, (_, _, _, port) in _MODEL_SERVICES.items()
        ])
    except Exception as exc:
        print(f"model-servers: failed to stop persistent servers: {exc}", flush=True)


def _stop_unselected_services(processes: list[Process]) -> None:
    """Free capacity held by services outside the selected profile.

    Stops by port, keeping any port the profile uses: a persistent server
    already holding a selected port is reused (or evicted by the incoming
    wrapper when a different container owns it).
    """
    selected_ports = {process.port for process in processes}
    unselected = [
        (service, port)
        for service, (_, _, _, port) in _MODEL_SERVICES.items()
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
             "vlm_llm_nim, vlm_speech_nim) or a path to a profile JSON.",
    )
    p.set_defaults(models="default")
    p.add_argument("--allow-anonymous", action="store_true",
                   help="Start without HF_TOKEN (unauthenticated downloads "
                        "of the multi-GB checkpoints may stall indefinitely).")
    p.add_argument(
        "--gpu-profile", choices=("dual_48G_ada", "96G_blackwell", "spark"),
        help="Explicit hardware profile. By default XR-AI requires an exact "
             "per-GPU topology match; this override is for reviewed custom hosts.",
    )
    ns, _ = p.parse_known_args()

    if ns.stop:
        _stop_models()
        return

    gpu_profile = ns.gpu_profile or detect_gpu_config()
    processes, credentials = _build_processes(ns.models, gpu_profile)

    # A missing HF_TOKEN silently stalls the multi-GB first-run download; see
    # docs/source/getting_started/credentials.md.
    require_credentials("HF_TOKEN", allow_missing=ns.allow_anonymous)
    for credential in credentials:
        require_credentials(credential)
    _stop_unselected_services(processes)
    processes = _apply_vram_plan(
        processes, selection=ns.models, gpu_profile=gpu_profile,
    )
    run_stack(processes, _BASE, exit_after_ready=True)


if __name__ == "__main__":
    run()
