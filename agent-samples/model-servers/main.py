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
from pathlib import Path

from xr_ai_launcher import (
    Process,
    detect_gpu_config,
    load_deployment_profile,
    require_credentials,
    run_stack,
)
from xr_ai_launcher._config import _resolve_config_variant
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


def _build_processes(selection: str) -> tuple[list[Process], tuple[str, ...]]:
    profile_path = _profile_path(selection)
    deployment = load_deployment_profile(profile_path)
    unknown = deployment.services.keys() - _MODEL_SERVICES.keys()
    if unknown:
        raise ValueError(f"model profile declares unknown services: {sorted(unknown)}")

    config_dir = _BASE / "yaml" / detect_gpu_config()
    profile_key = profile_path.stem.removeprefix("models.")
    processes = [
        Process(
            service, project, command,
            config=_resolve_config_variant(config_dir, config_base, profile_key),
            launch_mode="persist", port=port,
        )
        for service, (project, command, config_base, port) in _MODEL_SERVICES.items()
        if deployment.launch_mode(service) == "own"
    ]
    return processes, deployment.required_credentials


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
    ns, _ = p.parse_known_args()

    if ns.stop:
        _stop_models()
        return

    processes, credentials = _build_processes(ns.models)

    # A missing HF_TOKEN silently stalls the multi-GB first-run download; see
    # docs/source/getting_started/credentials.md.
    require_credentials("HF_TOKEN", allow_missing=ns.allow_anonymous)
    for credential in credentials:
        require_credentials(credential)
    _stop_unselected_services(processes)
    run_stack(processes, _BASE, exit_after_ready=True)


if __name__ == "__main__":
    run()
