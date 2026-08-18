# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch native tea guidance with explicit voice and speech modes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from xr_ai_launcher import Process, ensure_credentials, load_model_deployment, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent
_WORKER_CONFIG = _BASE / "yaml/tea_making_worker.yaml"
_VOICE_CONFIGS = {
    "wake-word": _BASE / "yaml/voice_gate.yaml",
    "always-on": _BASE / "yaml/voice_gate.always-on.yaml",
}
_TTS_CONFIGS = {
    "piper": ("piper_tts", "http://localhost:8105"),
    "magpie": ("magpie_tts", "http://localhost:8104"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tea-making guidance with native XR agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""example:
  tea_making_sample --tts-mode piper

The LLM and VLM roles both use Nemotron-3-Nano-Omni on port 8108.
""",
    )
    parser.add_argument(
        "--voice-mode",
        default="wake-word",
        choices=("wake-word", "always-on"),
        help=("wake-word (default): require Agent/Hey Agent; always-on: accept every utterance"),
    )
    parser.add_argument(
        "--tts-mode",
        required=True,
        choices=("piper", "magpie"),
        help="piper: lightweight CPU speech; magpie: neural speech on CUDA when available",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace | None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not arguments:
        parser.print_help()
        return None
    return parser.parse_args(arguments)


def _tts_process(tts_mode: str) -> Process:
    if tts_mode == "piper":
        return Process(
            "tts",
            "../../services/piper-tts",
            "piper_tts_server",
            config="yaml/piper_tts_server.yaml",
        )
    if tts_mode == "magpie":
        return Process(
            "tts",
            "../../services/magpie-tts",
            "magpie_tts_server",
            config="yaml/magpie_tts_server.yaml",
        )
    raise ValueError(f"unknown TTS mode: {tts_mode!r}")


def _model_processes(tts_mode: str) -> dict[str, Process]:
    return {
        "stt": Process("stt", "../../services/stt-server", "stt_server"),
        "omni": Process(
            "omni",
            "../../services/nemotron-omni-llm",
            "nemotron_omni_llm_server",
        ),
        "embedding": Process(
            "embedding",
            "../../services/embedding-server",
            "embedding_server",
        ),
        "tts": _tts_process(tts_mode),
    }


def _build_processes(worker_config: Path, tts_mode: str) -> tuple[list[Process], tuple[str, ...]]:
    deployment = load_model_deployment(worker_config)
    available = _model_processes(tts_mode)
    unknown = deployment.services.keys() - available.keys()
    if unknown:
        raise ValueError(f"model profile declares unknown services: {sorted(unknown)}")

    processes = [
        Process(
            "hub",
            "../../services/xr-media-hub",
            "xr_media_hub",
            config="yaml/xr_media_hub.yaml",
        )
    ]
    for service, process in available.items():
        launch_mode = deployment.launch_mode(service)
        if launch_mode is not None:
            processes.append(replace(process, launch_mode=launch_mode))
    processes.extend(
        [
            Process(
                "rag",
                "../../services/rag-service",
                "rag_service",
                config="yaml/rag_service.yaml",
            ),
            Process(
                "worker",
                "worker",
                "tea_making_worker",
                config=worker_config,
            ),
        ]
    )
    return processes, deployment.required_credentials


def _write_config(source: Path, target: Path, overrides: Mapping[str, Path]) -> None:
    pending = set(overrides)
    lines: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        key = line.partition(":")[0]
        if key in pending and line.startswith(f"{key}:"):
            lines.append(f"{key}: {json.dumps(str(overrides[key]))}")
            pending.remove(key)
        else:
            lines.append(line)
    if pending:
        raise ValueError(f"{source} has no top-level fields: {sorted(pending)}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_models_config(target: Path, tts_mode: str) -> None:
    preset, base_url = _TTS_CONFIGS[tts_mode]
    source = _BASE / "yaml/models.local.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    tts = config["models"]["tts"]
    tts["adapter"] = {"preset": preset}
    tts["endpoint"]["base_url"] = base_url
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _materialize_worker_config(runtime_dir: Path, voice_mode: str, tts_mode: str) -> Path:
    models = runtime_dir / "models.json"
    _write_models_config(models, tts_mode)
    worker_config = runtime_dir / "tea_making_worker.yaml"
    _write_config(
        _WORKER_CONFIG,
        worker_config,
        {
            "models_config": models,
            "workflow_config": (_BASE / "yaml/workflow.yaml").resolve(),
            "voice_gate_yaml": _VOICE_CONFIGS[voice_mode].resolve(),
            "artifacts_dir": (_BASE / "artifacts").resolve(),
        },
    )
    return worker_config


def run(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args is None:
        return
    setup_logging("orchestrator", namespace="tea-making-sample")
    logging.getLogger(__name__).info(
        "launch selection voice_mode=%s tts_mode=%s",
        args.voice_mode,
        args.tts_mode,
    )
    with tempfile.TemporaryDirectory(prefix="tea-making-config-") as directory:
        worker_config = _materialize_worker_config(
            Path(directory),
            args.voice_mode,
            args.tts_mode,
        )
        processes, credentials = _build_processes(worker_config, args.tts_mode)
        for credential in credentials:
            ensure_credentials(credential)
        run_stack(processes, _BASE)


if __name__ == "__main__":
    run()
