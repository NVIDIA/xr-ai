# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the tea guide with explicit model, voice, and speech modes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from xr_ai_launcher import Process, detect_gpu_config, load_model_deployment, run_stack, warn_if_missing
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent
_MODEL_CONFIGS = {
    "omni": "models.omni.json",
    "cosmos": "models.cosmos.json",
}
_VOICE_CONFIGS = {
    "wake-word": "voice_gate.wake-word.yaml",
    "always-on": "voice_gate.always-on.yaml",
}
_TTS_CONFIGS = {
    "piper": ("piper_tts", "http://localhost:8105"),
    "magpie": ("magpie_tts", "http://localhost:8104"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tea-making guidance sample. All launch modes must be selected explicitly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  tea_making_sample --model-mode omni --voice-mode wake-word --tts-mode piper
  tea_making_sample --model-mode cosmos --voice-mode always-on --tts-mode magpie

shared model servers:
  model_servers""",
    )
    parser.add_argument(
        "--model-mode",
        required=True,
        choices=tuple(_MODEL_CONFIGS),
        help="omni: Omni for vision and agents; cosmos: Cosmos vision with Omni agents",
    )
    parser.add_argument(
        "--voice-mode",
        required=True,
        choices=tuple(_VOICE_CONFIGS),
        help="wake-word: require Agent/Hey Agent; always-on: accept every utterance",
    )
    parser.add_argument(
        "--tts-mode",
        required=True,
        choices=tuple(_TTS_CONFIGS),
        help="piper: lightweight CPU speech; magpie: NeMo speech on CUDA when available",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace | None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not arguments:
        parser.print_help()
        return None
    return parser.parse_args(arguments)


def _model_processes(tts_mode: str) -> dict[str, Process]:
    detected = detect_gpu_config()
    profile = {"spark": "96G_blackwell"}.get(detected, detected)
    if profile not in {"96G_blackwell", "dual_48G_ada"}:
        raise RuntimeError(f"unsupported local model profile: {profile}")
    return {
        "vlm": Process("vlm", "../../ai-services/vlm-server", "vlm_server"),
        "embedding": Process(
            "embedding",
            "../../ai-services/embedding-server",
            "embedding_server",
            config=f"yaml/{profile}/embedding_server.yaml",
        ),
        "stt": Process(
            "stt",
            "../../ai-services/stt-server",
            "stt_server",
            config=f"yaml/{profile}/stt_server.yaml",
        ),
        "omni": Process(
            "omni",
            "../../ai-services/llm/nemotron_omni",
            "nemotron_omni_llm_server",
            config=f"yaml/{profile}/nemotron_omni_llm_server.yaml",
        ),
        "tts": _tts_process(tts_mode),
    }


def _tts_process(tts_mode: str) -> Process:
    if tts_mode == "piper":
        return Process(
            "tts",
            "../../ai-services/tts/piper",
            "piper_tts_server",
            config="yaml/piper_tts_server.yaml",
        )
    if tts_mode == "magpie":
        return Process(
            "tts",
            "../../ai-services/tts/magpie",
            "magpie_tts_server",
            config="yaml/magpie_tts_server.yaml",
        )
    raise ValueError(f"unknown TTS mode: {tts_mode!r}")


def _build_processes(worker_config: Path, rag_config: Path, tts_mode: str) -> list[Process]:
    deployment = load_model_deployment(worker_config)
    available = _model_processes(tts_mode)
    unknown = deployment.services.keys() - available.keys()
    if unknown:
        raise ValueError(f"model profile declares unknown services: {sorted(unknown)}")
    processes = [replace(available[name], launch_mode=mode) for name, mode in deployment.services.items()]
    processes.extend(
        [
            Process("rag", "../../services/rag-service", "rag_service", config=rag_config),
            Process("hub", "../../server-runtime", "xr_media_hub", config="yaml/xr_media_hub.yaml"),
            Process(
                "activity-viewer",
                ".",
                "tea_making_activity_viewer",
                config="yaml/activity_viewer.json",
            ),
            Process("worker", "worker", "tea_making_worker", config=worker_config),
        ]
    )
    return processes


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


def _materialize_configs(
    runtime_dir: Path,
    model_mode: str,
    voice_mode: str,
    tts_mode: str,
) -> tuple[Path, Path]:
    yaml_dir = _BASE / "yaml"
    models = runtime_dir / "models.json"
    _write_models_config(yaml_dir / _MODEL_CONFIGS[model_mode], models, tts_mode)
    voice_gate = (yaml_dir / _VOICE_CONFIGS[voice_mode]).resolve()
    worker_config = runtime_dir / "tea_making_worker.yaml"
    rag_config = runtime_dir / "rag_service.yaml"
    _write_config(
        yaml_dir / "tea_making_worker.yaml",
        worker_config,
        {
            "models_config": models,
            "workflow_config": (yaml_dir / "workflow.yaml").resolve(),
            "applications_config": (yaml_dir / "applications.yaml").resolve(),
            "voice_gate_config": voice_gate,
        },
    )
    _write_config(
        yaml_dir / "rag_service.yaml",
        rag_config,
        {
            "models_config": models,
            "documents_dir": (_BASE / "rag-documents").resolve(),
            "cache_dir": (_BASE.parents[1] / "models" / "tea-making-rag-cache").resolve(),
        },
    )
    return worker_config, rag_config


def _write_models_config(source: Path, target: Path, tts_mode: str) -> None:
    preset, base_url = _TTS_CONFIGS[tts_mode]
    config = json.loads(source.read_text(encoding="utf-8"))
    tts = config["models"]["tts"]
    tts["adapter"] = {"preset": preset}
    tts["endpoint"]["base_url"] = base_url
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args is None:
        return
    setup_logging("orchestrator", namespace="tea-making-sample")
    warn_if_missing("HF_TOKEN")
    logging.getLogger(__name__).info(
        "launch selection model_mode=%s voice_mode=%s tts_mode=%s",
        args.model_mode,
        args.voice_mode,
        args.tts_mode,
    )
    with tempfile.TemporaryDirectory(prefix="tea-making-config-") as directory:
        worker_config, rag_config = _materialize_configs(
            Path(directory),
            args.model_mode,
            args.voice_mode,
            args.tts_mode,
        )
        run_stack(_build_processes(worker_config, rag_config, args.tts_mode), _BASE)


if __name__ == "__main__":
    run()
