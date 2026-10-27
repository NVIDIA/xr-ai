# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-local speaker defaults, explicit batch selection and scoped stop."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from xr_ai_launcher import _ownership
from xr_ai_model_server_stack import _stack, run_model_stack
from xr_ai_models import load_models_config
from xr_ai_vllm import _docker
from xr_ai_voicegate import load_voice_gate_config

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_stack_logging(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "orchestrator.log"
    log_file.parent.mkdir()
    monkeypatch.setattr(_stack, "setup_logging", lambda *_a, **_kw: log_file)


_SAMPLES = {
    "simple-vlm-example": ("speaker-stt", "tts", "vlm"),
    "conversation-example": ("speaker-stt", "tts", "omni", "vlm", "clef"),
    "lab-instrument-monitoring": ("speaker-stt", "tts", "omni", "vlm"),
    "tea-making-sample": ("speaker-stt", "tts", "omni", "embedding"),
    "xr-render-demo": ("speaker-stt", "tts", "omni", "vlm"),
}


@pytest.mark.parametrize("hardware", ["dual_48G_ada", "96G_blackwell", "spark"])
@pytest.mark.parametrize("sample, services", _SAMPLES.items())
def test_sample_stack_matches_required_worker_speech(sample, services, hardware, tmp_path):
    base = _ROOT / "agent-samples" / sample
    configs = _stack._configs(base / "model-servers", services, hardware)
    processes = _stack._processes(configs, tmp_path, _ROOT)
    speaker = next(p for p in processes if p.name == "speaker-stt")
    assert speaker.port == 8102 and speaker.launch_mode == "persist"
    assert speaker.command == "speaker_stt"
    assert "stt" not in configs
    cfg = yaml.safe_load(Path(speaker.config).read_text())
    assert cfg["host"] == "127.0.0.1"
    assert cfg["cuda_visible_devices"] == ("1" if hardware == "dual_48G_ada" else "0")
    assert Path(cfg["model_cache"]) == _ROOT / "models"
    model_file = "models.local.json" if sample == "tea-making-sample" else "models.json"
    assert "stt" not in load_models_config(base / "yaml" / model_file).entries
    for gate_file in (base / "yaml").glob("voice_gate*.yaml"):
        gate = load_voice_gate_config(gate_file)
        assert gate._speaker.backend == "required"
        assert gate._speaker.base_url == f"http://{cfg['host']}:{cfg['port']}"


def test_explicit_batch_selection_remains_independent(tmp_path, monkeypatch):
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml/custom.yaml").write_text("stt: {}\n")
    configs = _stack._configs(tmp_path, ("stt",), "custom")
    assert configs["stt"]["port"] == 8103
    assert "speaker-stt" not in configs
    stopped = []
    monkeypatch.setattr(_stack, "stop_persistent_servers", lambda targets: stopped.extend(targets) or True)
    run_model_stack(tmp_path, ("stt",), stop=True, gpu_profile="custom")
    assert stopped == [("stt", 8103)]


def test_speaker_stop_uses_selected_configured_port(tmp_path, monkeypatch):
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml/custom.yaml").write_text("speaker-stt: {port: 12345}\n")
    stopped = []
    monkeypatch.setattr(_stack, "stop_persistent_servers", lambda targets: stopped.extend(targets) or True)
    run_model_stack(tmp_path, ("speaker-stt",), stop=True, gpu_profile="custom")
    assert stopped == [("speaker-stt", 12345)]


@pytest.mark.parametrize("environment, owned", [
    (b"XR_AI_VLLM_MANAGED=1\0XR_AI_VLLM_PORT=8102\0", True),
    (b"XR_AI_VLLM_MANAGED=1\0XR_AI_VLLM_PORT=8103\0", False),
    (b"PATH=/bin\0", False),
])
def test_speaker_cleanup_requires_matching_managed_listener(tmp_path, monkeypatch, environment, owned):
    proc = tmp_path / "proc" / "1234"
    proc.mkdir(parents=True)
    (proc / "cmdline").write_text("python\0-m\0speaker_stt\0--_serve")
    (proc / "environ").write_bytes(environment)
    monkeypatch.setattr(_docker, "Path", lambda path: proc / path.rsplit("/", 1)[-1])
    monkeypatch.setattr(_ownership, "Path", lambda path: proc / path.rsplit("/", 1)[-1])
    assert _docker.is_xr_ai_server_process(1234, "speaker-stt", 8102) is owned
