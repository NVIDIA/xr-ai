# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU model selection and Docker placement through real service entry points."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("service", ["nemotron3-nano-llm", "nemotron-omni-llm"])
@pytest.mark.parametrize(
    ("inherited", "configured", "device", "model"),
    [
        ("1", None, "1", "test/nvfp4"),
        ("0", None, "0", "test/fp8"),
        ("GPU-blackwell", None, "GPU-blackwell", "test/nvfp4"),
        ("1", 0, "0", "test/fp8"),
        ("0", "1", "1", "test/nvfp4"),
        (None, None, "all", "test/fp8"),
    ],
)
def test_nemotron_model_and_docker_use_selected_gpu(
    monkeypatch, tmp_path, service, inherited, configured, device, model,
):
    module_name = service.replace("-llm", "_llm_server").replace("-", "_")
    path = _REPO_ROOT / "services" / service / module_name / "__main__.py"
    spec = importlib.util.spec_from_file_location("_test_gpu_selection_server", path)
    assert spec is not None and spec.loader is not None
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    cfg = {
        "vllm_backend": "docker", "spark_uma": False,
        "model_blackwell": "test/nvfp4", "model_ada": "test/fp8",
    }
    if configured is not None:
        cfg["cuda_visible_devices"] = configured
    if inherited is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", inherited)
    for key in ("HF_HOME", "TRANSFORMERS_CACHE"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "nano_v3_reasoning_parser.py").write_text("# Cached parser fixture.\n")
    captured = {}

    def query(argv, **_kwargs):
        if "--id" not in argv:
            return "8.9\n12.0\n"
        selected = argv[argv.index("--id") + 1]
        return {"0": "8.9\n", "1": "12.0\n", "GPU-blackwell": "12.0\n"}[selected]

    monkeypatch.setattr(server, "setup_logging", lambda _name: None)
    monkeypatch.setattr(server, "load_config", lambda: (cfg, path.parent, None))
    monkeypatch.setattr(server, "resolve_model_cache", lambda *_a, **_kw: tmp_path)
    monkeypatch.setattr("xr_ai_vllm._config.subprocess.check_output", query)
    monkeypatch.setattr("xr_ai_vllm._docker.run_container", lambda **kwargs: captured.update(kwargs))

    server.run()

    assert captured["diagnostic_argv"][2] == model
    assert f"NVIDIA_VISIBLE_DEVICES={device}" in captured["argv"]
