# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM server configuration coverage."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER_MAIN = _REPO_ROOT / "services" / "vlm-server" / "vlm_server" / "__main__.py"
_SERVER_YAML = _REPO_ROOT / "services" / "vlm-server" / "vlm_server.yaml"
_SAMPLES = _REPO_ROOT / "agent-samples"
_MODEL_PROFILES = _SAMPLES / "model-servers" / "yaml"
_LOCAL_VLM_CONFIGS = (
    _SERVER_YAML,
    _SAMPLES / "simple-vlm-example" / "yaml" / "vlm_server.yaml",
    _MODEL_PROFILES / "96G_blackwell" / "vlm_server.yaml",
    _MODEL_PROFILES / "dual_48G_ada" / "vlm_server.yaml",
    _MODEL_PROFILES / "spark" / "vlm_server.yaml",
)


def _load_server_module():
    spec = importlib.util.spec_from_file_location("_test_vlm_server_main", _SERVER_MAIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_cosmos3_service_uses_reasoner_only_path(monkeypatch, tmp_path) -> None:
    server = _load_server_module()
    cfg = yaml.safe_load(_SERVER_YAML.read_text())
    captured = {}

    monkeypatch.setattr(server, "setup_logging", lambda _name: None)
    monkeypatch.setattr(
        server,
        "load_config",
        lambda: (cfg, _SERVER_YAML.parent, None),
    )
    monkeypatch.setattr(
        server,
        "resolve_model_cache",
        lambda *_args, **_kwargs: tmp_path,
    )
    monkeypatch.setattr(server, "setup_hf_env", lambda *_args: None)
    monkeypatch.setattr(server, "serve", lambda **kwargs: captured.update(kwargs))

    server.run()

    args = captured["extra_serve_args"]
    assert captured["model"] == "nvidia/Cosmos3-Nano"
    assert captured["image"] == "nvcr.io/nvidia/vllm:26.07-py3"
    assert "--async-scheduling" in args
    overrides_index = args.index("--hf-overrides")
    assert json.loads(args[overrides_index + 1]) == {
        "architectures": ["Cosmos3ForConditionalGeneration"],
    }
    encoder_mode_index = args.index("--mm-encoder-tp-mode")
    assert args[encoder_mode_index + 1] == "data"
    assert "--omni" not in args
    assert "--model-class-name" not in args
    assert "Cosmos3OmniDiffusersPipeline" not in args


def test_all_local_profiles_select_cosmos3_reasoner_runtime() -> None:
    for config_path in _LOCAL_VLM_CONFIGS:
        cfg = yaml.safe_load(config_path.read_text())
        assert cfg["model"] == "nvidia/Cosmos3-Nano", config_path
        assert cfg["vllm_image"] == "nvcr.io/nvidia/vllm:26.07-py3", config_path
        assert cfg["async_scheduling"] is True, config_path
        assert cfg["hf_overrides"] == {
            "architectures": ["Cosmos3ForConditionalGeneration"],
        }, config_path
        assert cfg["mm_encoder_tp_mode"] == "data", config_path


def test_hardware_profiles_do_not_hardcode_vllm_memory_percentages() -> None:
    for hardware in ("96G_blackwell", "dual_48G_ada", "spark"):
        for config_name in (
            "embedding_server.yaml", "nemotron_omni_llm_server.yaml", "vlm_server.yaml",
        ):
            config = yaml.safe_load(
                (_MODEL_PROFILES / hardware / config_name).read_text()
            )
            assert "gpu_memory_utilization" not in config


def test_cosmos3_rejects_missing_reasoner_override(monkeypatch, tmp_path) -> None:
    server = _load_server_module()
    cfg = yaml.safe_load(_SERVER_YAML.read_text())
    del cfg["hf_overrides"]

    monkeypatch.setattr(server, "setup_logging", lambda _name: None)
    monkeypatch.setattr(
        server,
        "load_config",
        lambda: (cfg, _SERVER_YAML.parent, None),
    )
    monkeypatch.setattr(
        server,
        "resolve_model_cache",
        lambda *_args, **_kwargs: tmp_path,
    )
    monkeypatch.setattr(server, "setup_hf_env", lambda *_args: None)

    with pytest.raises(SystemExit, match="1"):
        server.run()


def test_sample_model_profiles_select_cosmos3_reasoner() -> None:
    simple = _SAMPLES / "simple-vlm-example" / "yaml"
    simple_local = json.loads((simple / "models.local.json").read_text())
    simple_hosted = json.loads((simple / "models.hosted.json").read_text())
    assert simple_local["models"]["vlm"]["adapter"]["preset"] == (
        "cosmos3_nano_reasoner"
    )
    assert simple_hosted["models"]["vlm"]["adapter"]["model_name"] == (
        "nvidia/cosmos3-nano-reasoner"
    )

    render = _SAMPLES / "xr-render-demo" / "yaml"
    render_local = json.loads((render / "models.local.json").read_text())
    render_hosted = json.loads((render / "models.hosted.json").read_text())
    assert render_local["models"]["vlm"]["adapter"]["preset"] == (
        "cosmos3_nano_reasoner"
    )
    assert render_hosted["models"]["vlm"]["adapter"]["model_name"] == (
        "nvidia/cosmos3-nano-reasoner"
    )
