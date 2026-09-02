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
    assert "--gpu-memory-utilization" in args
    assert "--kv-cache-memory-bytes" not in args


def test_spark_profile_uses_explicit_kv_cache(monkeypatch, tmp_path) -> None:
    server = _load_server_module()
    spark_config = _MODEL_PROFILES / "spark" / "vlm_server.yaml"
    cfg = yaml.safe_load(spark_config.read_text())
    captured = {}

    monkeypatch.setattr(server, "setup_logging", lambda _name: None)
    monkeypatch.setattr(
        server,
        "load_config",
        lambda: (cfg, spark_config.parent, None),
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
    cache_index = args.index("--kv-cache-memory-bytes")
    assert args[cache_index + 1] == "1610612736"
    memory_index = args.index("--gpu-memory-utilization")
    assert args[memory_index + 1] == "0.2"
    assert captured["spark_uma"] is True


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


def test_hardware_profiles_reserve_measured_reasoner_memory() -> None:
    blackwell = yaml.safe_load(
        (_MODEL_PROFILES / "96G_blackwell" / "vlm_server.yaml").read_text()
    )
    dual_ada = yaml.safe_load(
        (_MODEL_PROFILES / "dual_48G_ada" / "vlm_server.yaml").read_text()
    )
    spark = yaml.safe_load(
        (_MODEL_PROFILES / "spark" / "vlm_server.yaml").read_text()
    )

    assert blackwell["gpu_memory_utilization"] == 0.23
    assert dual_ada["gpu_memory_utilization"] == 0.47
    assert "kv_cache_memory_bytes" not in blackwell
    assert "kv_cache_memory_bytes" not in dual_ada
    assert "spark_uma" not in blackwell
    assert "spark_uma" not in dual_ada
    assert blackwell["max_num_seqs"] == 4
    assert "max_num_seqs" not in dual_ada
    assert spark["kv_cache_memory_bytes"] == 1610612736
    assert spark["gpu_memory_utilization"] == 0.20
    assert spark["max_num_seqs"] == 1
    assert spark["spark_uma"] is True


@pytest.mark.parametrize("value", [True, 0, -1, "invalid"])
def test_vlm_rejects_invalid_explicit_kv_cache(
    monkeypatch, value: object,
) -> None:
    server = _load_server_module()
    cfg = yaml.safe_load(_SERVER_YAML.read_text())
    cfg["kv_cache_memory_bytes"] = value

    monkeypatch.setattr(server, "setup_logging", lambda _name: None)
    monkeypatch.setattr(
        server,
        "load_config",
        lambda: (cfg, _SERVER_YAML.parent, None),
    )

    with pytest.raises(SystemExit, match="1"):
        server.run()


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


def test_sample_model_configs_select_cosmos3_reasoner() -> None:
    simple = _SAMPLES / "simple-vlm-example" / "yaml"
    simple_models = json.loads((simple / "models.json").read_text())
    assert simple_models["models"]["vlm"]["adapter"]["preset"] == (
        "cosmos3_nano_reasoner"
    )

    render = _SAMPLES / "xr-render-demo" / "yaml"
    render_models = json.loads((render / "models.json").read_text())
    assert render_models["models"]["vlm"]["adapter"]["preset"] == (
        "cosmos3_nano_reasoner"
    )
