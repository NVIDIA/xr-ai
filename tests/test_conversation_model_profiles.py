# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile selection and real model-adapter boundaries without GPU inference."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import httpx
import pytest
import yaml
from xr_ai_models import ChatMessage, DecisionQuestion, load_models_config, make_decision, make_llm, make_vlm

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "agent-samples/conversation-example"
_BASE = _SAMPLE / "model-servers"
_PROFILES = _SAMPLE / "yaml/model_profiles"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stack():
    return _module("_test_conversation_profile_stack", _BASE / "_profile_stack.py")


@pytest.mark.parametrize("gemma", ["e2b", "e4b"])
@pytest.mark.parametrize("d3", ["nano", "mini", "flash"])
@pytest.mark.parametrize("gpu", ["dual_48G_ada", "96G_blackwell", "spark"])
def test_profiles_share_one_gemma_server_and_select_checkpoints(stack, gemma, d3, gpu):
    profile = _PROFILES / f"gemma4-{gemma}-d3-{d3}.yaml"
    configs = stack._configs(_BASE, profile, gpu)
    models = load_models_config(profile)
    assert set(configs) == {"speaker-stt", "tts", "gemma4", "d3"}
    assert configs["gemma4"]["model"] == f"google/gemma-4-{gemma.upper()}-it"
    assert configs["d3"]["model"] == f"vllm-sr/d3-{d3}"
    assert models.llm("llm").endpoint == models.vlm("vlm").endpoint
    assert models.llm("llm").adapter.thinking.default is True
    assert models.vlm("vlm").adapter.capabilities["vision"] is True
    assert models.decision("router").kind == "systemone"
    assert all(Path(cfg["model_cache"]) == (_ROOT / "models").resolve() for cfg in configs.values())
    assert configs["d3"]["cuda_visible_devices"] == ("1" if gpu == "dual_48G_ada" else "0")


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "repo"
    (root / "services").parent.mkdir(parents=True)
    (root / "services").symlink_to(_ROOT / "services", target_is_directory=True)
    sample = root / "agent-samples/conversation-example"
    shutil.copytree(_SAMPLE / "yaml", sample / "yaml")
    shutil.copytree(_BASE / "yaml", sample / "model-servers/yaml")
    worker = sample / "yaml/conversation_example_worker.yaml"
    worker.write_text("models_config: model_profiles/gemma4-e2b-d3-nano.yaml\n")
    return sample


def test_worker_and_launcher_resolve_same_profile(sample, stack):
    worker = _module("_test_conversation_worker_config", _SAMPLE / "worker/conversation_example_worker/config.py")
    config = worker.load_config(sample / "yaml/conversation_example_worker.yaml")
    assert config.models_config == sample / "yaml/model_profiles/gemma4-e2b-d3-nano.yaml"
    configs = stack._configs(sample / "model-servers", config.models_config, "spark")
    assert configs["gemma4"]["model"] == "google/gemma-4-E2B-it"
    assert configs["d3"]["model"] == "vllm-sr/d3-nano"
    (sample / "yaml/conversation_example_worker.yaml").write_text("{}\n")
    default = worker.load_config(sample / "yaml/conversation_example_worker.yaml")
    assert default.models_config == sample / "yaml/models.json"
    assert stack._run_selected_profile(sample / "model-servers", stop=False, gpu_profile=None) is False


@pytest.mark.parametrize("selected", [None, "", [], 42])
def test_invalid_selection_is_rejected_by_worker_and_launcher(sample, stack, selected):
    worker = _module("_test_conversation_worker_config", _SAMPLE / "worker/conversation_example_worker/config.py")
    path = sample / "yaml/conversation_example_worker.yaml"
    path.write_text(yaml.safe_dump({"models_config": selected}))
    with pytest.raises(ValueError, match="models_config must be a non-empty path"):
        worker.load_config(path)
    with pytest.raises(ValueError, match="models_config must be a non-empty path"):
        stack._run_selected_profile(sample / "model-servers", stop=True, gpu_profile=None)


@pytest.mark.parametrize("stop", [False, True])
def test_selected_profile_launch_and_scoped_stop(sample, stack, monkeypatch, tmp_path, stop):
    logfile = tmp_path / "orchestrator.log"
    monkeypatch.setattr(stack, "setup_logging", lambda *_a, **_kw: logfile)
    monkeypatch.setattr(stack, "detect_gpu_config", lambda: pytest.fail("GPU detection must not run"))
    credentials = []
    monkeypatch.setattr(stack, "require_credentials", credentials.append)
    stopped = []
    monkeypatch.setattr(stack, "stop_persistent_servers", lambda ports: stopped.extend(ports) or True)
    launched = []

    def run(processes, base, **kwargs):
        assert base == sample / "model-servers" and kwargs == {"exit_after_ready": True}
        for process in processes:
            config = yaml.safe_load(Path(process.config).read_text())
            assert Path(process.config).stat().st_mode & 0o777 == 0o600
            assert process.launch_mode == "persist" and process.port == config["port"]
            assert Path(process.project).is_dir()
            launched.append((process.name, process.config, config))

    monkeypatch.setattr(stack, "run_stack", run)
    assert stack._run_selected_profile(sample / "model-servers", stop=stop,
                                       gpu_profile=None if stop else "spark")
    if stop:
        assert stopped == [("speaker-stt", 8102), ("tts", 8105), ("gemma4", 8108), ("d3", 8120)]
        assert credentials == [] and launched == []
    else:
        assert credentials == ["HF_TOKEN"] and stopped == []
        assert [name for name, _, _ in launched] == ["speaker-stt", "tts", "gemma4", "d3"]
        assert all(not Path(path).exists() for _, path, _ in launched)
        assert launched[-1][2]["model"] == "vllm-sr/d3-nano"


def test_profile_overrides_hardware_and_resolves_cache_at_source(sample, stack):
    profile = sample / "yaml/model_profiles/gemma4-e2b-d3-nano.yaml"
    raw = yaml.safe_load(profile.read_text())
    raw["servers"]["gemma4"].update(model="google/gemma-4-31B-it", model_cache="weights", max_num_seqs=2)
    profile.write_text(yaml.safe_dump(raw))
    configs = stack._configs(sample / "model-servers", profile, "spark")
    assert configs["gemma4"]["model"] == "google/gemma-4-31B-it"
    assert configs["gemma4"]["max_num_seqs"] == 2
    assert Path(configs["gemma4"]["model_cache"]) == profile.parent / "weights"


@pytest.mark.parametrize("error", ["missing", "port", "alias", "collision", "gpu"])
def test_invalid_profile_rejected_before_launch(sample, stack, error):
    profile = sample / "yaml/model_profiles/gemma4-e2b-d3-nano.yaml"
    raw = yaml.safe_load(profile.read_text())
    if error == "missing":
        del raw["servers"]["d3"]
    elif error == "port":
        raw["servers"]["d3"]["port"] = 8121
    elif error == "alias":
        raw["servers"]["d3"]["served_model_name"] = "other"
    elif error == "collision":
        raw["servers"]["d3"]["port"] = 8108
    profile.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        stack._configs(sample / "model-servers", profile, "unknown" if error == "gpu" else "spark")


def test_failed_start_retains_private_configs_and_redacts_credentials(sample, stack, monkeypatch, tmp_path):
    profile = sample / "yaml/model_profiles/gemma4-e2b-d3-nano.yaml"
    raw = yaml.safe_load(profile.read_text())
    raw["servers"]["gemma4"]["hf_token"] = "inline-test-credential"
    profile.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(stack, "setup_logging", lambda *_a, **_kw: tmp_path / "orchestrator.log")
    monkeypatch.setattr(stack, "require_credentials", lambda *_a: None)
    paths = []

    def fail(processes, *_a, **_kw):
        paths.extend(Path(p.config) for p in processes)
        raise RuntimeError("readiness failed")

    monkeypatch.setattr(stack, "run_stack", fail)
    with pytest.raises(RuntimeError, match="readiness failed"):
        stack._run_selected_profile(sample / "model-servers", stop=False, gpu_profile="spark")
    assert len(paths) == 4 and all(path.exists() for path in paths)
    for path in paths:
        assert path.stat().st_mode & 0o777 == 0o600
        assert "inline-test-credential" not in path.read_text()


@pytest.mark.parametrize("service", ["gemma4", "d3"])
def test_server_uses_shared_docker_lifecycle_and_model_override(service, monkeypatch, tmp_path):
    module = _module(f"_test_{service}_main", _ROOT / f"services/{service}-server/{service}_server/__main__.py")
    config_path = _ROOT / f"services/{service}-server/{service}_server.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.update(model="custom/checkpoint", cuda_visible_devices="1", spark_uma=False,
                  kv_cache_memory_bytes=1234, enforce_eager=True)
    monkeypatch.setattr(module, "setup_logging", lambda *_a: None)
    monkeypatch.setattr(module, "load_config", lambda: (config, config_path.parent, tmp_path / "ready"))
    monkeypatch.setattr(module, "resolve_model_cache", lambda *_a, **_kw: tmp_path)
    monkeypatch.setattr(module, "setup_hf_env", lambda *_a: "1")
    captured = {}
    monkeypatch.setattr(module, "serve", lambda **kwargs: captured.update(kwargs))
    module.run()
    assert captured["model"] == "custom/checkpoint"
    assert captured["backend"] == "docker" and captured["persistent"] is True
    assert captured["cuda_visible_devices"] == "1" and captured["ready_file"] == tmp_path / "ready"
    assert captured["image"] == "vllm/vllm-openai:v0.33.0"
    args = captured["extra_serve_args"]
    assert args[args.index("--served-model-name") + 1] == service
    assert "--enforce-eager" in args and args[args.index("--kv-cache-memory-bytes") + 1] == "1234"
    if service == "gemma4":
        assert args[args.index("--reasoning-parser") + 1] == "gemma4"
        assert json.loads(args[args.index("--limit-mm-per-prompt") + 1]) == {"image": 1, "audio": 0, "video": 0}
    else:
        assert "--no-enable-prefix-caching" in args and "--no-enable-chunked-prefill" in args
        assert "--reasoning-parser" not in args


async def test_gemma_text_vision_and_d3_use_existing_adapter_wire_contracts(monkeypatch):
    models = load_models_config(_PROFILES / "gemma4-e2b-d3-nano.yaml")
    real_client = httpx.AsyncClient
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jN60AAAAASUVORK5CYII="
    )
    calls = []

    def handle(request):
        payload = json.loads(request.content)
        calls.append((request.url.path, payload))
        if request.url.path == "/v1/systemone":
            assert payload["model"] == "d3"
            return httpx.Response(200, json={"model": "d3", "usage": {}, "answers": {"route": {
                "type": "choice", "choice": "vision", "confidence": 0.8,
                "probabilities": {"vision": 0.8, "conversation": 0.2},
            }}})
        assert payload["model"] == "gemma4"
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}
        assert "reasoning_budget" not in payload["chat_template_kwargs"]
        if payload.get("stream"):
            events = [
                {"choices": [{"delta": {"reasoning": "private reasoning"}}]},
                {"choices": [{"delta": {"content": "answer"}}]},
            ]
            data = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
            return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer", "reasoning": "reason"}}]})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    async with make_llm(models, "llm") as llm, make_vlm(models, "vlm") as vlm, make_decision(models, "router") as d3:
        answer = await llm.chat([ChatMessage(role="user", content="Explain this")])
        assert answer.content == "answer" and answer.reasoning == "reason"
        await vlm.ask_image(image, "What is visible?")
        assert [part async for part in llm.stream([ChatMessage(role="user", content="Explain this")])] == ["answer"]
        assert [part async for part in vlm.stream(image, "What is visible?")] == ["answer"]
        decision = await d3.decide("Look at this", {"route": DecisionQuestion(
            instructions="Choose a destination",
            criteria={"vision": "Visual question", "conversation": "Text question"},
        )}, images=[image])
        assert decision.answers["route"].choice == "vision"
    assert [path for path, _ in calls] == ["/v1/chat/completions"] * 4 + ["/v1/systemone"]
    assert calls[1][1]["messages"][-1]["content"][0]["type"] == "image_url"
    assert calls[-1][1]["images"][0].startswith("data:image/")


async def test_standard_omni_stack_shares_endpoint_with_separate_text_and_vision_policies(monkeypatch):
    models = load_models_config(_SAMPLE / "yaml/models.json")
    real_client = httpx.AsyncClient
    calls = []
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jN60AAAAASUVORK5CYII="
    )

    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert request.url.port == 8108 and request.url.path == "/v1/chat/completions"
        assert payload["model"] == "llm"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ready"}}]})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    async with make_llm(models, "llm") as llm, make_vlm(models, "vlm") as vlm:
        await llm.chat([ChatMessage(role="user", content="Explain this")])
        await vlm.ask_image(image, "What is visible?")
    conversation, vision = calls
    assert conversation["chat_template_kwargs"] == {"reasoning_budget": 4096, "enable_thinking": True}
    assert vision["chat_template_kwargs"] == {"enable_thinking": False}
    assert vision["messages"][-1]["content"][0]["type"] == "image_url"
    assert conversation["messages"][-1]["content"] == "Explain this"
