# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the sample's lifecycle, model wiring, and NIM request adaptation."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import shlex
import sys
from pathlib import Path

import httpx
import pytest
import yaml
from nim_embedding_adapter.__main__ import build_app
from xr_ai_models import load_models_config, make_embedding, make_llm, make_stt, make_tts, make_vlm
from xr_ai_vllm._nim import build_nim_run_argv

BASE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("nim_sample_main", BASE / "main.py")
assert SPEC and SPEC.loader
sample = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sample)
HARDWARE = ("96G_blackwell", "dual_48G_ada", "spark")


@pytest.mark.parametrize("hardware", HARDWARE)
def test_profiles_build_real_sdk_clients_and_persistent_processes(hardware, tmp_path):
    processes, credentials, profile = sample._build_processes(hardware)
    assert len(processes) == 6  # Five models plus the embedding API adapter.
    assert all(process.launch_mode == "persist" for process in processes)
    assert len({process.port for process in processes}) == len(processes)
    assert [p.name for p in processes][-2:] == ["llm-nim", "vlm-nim"]
    expected_credentials = {"NGC_API_KEY", "HF_TOKEN"} if hardware == "spark" else {"NGC_API_KEY"}
    assert set(credentials) == expected_credentials
    assert all((BASE / p.project / "pyproject.toml").is_file() for p in processes)

    destination = tmp_path / "client.json"
    original = profile.read_bytes()
    sample._export_models(profile, destination)
    assert profile.read_bytes() == original
    config = load_models_config(destination)
    assert set(config.entries) == {"stt", "tts", "llm", "agent_llm", "vlm", "embedding"}
    assert config.llm("llm") == config.llm("agent_llm")
    assert all(spec.deployment.ownership == "reused" for spec in config.entries.values())
    assert all(not spec.deployment.credentials for spec in config.entries.values())
    assert config.tts("tts").kind == "riva_grpc"
    assert "Magpie" in config.tts("tts").voice
    assert config.stt("stt").kind == ("openai_compat" if hardware == "spark" else "riva_grpc")

    async def construct():
        for factory, name in [(make_stt, "stt"), (make_tts, "tts"), (make_llm, "llm"),
                              (make_llm, "agent_llm"), (make_vlm, "vlm"), (make_embedding, "embedding")]:
            client = factory(config, name)
            await client.close()
    asyncio.run(construct())


@pytest.mark.parametrize("hardware", HARDWARE)
def test_nim_configs_generate_correct_docker_ports_and_gpu_filters(hardware):
    processes, _, _ = sample._build_processes(hardware)
    for process in processes:
        cfg = yaml.safe_load(Path(process.config).read_text())
        if "image" not in cfg:
            continue
        args = build_nim_run_argv(
            image=cfg["image"], container_name=cfg["container_name"],
            http_port=cfg["http_port"], grpc_port=cfg.get("grpc_port"),
            nim_cache=Path("/tmp/cache"), cuda_visible_devices=cfg["cuda_visible_devices"],
            extra_env=cfg["env"],
        )
        assert args[-1] == cfg["image"]
        assert ":latest" not in args[-1]
        internal = 9000 if "grpc_port" in cfg else 8000
        assert f"{process.port}:{internal}" in args
        if "grpc_port" in cfg:
            assert f"{cfg['grpc_port']}:50051" in args
        assert f"NVIDIA_VISIBLE_DEVICES={cfg['cuda_visible_devices']}" in args
        assert "NGC_API_KEY" in args


def test_dual_ada_memory_plan_leaves_headroom_on_both_devices():
    root = BASE / "yaml/dual_48G_ada"
    configs = {p.stem: yaml.safe_load(p.read_text()) for p in root.glob("nim_*.yaml")}
    for name in ("nim_stt_server", "nim_tts_server", "nim_vlm_server"):
        assert configs[name]["cuda_visible_devices"] == "0"
    for name in ("nim_llm_server", "nim_embedding_server"):
        assert configs[name]["cuda_visible_devices"] == "1"
    cosmos = configs["nim_vlm_server"]["env"]
    assert cosmos["NIM_KVCACHE_PERCENT"] == cosmos["NIM_GPU_MEMORY_UTILIZATION"]
    omni = shlex.split(configs["nim_llm_server"]["env"]["NIM_PASSTHROUGH_ARGS"])
    # Planning bounds, not assertions of measured whole-stack GPU usage.
    assert 48 * float(cosmos["NIM_GPU_MEMORY_UTILIZATION"]) + 14.02 + 13 < 45
    assert 48 * float(omni[omni.index("--gpu-memory-utilization") + 1]) + 4 < 45
    assert configs["nim_tts_server"]["env"]["NIM_TAGS_SELECTOR"] == "batch_size=8"


def test_export_rejects_overwriting_deployment():
    profile = BASE / "yaml/spark/models.json"
    with pytest.raises(ValueError, match="must not overwrite"):
        sample._export_models(profile, profile)


def test_unknown_service_fails_before_launch(tmp_path):
    profile = json.loads((BASE / "yaml/spark/models.json").read_text())
    profile["models"]["stt"]["deployment"]["service"] = "typo"
    path = tmp_path / "models.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="unknown model services"):
        sample._build_processes("spark", path)


def test_dry_run_never_requests_credentials_or_touches_servers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run attempted a runtime operation")
    monkeypatch.setattr(sample, "require_credentials", forbidden)
    monkeypatch.setattr(sample, "run_stack", forbidden)
    monkeypatch.setattr(sample, "stop_persistent_servers", forbidden)
    monkeypatch.setattr(sample, "detect_gpu_config", forbidden)
    monkeypatch.setattr(sys, "argv", ["model_servers_nim", "--gpu-profile", "spark", "--dry-run"])
    sample.run()


def test_stop_does_not_need_gpu_detection_or_credentials(monkeypatch):
    stopped = []
    monkeypatch.setattr(sample, "stop_persistent_servers", lambda targets: stopped.extend(targets) or True)
    monkeypatch.setattr(sample, "detect_gpu_config", lambda: pytest.fail("unexpected GPU detection"))
    monkeypatch.setattr(sys, "argv", ["model_servers_nim", "--stop"])
    sample.run()
    assert {port for _, port in stopped} == {8100, 8103, 8108, 8109, 8119, 9010, 9011}


class Backend:
    def __init__(self, offset):
        self.calls = []
        self.offset = offset
        self.ready = True

    async def embed(self, texts):
        self.calls.append(texts)
        return [[self.offset + len(text)] for text in texts]

    async def health(self):
        return self.ready

    async def close(self):
        pass


def test_embedding_adapter_groups_prefixes_and_preserves_input_order():
    async def check():
        query, passage = Backend(100), Backend(200)
        app = build_app("http://unused", "nvidia/model", clients={"query": query, "passage": passage})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://adapter") as client:
            response = await client.post("/v1/embeddings", json={
                "model": "embed", "input": ["passage: alpha", "query: beta", "gamma"],
            })
            assert response.status_code == 200
            assert [row["embedding"] for row in response.json()["data"]] == [[205], [104], [205]]
            assert query.calls == [["beta"]]
            assert passage.calls == [["alpha", "gamma"]]
            assert (await client.get("/health")).status_code == 200
            passage.ready = False
            assert (await client.get("/health")).status_code == 503
            conflict = await client.post("/v1/embeddings", json={"input": "query: x", "input_type": "passage"})
            assert conflict.status_code == 422
    asyncio.run(check())


def test_embedding_adapter_uses_actual_sdk_and_nim_model_suffixes(monkeypatch):
    import xr_ai_models._openai_compat as sdk
    real_client = httpx.AsyncClient
    requests = []

    def upstream(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] in ("nvidia/model-query", "nvidia/model-passage")
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [1.0]} for i in range(len(body["input"]))]})

    monkeypatch.setattr(
        sdk.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(upstream), **kw),
    )

    async def check():
        app = build_app("http://nim", "nvidia/model")
        async with app.router.lifespan_context(app):
            async with real_client(transport=httpx.ASGITransport(app), base_url="http://adapter") as client:
                response = await client.post("/v1/embeddings", json={"input": ["query: x", "passage: y"]})
                assert response.status_code == 200
        assert requests == [
            {"model": "nvidia/model-query", "input": ["x"]},
            {"model": "nvidia/model-passage", "input": ["y"]},
        ]
    asyncio.run(check())
