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
from nim_model_adapter.embedding import build_app
from xr_ai_models import load_models_config, make_embedding, make_llm, make_stt, make_tts, make_vlm
from xr_ai_vllm._nim import build_nim_run_argv

BASE = Path(__file__).resolve().parents[1] / "model-server-samples/model-servers-nim"
SPEC = importlib.util.spec_from_file_location("nim_sample_main", BASE / "main.py")
assert SPEC and SPEC.loader
sample = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sample)
HARDWARE = ("96G_blackwell", "dual_48G_ada", "spark")


@pytest.mark.parametrize("hardware", HARDWARE)
def test_profiles_build_real_sdk_clients_and_persistent_processes(hardware, tmp_path):
    processes, credentials, profile = sample._build_processes(hardware)
    assert len(processes) == (9 if hardware == "spark" else 10)
    assert all(process.launch_mode == "persist" for process in processes)
    for process in processes:
        if process.name in {"stt-nim", "tts-nim"}:
            assert process.project == "riva-server"
            assert process.command == "nim_riva_server"
        if process.name == "tts-adapter":
            assert process.project == "../../services/magpie-nim-tts"
            assert process.command == "magpie_nim_tts"
    assert len({process.port for process in processes}) == len(processes)
    assert [p.name for p in processes][-4:] == ["llm-nim", "llm-adapter", "vlm-nim", "vlm-adapter"]
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
    original = load_models_config(BASE.parent / "model-servers/yaml/models.default.json")
    for name, spec in config.entries.items():
        assert spec.adapter == original.entries[name].adapter
        assert spec.endpoint == original.entries[name].endpoint

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


@pytest.mark.parametrize("deployment", [{}, {"credentials": ["NGC_API_KEY"]}])
def test_cli_exports_custom_profile_with_default_external_ownership(tmp_path, monkeypatch, deployment):
    model = {
        "adapter": {"preset": "nemotron_omni"},
        "endpoint": {"base_url": "https://example.com", "api_key_env": "EXTERNAL_API_KEY"},
        "deployment": deployment,
    }
    profile = tmp_path / "custom.json"
    profile.write_text(json.dumps({"models": {"llm": model}}))
    original = profile.read_bytes()
    processes, _, selected = sample._build_processes("96G_blackwell", profile)
    assert processes == []
    assert selected == profile

    destination = tmp_path / "client.json"
    monkeypatch.setattr(sys, "argv", [
        "model_servers_nim", "--gpu-profile", "96G_blackwell", "--models", str(profile),
        "--export-models", str(destination),
    ])
    sample.run()

    assert profile.read_bytes() == original
    exported = json.loads(destination.read_text())["models"]["llm"]
    assert exported["adapter"] == model["adapter"]
    assert exported["endpoint"] == model["endpoint"]
    assert exported["deployment"] == {}
    client = load_models_config(destination).llm("llm")
    assert client.deployment.ownership == "external"
    assert not client.deployment.credentials


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
    monkeypatch.setattr(sample, "pid_on_port_checked", lambda port: (None, True, False))
    monkeypatch.setattr(sample, "require_credentials", lambda *args, **kwargs: pytest.fail("unexpected credentials"))
    monkeypatch.setattr(sample, "stop_persistent_servers", lambda targets: stopped.extend(targets) or True)
    monkeypatch.setattr(sample, "detect_gpu_config", lambda: pytest.fail("unexpected GPU detection"))
    monkeypatch.setattr(sys, "argv", ["model_servers_nim", "--stop"])
    sample.run()
    assert len(stopped) == len({port for _, port in stopped})
    assert {port for _, port in stopped} == {8100, 8103, 8105, 8108, 8109, 8110, 8118, 8119, 9010, 9011}


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


@pytest.mark.parametrize("text", ["", "port: bad", "port: 0", "http_port: 65536"])
def test_invalid_port_is_rejected(tmp_path, text):
    config = tmp_path / "server.yaml"
    config.write_text(text)
    with pytest.raises(ValueError, match="port"):
        sample._port(config)


def test_duplicate_ports_fail_before_launch(monkeypatch):
    monkeypatch.setattr(sample, "_port", lambda config: 8100)
    with pytest.raises(ValueError, match="multiple services use port"):
        sample._build_processes("spark")


@pytest.mark.parametrize("marked,label", [(False, "stt"), (True, "stt-adapter")])
def test_known_ports_selects_stt_cleanup_by_listener_ownership(monkeypatch, marked, label):
    monkeypatch.setattr(sample, "pid_on_port_checked", lambda port: (123, True, True))
    monkeypatch.setattr(sample, "has_xr_ai_ownership_marker", lambda pid, port: marked)
    assert (label, 8103) in sample._known_ports()



def test_gpu_profile_names_follow_checked_in_directories(tmp_path, monkeypatch):
    root = tmp_path / "yaml" / "custom"
    root.mkdir(parents=True)
    (root / "models.json").write_text("{}")
    monkeypatch.setattr(sample, "_BASE", tmp_path)
    assert sample._gpu_profile_name("custom") == "custom"
    with pytest.raises(sample.argparse.ArgumentTypeError, match="available profiles: custom"):
        sample._gpu_profile_name("unknown")


def reduced_profile(tmp_path, *, reused_backend=False, reused_adapter=False):
    original = json.loads((BASE / "yaml/96G_blackwell/models.json").read_text())["models"]
    tts = original["tts"]
    tts["deployment"]["credentials"] = []
    models = {"tts": tts}
    if reused_backend:
        models["tts_backend"] = {
            "adapter": {"kind": "riva_grpc"}, "endpoint": {"base_url": "localhost:50052"},
            "deployment": {"ownership": "reused", "service": "tts-nim"},
        }
    if reused_adapter:
        models["vlm"] = original["vlm"]
        models["vlm"]["deployment"].update(ownership="reused", credentials=[])
    path = tmp_path / "reduced.json"
    path.write_text(json.dumps({"models": models}))
    return path


def test_owned_adapter_does_not_launch_explicitly_reused_backend(tmp_path):
    profile = reduced_profile(tmp_path, reused_backend=True)
    processes, credentials, _ = sample._build_processes("96G_blackwell", profile)
    assert [process.name for process in processes] == ["tts-adapter"]
    assert not credentials


@pytest.mark.parametrize("reused_backend", [False, True])
@pytest.mark.parametrize("reused_adapter", [False, True])
def test_reduced_profile_stops_omitted_services_before_launch(
    tmp_path, monkeypatch, reused_backend, reused_adapter,
):
    profile = reduced_profile(tmp_path, reused_backend=reused_backend, reused_adapter=reused_adapter)
    known = [("tts-adapter", 8105), ("tts-nim", 9011), ("vlm-adapter", 8100),
             ("vlm-nim", 8110), ("llm-adapter", 8108), ("llm-nim", 8118)]
    monkeypatch.setattr(sample, "_known_ports", lambda: known)
    events = []
    monkeypatch.setattr(sample, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(sample, "require_credentials", lambda *args, **kwargs: pytest.fail("no credentials needed"))
    monkeypatch.setattr(sample, "stop_persistent_servers", lambda ports: events.append(("stop", ports)) or True)
    monkeypatch.setattr(sample, "run_stack", lambda *args, **kwargs: events.append(("launch", args[0])))
    monkeypatch.setattr(sys, "argv", ["model_servers_nim", "--gpu-profile", "96G_blackwell", "--models", str(profile)])
    sample.run()
    assert [event for event, _ in events] == ["stop", "launch"]
    stopped = set(events[0][1])
    assert ("tts-adapter", 8105) not in stopped and ("tts-nim", 9011) not in stopped
    assert ("llm-adapter", 8108) in stopped and ("llm-nim", 8118) in stopped
    assert (("vlm-adapter", 8100) not in stopped) == reused_adapter
    assert (("vlm-nim", 8110) not in stopped) == reused_adapter
    launched = {process.name for process in events[1][1]}
    assert launched == ({"tts-adapter"} if reused_backend else {"tts-adapter", "tts-nim"})


def test_failed_profile_cleanup_prevents_launch(tmp_path, monkeypatch, capsys):
    profile = reduced_profile(tmp_path)
    monkeypatch.setattr(sample, "_known_ports", lambda: [("llm-nim", 8118)])
    monkeypatch.setattr(sample, "stop_persistent_servers", lambda ports: False)
    monkeypatch.setattr(sample, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(sample, "run_stack", lambda *args, **kwargs: pytest.fail("launched despite failed cleanup"))
    monkeypatch.setattr(sys, "argv", ["model_servers_nim", "--gpu-profile", "96G_blackwell", "--models", str(profile)])
    with pytest.raises(SystemExit) as caught:
        sample.run()
    assert caught.value.code == 2
    assert "could not stop persistent model servers" in capsys.readouterr().err



def test_cleanup_preserves_external_local_endpoint_but_not_old_owned_port(tmp_path, monkeypatch):
    profile = reduced_profile(tmp_path)
    data = json.loads(profile.read_text())
    data["models"]["external"] = {
        "adapter": {"preset": "nemotron_omni"},
        "endpoint": {"base_url": "http://localhost:8118"},
        "deployment": {"ownership": "external"},
    }
    profile.write_text(json.dumps(data))
    processes, _, _ = sample._build_processes("96G_blackwell", profile)
    monkeypatch.setattr(sample, "_known_ports", lambda: [("tts-adapter", 8105), ("tts-adapter", 8205),
                                                        ("tts-nim", 9011), ("llm-nim", 8118)])
    stopped = []
    monkeypatch.setattr(sample, "stop_persistent_servers", lambda ports: stopped.extend(ports) or True)
    sample._stop_unselected_services(processes, profile)
    assert stopped == [("tts-adapter", 8205)]
