# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stdlib-only launcher model deployment reads."""

import importlib.util
import json
from pathlib import Path

import pytest
from xr_ai_launcher import load_model_deployment
from xr_ai_models import load_models_config

_ROOT = Path(__file__).resolve().parents[1]
_SIMPLE_VLM_YAML = _ROOT / "agent-samples" / "simple-vlm-example" / "yaml"
_SIMPLE_VLM_MAIN = _ROOT / "agent-samples" / "simple-vlm-example" / "main.py"
_SIMPLE_VLM_SPEC = importlib.util.spec_from_file_location(
    "simple_vlm_example_main",
    _SIMPLE_VLM_MAIN,
)
assert _SIMPLE_VLM_SPEC and _SIMPLE_VLM_SPEC.loader
_simple_vlm = importlib.util.module_from_spec(_SIMPLE_VLM_SPEC)
_SIMPLE_VLM_SPEC.loader.exec_module(_simple_vlm)


def _write_profile(path: Path, *, credential: str | None = None) -> None:
    endpoint: dict[str, str] = {"base_url": "http://localhost:8100"}
    if credential:
        endpoint["api_key_env"] = credential
    path.write_text(
        json.dumps({
            "models": {
                "vision": {
                    "adapter": {"preset": "cosmos_vlm"},
                    "endpoint": endpoint,
                    "deployment": {"ownership": "managed", "service": "vlm"},
                }
            }
        }),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "selection",
    [
        "models.hosted.json",
        "'models.hosted.json'",
        '"models.hosted.json"',
        '"models.hosted.json" # hosted profile',
    ],
)
def test_profile_selection_supports_plain_quoted_and_commented_values(
    tmp_path, selection
) -> None:
    _write_profile(tmp_path / "models.hosted.json", credential="NVIDIA_API_KEY")
    config = tmp_path / "worker.yaml"
    config.write_text(f"models_config: {selection}\n", encoding="utf-8")

    deployment = load_model_deployment(config)

    assert deployment.profile_path == tmp_path / "models.hosted.json"
    assert deployment.launch_mode("vlm") == "own"
    assert deployment.required_credentials == ("NVIDIA_API_KEY",)


def test_profile_selection_ignores_nested_and_block_scalar_text(tmp_path) -> None:
    _write_profile(tmp_path / "selected.json")
    _write_profile(tmp_path / "injected.json", credential="WRONG_KEY")
    config = tmp_path / "worker.yaml"
    config.write_text(
        "system_prompt: |\n"
        "  models_config: injected.json\n"
        "nested:\n"
        "  models_config: injected.json\n"
        "models_config: selected.json\n",
        encoding="utf-8",
    )

    deployment = load_model_deployment(config)

    assert deployment.profile_path == tmp_path / "selected.json"
    assert deployment.required_credentials == ()


def test_empty_profile_selection_uses_default_without_consuming_next_key(tmp_path) -> None:
    _write_profile(tmp_path / "models.local.json")
    config = tmp_path / "worker.yaml"
    config.write_text(
        "models_config:\nprofile: injected.json\n",
        encoding="utf-8",
    )

    deployment = load_model_deployment(config)

    assert deployment.profile_path == tmp_path / "models.local.json"


@pytest.mark.parametrize("selection", ["# use default", "''", '\"\"'])
def test_effectively_empty_profile_selection_uses_default(
    tmp_path, selection
) -> None:
    _write_profile(tmp_path / "models.local.json")
    config = tmp_path / "worker.yaml"
    config.write_text(f"models_config: {selection}\n", encoding="utf-8")

    deployment = load_model_deployment(config)

    assert deployment.profile_path == tmp_path / "models.local.json"


@pytest.mark.parametrize(
    "profile_name",
    ["models.local.json", "models.hosted.json", "models.omni.json"],
)
def test_bundled_simple_vlm_profiles_have_launcher_sdk_parity(
    tmp_path, profile_name
) -> None:
    profile = _SIMPLE_VLM_YAML / profile_name
    worker_config = tmp_path / "worker.yaml"
    worker_config.write_text(f'models_config: "{profile}"\n', encoding="utf-8")

    deployment = load_model_deployment(worker_config)
    models = load_models_config(profile)
    expected_services = {
        spec.deployment.service: (
            "own" if spec.deployment.ownership == "managed" else "reuse"
        )
        for spec in models.entries.values()
        if spec.deployment.ownership != "external"
    }

    assert deployment.services == expected_services
    assert deployment.required_credentials == models.required_credentials
    expected_reused = {
        spec.deployment.service: (spec.endpoint.base_url, spec.endpoint.readiness)
        for spec in models.entries.values()
        if spec.deployment.ownership == "reused"
    }
    assert {
        service: (endpoint.base_url, endpoint.readiness)
        for service, endpoint in deployment.reused_endpoints.items()
    } == expected_reused
    expected_external = {
        role: (
            spec.endpoint.base_url,
            spec.endpoint.readiness,
            spec.endpoint.api_key_env,
        )
        for role, spec in models.entries.items()
        if spec.deployment.ownership == "external"
    }
    assert {
        role: (endpoint.base_url, endpoint.readiness, endpoint.api_key_env)
        for role, endpoint in deployment.external_endpoints.items()
    } == expected_external


def test_launcher_rejects_worker_only_yaml_profile(tmp_path) -> None:
    profile = tmp_path / "models.custom.yaml"
    profile.write_text(
        "models:\n"
        "  vlm:\n"
        "    category: vlm\n"
        "    adapter:\n"
        "      kind: openai_compat\n"
        "      model_name: custom-vlm\n"
        "    endpoint:\n"
        "      base_url: http://localhost:8100\n"
        "    deployment:\n"
        "      ownership: managed\n"
        "      service: vlm\n",
        encoding="utf-8",
    )
    worker_config = tmp_path / "worker.yaml"
    worker_config.write_text(
        "models_config: models.custom.yaml\n",
        encoding="utf-8",
    )

    assert load_models_config(profile).vlm("vlm").deployment.service == "vlm"
    with pytest.raises(ValueError, match=r"must use a \.json file"):
        load_model_deployment(worker_config)


def test_launcher_rejects_worker_only_flat_json_profile(tmp_path) -> None:
    profile = tmp_path / "models.custom.json"
    profile.write_text(
        json.dumps({
            "models": {
                "vlm": {
                    "kind": "preset:cosmos_vlm",
                    "base_url": "http://localhost:8100",
                },
            },
        }),
        encoding="utf-8",
    )
    worker_config = tmp_path / "worker.yaml"
    worker_config.write_text(
        "models_config: models.custom.json\n",
        encoding="utf-8",
    )

    assert load_models_config(profile).vlm("vlm").base_url.endswith(":8100")
    with pytest.raises(ValueError, match="must define adapter, endpoint"):
        load_model_deployment(worker_config)


@pytest.mark.parametrize(
    "profile_name",
    ["models.local.json", "models.hosted.json", "models.omni.json"],
)
def test_bundled_simple_vlm_profiles_support_worker_accessors(profile_name) -> None:
    models = load_models_config(_SIMPLE_VLM_YAML / profile_name)

    models.stt("stt")
    models.vlm("vlm")
    models.tts("tts")


def test_bundled_simple_vlm_profiles_select_expected_ownership(tmp_path) -> None:
    def load(profile_name: str):
        config = tmp_path / f"{profile_name}.yaml"
        config.write_text(
            f'models_config: "{_SIMPLE_VLM_YAML / profile_name}"\n',
            encoding="utf-8",
        )
        return load_model_deployment(config)

    local = load("models.local.json")
    hosted = load("models.hosted.json")
    omni = load("models.omni.json")

    assert local.services == {"stt": "own", "vlm": "own", "tts": "own"}
    assert hosted.services == {"stt": "own", "tts": "own"}
    assert hosted.required_credentials == ("NGC_API_KEY",)
    assert omni.services == {"stt": "own", "vlm-omni": "reuse", "tts": "own"}


def test_bundled_worker_selects_local_profile() -> None:
    deployment = load_model_deployment(
        _SIMPLE_VLM_YAML / "simple_vlm_example_worker.yaml"
    )

    assert deployment.profile_path == _SIMPLE_VLM_YAML / "models.local.json"


def _write_reused_omni_profile(path: Path, *, readiness: str) -> None:
    path.write_text(
        json.dumps({
            "models": {
                "vlm": {
                    "category": "vlm",
                    "adapter": {
                        "kind": "openai_compat",
                        "model_name": "llm",
                    },
                    "endpoint": {
                        "base_url": "http://localhost:9000",
                        "readiness": readiness,
                    },
                    "deployment": {
                        "ownership": "reused",
                        "service": "vlm-omni",
                    },
                }
            }
        }),
        encoding="utf-8",
    )


def test_omni_reused_service_uses_profile_health_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "models.omni.json"
    _write_reused_omni_profile(profile, readiness="health")
    worker_config = tmp_path / "worker.yaml"
    worker_config.write_text(
        f'models_config: "{profile}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(_simple_vlm, "_WORKER_CONFIG", str(worker_config))

    processes, _credentials, _endpoint_probes = _simple_vlm._build_processes()
    reused = next(process for process in processes if process.launch_mode == "reuse")

    assert reused.name == "vlm-omni"
    assert reused.health_url == "http://localhost:9000/health"
    assert reused.readiness == "health"


def test_omni_reused_service_honors_disabled_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "models.omni.json"
    _write_reused_omni_profile(profile, readiness="none")
    worker_config = tmp_path / "worker.yaml"
    worker_config.write_text(f'models_config: "{profile}"\n', encoding="utf-8")
    monkeypatch.setattr(_simple_vlm, "_WORKER_CONFIG", str(worker_config))

    processes, _credentials, _endpoint_probes = _simple_vlm._build_processes()
    reused = next(process for process in processes if process.launch_mode == "reuse")

    assert reused.health_url == "http://localhost:9000/health"
    assert reused.readiness == "none"


def test_external_remote_service_becomes_authenticated_endpoint_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "models.remote.json"
    profile.write_text(
        json.dumps({
            "models": {
                "vlm": {
                    "category": "vlm",
                    "adapter": {
                        "kind": "openai_compat",
                        "model_name": "remote-vlm",
                    },
                    "endpoint": {
                        "base_url": "https://models.example.test/api",
                        "api_key_env": "REMOTE_MODEL_KEY",
                        "readiness": "health",
                    },
                    "deployment": {"ownership": "external"},
                }
            }
        }),
        encoding="utf-8",
    )
    worker_config = tmp_path / "worker.yaml"
    worker_config.write_text(f'models_config: "{profile}"\n', encoding="utf-8")
    monkeypatch.setattr(_simple_vlm, "_WORKER_CONFIG", str(worker_config))

    _processes, credentials, endpoint_probes = _simple_vlm._build_processes()

    assert credentials == ("REMOTE_MODEL_KEY",)
    assert endpoint_probes == (
        _simple_vlm.EndpointProbe(
            name="vlm",
            health_url="https://models.example.test/api/health",
            api_key_env="REMOTE_MODEL_KEY",
        ),
    )
