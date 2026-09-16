# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stdlib-only launcher model deployment reads."""

import json
from pathlib import Path

import pytest
from xr_ai_launcher import load_deployment_profile, load_model_deployment
from xr_ai_launcher._config import _resolve_config_variant
from xr_ai_models import DeploymentSpec, load_models_config

_ROOT = Path(__file__).resolve().parents[1]
_SIMPLE_VLM_YAML = _ROOT / "agent-samples" / "simple-vlm-example" / "yaml"
_RENDER_YAML = _ROOT / "agent-samples" / "xr-render-demo" / "yaml"


def test_service_config_variant_precedes_default(tmp_path: Path) -> None:
    default = tmp_path / "vlm_server.yaml"
    variant = tmp_path / "vlm_server_default.yaml"
    default.write_text("model: default\n", encoding="utf-8")

    assert _resolve_config_variant(tmp_path, "vlm_server", "default") == default

    variant.write_text("model: variant\n", encoding="utf-8")

    assert _resolve_config_variant(tmp_path, "vlm_server", "default") == variant


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


@pytest.mark.parametrize("relative", [
    "simple-vlm-example/yaml/models.json",
    "xr-render-demo/yaml/models.json",
    "lab-instrument-monitoring/yaml/models.json",
    "tea-making-sample/yaml/models.local.json",
])
def test_consumer_profiles_connect_without_model_lifecycle_metadata(relative) -> None:
    profile = _ROOT / "agent-samples" / relative
    raw = json.loads(profile.read_text())["models"]
    assert all("deployment" not in model for model in raw.values())
    models = load_models_config(profile)
    deployment = load_deployment_profile(profile)

    assert all(spec.deployment == DeploymentSpec() for spec in models.entries.values())
    assert all(spec.endpoint.readiness == "health" for spec in models.entries.values())
    assert deployment.services == {}
    assert deployment.required_credentials == ()


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
    with pytest.raises(ValueError, match="must define adapter and endpoint"):
        load_model_deployment(worker_config)


def test_bundled_simple_vlm_config_supports_worker_accessors() -> None:
    models = load_models_config(_SIMPLE_VLM_YAML / "models.json")

    models.stt("stt")
    models.vlm("vlm")
    models.tts("tts")


def test_bundled_render_config_supports_worker_accessors() -> None:
    models = load_models_config(_RENDER_YAML / "models.json")

    models.llm("llm")
    models.llm("agent_llm")
    models.stt("stt")
    models.vlm("vlm")
    models.tts("tts")


def test_deployment_credentials_are_collected(tmp_path) -> None:
    profile = tmp_path / "models.vlm_llm_nim.json"
    profile.write_text(
        json.dumps({
            "models": {
                "vision": {
                    "adapter": {"preset": "cosmos_vlm"},
                    "endpoint": {"base_url": "http://localhost:8100"},
                    "deployment": {
                        "ownership": "managed",
                        "service": "vlm-nim",
                        "credentials": ["NGC_API_KEY"],
                    },
                }
            }
        }),
        encoding="utf-8",
    )
    config = tmp_path / "worker.yaml"
    config.write_text("models_config: models.vlm_llm_nim.json\n", encoding="utf-8")

    deployment = load_model_deployment(config)

    assert deployment.required_credentials == ("NGC_API_KEY",)


@pytest.mark.parametrize(
    ("credentials", "match"),
    [
        ("NGC_API_KEY", "must be a list"),
        ([123], "non-empty strings"),
        ([""], "non-empty strings"),
    ],
)
def test_invalid_deployment_credentials_rejected(
    tmp_path, credentials, match
) -> None:
    profile = tmp_path / "models.vlm_llm_nim.json"
    profile.write_text(
        json.dumps({
            "models": {
                "vision": {
                    "adapter": {"preset": "cosmos_vlm"},
                    "endpoint": {"base_url": "http://localhost:8100"},
                    "deployment": {
                        "ownership": "managed",
                        "service": "vlm-nim",
                        "credentials": credentials,
                    },
                }
            }
        }),
        encoding="utf-8",
    )
    config = tmp_path / "worker.yaml"
    config.write_text("models_config: models.vlm_llm_nim.json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_model_deployment(config)


_MODEL_SERVERS_YAML = _ROOT / "agent-samples" / "model-servers" / "yaml"


@pytest.mark.parametrize(
    "profile_name",
    [
        "models.default.json",
        "models.vlm_llm_nim.json",
    ],
)
def test_bundled_model_servers_profiles_have_launcher_sdk_parity(
    profile_name,
) -> None:
    profile = _MODEL_SERVERS_YAML / profile_name
    deployment = load_deployment_profile(profile)
    models = load_models_config(profile)
    expected_services = {
        spec.deployment.service: (
            "own" if spec.deployment.ownership == "managed" else "reuse"
        )
        for spec in models.entries.values()
        if spec.deployment.ownership != "external"
    }

    expected_credentials = set(models.required_credentials)
    for spec in models.entries.values():
        expected_credentials.update(spec.deployment.credentials)

    assert deployment.services == expected_services
    assert deployment.required_credentials == tuple(sorted(expected_credentials))


def test_load_deployment_profile_rejects_non_json(tmp_path) -> None:
    with pytest.raises(ValueError, match="must use a .json file"):
        load_deployment_profile(tmp_path / "models.local.yaml")


@pytest.mark.parametrize("deployment, expected_services", [
    (None, {}),
    ({}, {}),
    ({"ownership": "external"}, {}),
    ({"ownership": "reused", "service": "vlm"}, {"vlm": "reuse"}),
    ({"ownership": "managed", "service": "vlm", "credentials": ["NGC_API_KEY"]}, {"vlm": "own"}),
])
def test_optional_deployment_preserves_credentials_and_legacy_modes(tmp_path, deployment, expected_services):
    model = {
        "adapter": {"preset": "cosmos3_nano_reasoner"},
        "endpoint": {"base_url": "https://example.com", "api_key_env": "ENDPOINT_API_KEY", "readiness": "none"},
    }
    if deployment is not None:
        model["deployment"] = deployment
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": {"vlm": model}}))
    sdk = load_models_config(path).vlm("vlm")
    launcher = load_deployment_profile(path)

    assert launcher.services == expected_services
    assert launcher.required_credentials == tuple(sorted({"ENDPOINT_API_KEY", *sdk.deployment.credentials}))
    assert sdk.endpoint.readiness == "none"
    assert sdk.endpoint.api_key_env == "ENDPOINT_API_KEY"
    assert sdk.deployment.ownership == (deployment or {}).get("ownership", "external")


@pytest.mark.parametrize("deployment", [None, [], "reused", False])
def test_optional_deployment_still_rejects_explicit_non_objects(tmp_path, deployment):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": {"vlm": {
        "adapter": {"preset": "cosmos3_nano_reasoner"},
        "endpoint": {"base_url": "http://localhost:8100"},
        "deployment": deployment,
    }}}))
    for loader in (load_models_config, load_deployment_profile):
        with pytest.raises(ValueError, match="deployment"):
            loader(path)
