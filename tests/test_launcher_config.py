# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stdlib-only launcher model deployment reads."""

import json
from pathlib import Path

import pytest
from xr_ai_launcher import load_deployment_profile, load_model_deployment
from xr_ai_launcher._config import _resolve_config_variant
from xr_ai_models import load_models_config

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


def test_bundled_simple_vlm_config_reuses_every_model() -> None:
    models = load_models_config(_SIMPLE_VLM_YAML / "models.json")

    assert {
        name: spec.deployment.ownership
        for name, spec in models.entries.items()
    } == {"stt": "reused", "vlm": "reused", "tts": "reused"}


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


def test_bundled_simple_vlm_config_supports_worker_accessors() -> None:
    models = load_models_config(_SIMPLE_VLM_YAML / "models.json")

    models.stt("stt")
    models.vlm("vlm")
    models.tts("tts")


def test_bundled_render_config_reuses_every_model() -> None:
    models = load_models_config(_RENDER_YAML / "models.json")

    assert {
        name: spec.deployment.ownership
        for name, spec in models.entries.items()
    } == {
        "llm": "reused",
        "agent_llm": "reused",
        "stt": "reused",
        "tts": "reused",
        "vlm": "reused",
    }


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
