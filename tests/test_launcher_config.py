# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stdlib-only launcher model deployment reads."""

import json
from pathlib import Path

import pytest
from xr_ai_launcher import load_model_deployment
from xr_ai_models import load_models_config

_ROOT = Path(__file__).resolve().parents[1]
_SIMPLE_VLM_YAML = _ROOT / "agent-samples" / "simple-vlm-example" / "yaml"


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
    [
        "models.local.json",
        "models.local.magpie.json",
        "models.hosted.json",
        "models.omni.json",
    ],
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
    magpie = load("models.local.magpie.json")
    hosted = load("models.hosted.json")
    omni = load("models.omni.json")

    assert local.services == {"stt": "own", "vlm": "own", "tts": "own"}
    assert magpie.services == {
        "stt": "own",
        "vlm-magpie": "own",
        "tts-magpie": "own",
    }
    assert hosted.services == {"stt": "own", "tts": "own"}
    assert hosted.required_credentials == ("NGC_API_KEY",)
    assert omni.services == {"stt": "own", "vlm-omni": "reuse", "tts": "own"}


def test_bundled_worker_selects_local_profile() -> None:
    deployment = load_model_deployment(
        _SIMPLE_VLM_YAML / "simple_vlm_example_worker.yaml"
    )

    assert deployment.profile_path == _SIMPLE_VLM_YAML / "models.local.json"
