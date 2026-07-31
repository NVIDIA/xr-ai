# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stdlib-only launcher model-profile reads."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from xr_ai_launcher import load_model_deployment, read_config_scalar
from xr_ai_models import load_models_config

_SIMPLE_PROFILES = (
    Path(__file__).parents[1] / "agent-samples" / "simple-vlm-example" / "yaml"
)


def test_read_config_scalar_supports_plain_and_quoted_values(tmp_path) -> None:
    config = tmp_path / "worker.yaml"
    config.write_text(
        "models_config: models.hosted.json # hosted\n"
        "profile: 'apple-vision-pro'\n"
        'label: "demo worker"\n',
        encoding="utf-8",
    )

    assert read_config_scalar(config, "models_config") == "models.hosted.json"
    assert read_config_scalar(config, "profile") == "apple-vision-pro"
    assert read_config_scalar(config, "label") == "demo worker"


def test_read_config_scalar_returns_default_for_missing_input(tmp_path) -> None:
    assert read_config_scalar(tmp_path / "missing.yaml", "mode", "local") == "local"


def test_model_deployment_drives_processes_and_credentials(tmp_path) -> None:
    config = tmp_path / "worker.yaml"
    config.write_text("models_config: hosted.json\n", encoding="utf-8")
    (tmp_path / "hosted.json").write_text(
        json.dumps({
            "models": {
                "vision": {
                    "endpoint": {
                        "api_key_env": "NVIDIA_API_KEY",
                        "readiness": "none",
                    },
                    "deployment": {"ownership": "external"},
                },
                "speech": {
                    "endpoint": {"base_url": "http://localhost:8103"},
                    "deployment": {"ownership": "managed", "service": "stt"},
                },
                "reasoning": {
                    "endpoint": {"base_url": "http://localhost:8107"},
                    "deployment": {
                        "ownership": "reused",
                        "service": "agent-llm",
                    },
                },
            },
        }),
        encoding="utf-8",
    )

    deployment = load_model_deployment(config)

    assert deployment.launch_mode("stt") == "own"
    assert deployment.launch_mode("agent-llm") == "reuse"
    assert deployment.launch_mode("vlm") is None
    assert deployment.required_credentials == ("NVIDIA_API_KEY",)


@pytest.mark.parametrize(
    ("models", "message"),
    [
        (
            {
                "vlm": {
                    "endpoint": {"readiness": "socket"},
                    "deployment": {"ownership": "external"},
                },
            },
            "unsupported readiness",
        ),
        (
            {
                "vlm": {
                    "endpoint": {},
                    "deployment": {"ownership": "borrowed"},
                },
            },
            "unsupported ownership",
        ),
        (
            {
                "vlm": {
                    "endpoint": {},
                    "deployment": {"ownership": "managed"},
                },
            },
            "needs a service",
        ),
        (
            {
                "stt": {
                    "endpoint": {},
                    "deployment": {"ownership": "managed", "service": "speech"},
                },
                "tts": {
                    "endpoint": {},
                    "deployment": {"ownership": "reused", "service": "speech"},
                },
            },
            "conflicting ownership",
        ),
        (
            {
                "vlm": {
                    "endpoint": {"api_key_env": 42},
                    "deployment": {"ownership": "external"},
                },
            },
            "api_key_env",
        ),
    ],
)
def test_model_deployment_rejects_invalid_metadata(tmp_path, models, message) -> None:
    config = tmp_path / "worker.yaml"
    config.write_text("models_config: invalid.json\n", encoding="utf-8")
    (tmp_path / "invalid.json").write_text(
        json.dumps({"models": models}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_model_deployment(config)


def test_model_deployment_rejects_non_json_and_missing_models_root(tmp_path) -> None:
    config = tmp_path / "worker.yaml"
    config.write_text("models_config: profile.json\n", encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text("not JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot load model profile"):
        load_model_deployment(config)

    profile.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="'models' must be an object"):
        load_model_deployment(config)


@pytest.mark.parametrize(
    ("profile_name", "services", "credentials"),
    [
        (
            "models.local.json",
            {"stt": "own", "vlm": "own", "tts": "own"},
            (),
        ),
        (
            "models.hosted-nim.json",
            {"stt": "own", "tts": "own"},
            ("NGC_API_KEY",),
        ),
        (
            "models.omni.json",
            {"stt": "own", "omni": "own", "tts": "own"},
            (),
        ),
    ],
)
def test_shipped_simple_profiles_agree_across_worker_and_launcher_views(
    tmp_path,
    profile_name,
    services,
    credentials,
) -> None:
    profile = _SIMPLE_PROFILES / profile_name
    models = load_models_config(profile)
    worker = tmp_path / "worker.yaml"
    worker.write_text(f"models_config: {profile}\n", encoding="utf-8")

    deployment = load_model_deployment(worker)

    assert deployment.services == services
    assert deployment.required_credentials == credentials
    assert models.required_credentials == credentials
