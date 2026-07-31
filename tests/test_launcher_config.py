# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stdlib-only launcher model deployment reads."""

from xr_ai_launcher import load_model_deployment, read_config_scalar


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


def test_model_deployment_drives_processes_and_credentials(tmp_path) -> None:
    config = tmp_path / "worker.yaml"
    config.write_text("models_config: hosted.json\n", encoding="utf-8")
    (tmp_path / "hosted.json").write_text(
        """{
          "models": {
            "vision": {
              "endpoint": {"api_key_env": "NVIDIA_API_KEY"},
              "deployment": {"ownership": "external"}
            },
            "speech": {
              "endpoint": {"base_url": "http://localhost:8103"},
              "deployment": {"ownership": "managed", "service": "stt"}
            },
            "reasoning": {
              "endpoint": {"base_url": "http://localhost:8107"},
              "deployment": {"ownership": "reused", "service": "agent-llm"}
            }
          }
        }""",
        encoding="utf-8",
    )

    deployment = load_model_deployment(config)

    assert deployment.launch_mode("stt") == "own"
    assert deployment.launch_mode("agent-llm") == "reuse"
    assert deployment.launch_mode("vlm") is None
    assert deployment.required_credentials == ("NVIDIA_API_KEY",)
