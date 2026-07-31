# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``load_models_config`` + preset resolution coverage."""
from __future__ import annotations

import pytest

from xr_ai_models import (
    AdapterSpec,
    DeploymentSpec,
    EndpointSpec,
    LLMSpec,
    STTSpec,
    TTSSpec,
    VLMSpec,
    load_models_config,
    load_models_config_from_dict,
    make_llm,
    make_stt,
    make_tts,
    make_vlm,
)
from xr_ai_models.presets import available_presets, get_preset


# ── preset registry ───────────────────────────────────────────────────────


def test_seven_presets_registered() -> None:
    assert set(available_presets()) == {
        "cosmos_vlm",
        "llama_nemotron",
        "magpie_tts",
        "nemotron3_nano",
        "nemotron_omni",
        "parakeet_stt",
        "piper_tts",
    }


def test_get_preset_returns_deep_copy() -> None:
    p1 = get_preset("nemotron3_nano")
    p1["capabilities"]["tool_calls"] = False
    p2 = get_preset("nemotron3_nano")
    assert p2["capabilities"]["tool_calls"] is True


def test_unknown_preset_raises() -> None:
    with pytest.raises(KeyError, match="unknown preset"):
        get_preset("nope")


# ── YAML loader ───────────────────────────────────────────────────────────


def _write(tmp_path, text: str):
    p = tmp_path / "models.yaml"
    p.write_text(text)
    return p


def test_preset_reference_fills_in_defaults(tmp_path) -> None:
    cfg = load_models_config(_write(tmp_path, """
agent_llm:
  kind: preset:nemotron3_nano
  base_url: http://localhost:8107
"""))
    spec = cfg.llm("agent_llm")
    assert isinstance(spec, LLMSpec)
    assert spec.endpoint.base_url == "http://localhost:8107"
    assert spec.adapter.model_name == "llm"
    assert spec.adapter.reasoning_field == "reasoning"
    assert spec.adapter.capabilities["reasoning"] is True


def test_inline_spec_requires_category(tmp_path) -> None:
    with pytest.raises(ValueError, match="category"):
        load_models_config(_write(tmp_path, """
agent_llm:
  kind:       openai_compat
  base_url:   http://localhost:8107
  model_name: llm
"""))


def test_inline_spec_with_explicit_category(tmp_path) -> None:
    cfg = load_models_config(_write(tmp_path, """
agent_llm:
  kind:       openai_compat
  category:   llm
  base_url:   http://localhost:8107
  model_name: llm
  capabilities: { tool_calls: true }
"""))
    spec = cfg.llm("agent_llm")
    assert spec.model_name == "llm"
    assert spec.capabilities == {"tool_calls": True}


def test_entry_overrides_preset(tmp_path) -> None:
    cfg = load_models_config(_write(tmp_path, """
agent_llm:
  kind:       preset:nemotron3_nano
  base_url:   http://localhost:9999
  timeout:    120
  reasoning_field: reasoning_content
"""))
    spec = cfg.llm("agent_llm")
    assert spec.base_url        == "http://localhost:9999"
    assert spec.timeout         == 120.0
    assert spec.reasoning_field == "reasoning_content"


def test_health_check_defaults_true_and_parses_false(tmp_path) -> None:
    cfg = load_models_config(_write(tmp_path, """
local_llm:
  kind:     preset:nemotron3_nano
  base_url: http://localhost:8107
nim_llm:
  kind:        openai_compat
  category:    llm
  base_url:    https://integrate.api.nvidia.com
  model_name:  meta/llama-3.1-8b-instruct
  api_key_env: NGC_API_KEY
  health_check: false
"""))
    assert cfg.llm("local_llm").health_check is True
    nim = cfg.llm("nim_llm")
    assert nim.health_check is False
    assert nim.api_key_env == "NGC_API_KEY"
    assert nim.base_url == "https://integrate.api.nvidia.com"


def test_profile_separates_adapter_endpoint_and_deployment(tmp_path) -> None:
    cfg = load_models_config(_write(tmp_path, """
models:
  reasoning:
    category: llm
    adapter:
      preset: nemotron3_nano
    endpoint:
      base_url: http://localhost:8107
      readiness: health
    deployment:
      ownership: reused
      service: agent-llm
  hosted_vision:
    category: vlm
    adapter:
      kind: openai_compat
      model_name: nvidia/example-vlm
      capabilities: { vision: true }
    endpoint:
      base_url: https://example.test
      api_key_env: EXAMPLE_API_KEY
      readiness: none
    deployment:
      ownership: external
"""))

    reasoning = cfg.llm("reasoning")
    assert reasoning.reasoning_field == "reasoning"
    assert reasoning.deployment.service == "agent-llm"
    assert reasoning.health_check is True
    hosted = cfg.vlm("hosted_vision")
    assert hosted.health_check is False
    assert cfg.required_credentials == ("EXAMPLE_API_KEY",)


def test_profile_rejects_managed_role_without_service(tmp_path) -> None:
    with pytest.raises(ValueError, match="require a service name"):
        load_models_config(_write(tmp_path, """
models:
  reasoning:
    adapter: { preset: nemotron3_nano }
    endpoint: { base_url: http://localhost:8107 }
    deployment: { ownership: managed }
"""))


def test_profile_requires_credentials_under_endpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="api_key_env in endpoint"):
        load_models_config(_write(tmp_path, """
models:
  hosted_vision:
    category: vlm
    api_key_env: WRONG_LOCATION
    adapter:
      kind: openai_compat
      model_name: example-vlm
    endpoint:
      base_url: https://example.test
      readiness: none
    deployment: { ownership: external }
"""))


@pytest.mark.parametrize("section", ["adapter", "endpoint", "deployment"])
@pytest.mark.parametrize("value", [[], None])
def test_profile_rejects_non_mapping_sections(section, value) -> None:
    profile = {
        "models": {
            "vision": {
                "category": "vlm",
                "adapter": {"kind": "openai_compat", "model_name": "example-vlm"},
                "endpoint": {"base_url": "https://example.test"},
                "deployment": {"ownership": "external"},
            }
        }
    }
    profile["models"]["vision"][section] = value

    with pytest.raises(ValueError, match=rf"{section} must be a mapping"):
        load_models_config_from_dict(profile)


def test_vlm_preset(tmp_path) -> None:
    cfg = load_models_config(_write(tmp_path, """
vlm:
  kind:     preset:cosmos_vlm
  base_url: http://localhost:8100
"""))
    spec = cfg.vlm("vlm")
    assert isinstance(spec, VLMSpec)
    assert spec.model_name == "vlm"
    assert spec.default_extras == {
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert spec.capabilities.get("vision") is True
    assert spec.capabilities.get("video")  is True


def test_stt_and_tts_presets(tmp_path) -> None:
    cfg = load_models_config(_write(tmp_path, """
stt:
  kind:     preset:parakeet_stt
  base_url: http://localhost:8103
tts:
  kind:     preset:piper_tts
  base_url: http://localhost:8105
"""))
    assert isinstance(cfg.stt("stt"), STTSpec)
    assert isinstance(cfg.tts("tts"), TTSSpec)
    assert cfg.stt("stt").base_url == "http://localhost:8103"
    assert cfg.tts("tts").base_url == "http://localhost:8105"


def test_wrong_category_accessor_raises(tmp_path) -> None:
    cfg = load_models_config(_write(tmp_path, """
vlm:
  kind:     preset:cosmos_vlm
  base_url: http://localhost:8100
"""))
    with pytest.raises(TypeError, match="expected LLMSpec"):
        cfg.llm("vlm")


def test_missing_base_url_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="base_url"):
        load_models_config(_write(tmp_path, """
agent_llm:
  kind: preset:nemotron3_nano
"""))


def test_unknown_name_raises(tmp_path) -> None:
    cfg = load_models_config(_write(tmp_path, """
agent_llm:
  kind:     preset:nemotron3_nano
  base_url: http://localhost:8107
"""))
    with pytest.raises(KeyError, match="no spec named"):
        cfg.llm("nope")


# ── factory dispatch ──────────────────────────────────────────────────────


async def test_make_llm_constructs_client(tmp_path) -> None:
    cfg = load_models_config(_write(tmp_path, """
agent_llm:
  kind:     preset:nemotron3_nano
  base_url: http://localhost:8107
"""))
    llm = make_llm(cfg, "agent_llm")
    try:
        assert llm.capabilities.reasoning is True
    finally:
        await llm.close()


async def test_make_vlm_make_stt_make_tts(tmp_path) -> None:
    cfg = load_models_config(_write(tmp_path, """
vlm:
  kind:     preset:cosmos_vlm
  base_url: http://localhost:8100
stt:
  kind:     preset:parakeet_stt
  base_url: http://localhost:8103
tts:
  kind:     preset:piper_tts
  base_url: http://localhost:8105
"""))
    vlm = make_vlm(cfg, "vlm")
    stt = make_stt(cfg, "stt")
    tts = make_tts(cfg, "tts")
    try:
        assert vlm.capabilities.vision is True
        assert stt.health_url == "http://localhost:8103/health"
        assert tts.health_url == "http://localhost:8105/health"
    finally:
        await vlm.close()
        await stt.close()
        await tts.close()


@pytest.mark.parametrize(
    ("role", "category", "preset", "spec_type", "accessor", "base_url"),
    [
        ("llm", "llm", "nemotron3_nano", LLMSpec, "llm", "http://localhost:8107"),
        ("vlm", "vlm", "cosmos_vlm", VLMSpec, "vlm", "http://localhost:8100"),
        ("stt", "stt", "parakeet_stt", STTSpec, "stt", "http://localhost:8103"),
        ("tts", "tts", "piper_tts", TTSSpec, "tts", "http://localhost:8105"),
    ],
)
def test_every_role_loads_legacy_and_nested_profiles(
    role,
    category,
    preset,
    spec_type,
    accessor,
    base_url,
) -> None:
    legacy = load_models_config_from_dict({
        role: {
            "kind": f"preset:{preset}",
            "base_url": base_url,
        },
    })
    nested = load_models_config_from_dict({
        "models": {
            role: {
                "category": category,
                "adapter": {"preset": preset},
                "endpoint": {
                    "base_url": base_url,
                    "readiness": "health",
                },
                "deployment": {
                    "ownership": "managed",
                    "service": role,
                },
            },
        },
    })

    legacy_spec = getattr(legacy, accessor)(role)
    nested_spec = getattr(nested, accessor)(role)
    assert isinstance(legacy_spec, spec_type)
    assert isinstance(nested_spec, spec_type)
    assert legacy_spec.adapter == nested_spec.adapter
    assert legacy_spec.endpoint == nested_spec.endpoint
    assert legacy_spec.deployment == DeploymentSpec()
    assert nested_spec.deployment == DeploymentSpec(
        ownership="managed",
        service=role,
    )


def test_nested_profile_separates_adapter_endpoint_and_deployment() -> None:
    cfg = load_models_config_from_dict({
        "models": {
            "agent_llm": {
                "category": "llm",
                "adapter": {
                    "preset": "nemotron3_nano",
                    "reasoning_field": "reasoning_content",
                },
                "endpoint": {
                    "base_url": "https://models.example.test",
                    "api_key_env": "MODEL_TOKEN",
                    "timeout": 90,
                    "readiness": "none",
                },
                "deployment": {
                    "ownership": "reused",
                    "service": "agent-llm",
                },
            },
        },
    })

    spec = cfg.llm("agent_llm")
    assert spec.adapter == AdapterSpec(
        kind="openai_compat",
        model_name="llm",
        reasoning_field="reasoning_content",
        capabilities={
            "streaming": True,
            "tool_calls": True,
            "reasoning": True,
        },
        default_extras={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert spec.endpoint == EndpointSpec(
        base_url="https://models.example.test",
        api_key_env="MODEL_TOKEN",
        timeout=90.0,
        readiness="none",
    )
    assert spec.deployment == DeploymentSpec(
        ownership="reused",
        service="agent-llm",
    )
    assert cfg.required_credentials == ("MODEL_TOKEN",)


def test_direct_nested_mapping_without_models_root_is_supported() -> None:
    cfg = load_models_config_from_dict({
        "vlm": {
            "category": "vlm",
            "adapter": {"preset": "cosmos_vlm"},
            "endpoint": {"base_url": "http://localhost:8100"},
        },
    })
    assert cfg.vlm("vlm").adapter.model_name == "vlm"


def test_render_shape_fixture_remains_compatible() -> None:
    cfg = load_models_config_from_dict({
        "llm": {
            "kind": "preset:llama_nemotron",
            "base_url": "http://localhost:8106",
        },
        "agent_llm": {
            "kind": "preset:nemotron3_nano",
            "base_url": "http://localhost:8107",
        },
        "stt": {
            "kind": "preset:parakeet_stt",
            "base_url": "http://localhost:8103",
        },
        "tts": {
            "kind": "preset:piper_tts",
            "base_url": "http://localhost:8105",
        },
        "vlm": {
            "kind": "preset:cosmos_vlm",
            "base_url": "http://localhost:8100",
        },
    }, source="render-shape fixture")

    assert set(cfg.entries) == {"llm", "agent_llm", "stt", "tts", "vlm"}
    assert cfg.llm("agent_llm").adapter.reasoning_field == "reasoning"
    assert cfg.vlm("vlm").endpoint.base_url == "http://localhost:8100"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {
                "category": "llm",
                "adapter": [],
                "endpoint": {"base_url": "http://localhost"},
            },
            "adapter must be a mapping",
        ),
        (
            {
                "category": "llm",
                "adapter": {"kind": "other", "model_name": "llm"},
                "endpoint": {"base_url": "http://localhost"},
            },
            "unsupported adapter kind",
        ),
        (
            {
                "category": "llm",
                "adapter": {"kind": "openai_compat", "model_name": "llm"},
                "endpoint": {
                    "base_url": "http://localhost",
                    "readiness": "socket",
                },
            },
            "unsupported readiness",
        ),
        (
            {
                "category": "stt",
                "adapter": {"kind": "openai_compat"},
                "endpoint": {
                    "base_url": "http://localhost",
                    "api_key_env": 123,
                },
            },
            "api_key_env",
        ),
        (
            {
                "category": "tts",
                "adapter": {"kind": "openai_compat"},
                "endpoint": {
                    "base_url": "http://localhost",
                    "timeout": 0,
                },
            },
            "timeout",
        ),
        (
            {
                "category": "vlm",
                "adapter": {
                    "kind": "openai_compat",
                    "model_name": "vlm",
                    "capabilities": [],
                },
                "endpoint": {"base_url": "http://localhost"},
            },
            "capabilities must be a mapping",
        ),
        (
            {
                "category": "stt",
                "adapter": {"kind": "openai_compat"},
                "endpoint": {"base_url": "http://localhost"},
                "deployment": {"ownership": "managed"},
            },
            "require a service",
        ),
        (
            {
                "category": "stt",
                "adapter": {"kind": "openai_compat"},
                "endpoint": {"base_url": "http://localhost"},
                "deployment": {"ownership": "borrowed"},
            },
            "unsupported deployment ownership",
        ),
        (
            {
                "category": "stt",
                "base_url": "http://legacy",
                "adapter": {"kind": "openai_compat"},
                "endpoint": {"base_url": "http://nested"},
            },
            "also declared at role level",
        ),
    ],
)
def test_invalid_nested_profiles_are_rejected(body, message) -> None:
    with pytest.raises(ValueError, match=message):
        load_models_config_from_dict({"models": {"role": body}})


def test_role_specs_keep_read_only_flat_attribute_compatibility() -> None:
    spec = LLMSpec(
        adapter=AdapterSpec(model_name="llm"),
        endpoint=EndpointSpec(base_url="http://localhost", readiness="none"),
    )

    assert spec.model_name == "llm"
    assert spec.base_url == "http://localhost"
    assert spec.health_check is False
    with pytest.raises(AttributeError):
        spec.base_url = "http://other"
