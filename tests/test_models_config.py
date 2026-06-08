# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``load_models_config`` + preset resolution coverage."""
from __future__ import annotations

import pytest

from xr_ai_models import (
    LLMSpec,
    STTSpec,
    TTSSpec,
    VLMSpec,
    load_models_config,
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
    assert spec.base_url        == "http://localhost:8107"
    assert spec.model_name      == "llm"
    assert spec.reasoning_field == "reasoning"
    assert spec.capabilities["reasoning"] is True


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
