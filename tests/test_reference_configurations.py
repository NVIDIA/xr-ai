# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copyable package references exercise the real configuration loaders."""

from dataclasses import asdict, fields
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml
from xr_ai_models import load_models_config
from xr_ai_voicegate import VoiceGateConfig, load_voice_gate_config
from xr_ai_voicegate import config as voice_config
from xr_ai_voicegate._conversation import _ConversationConfig
from xr_ai_voicegate._speaker import _SpeakerConfig

_ROOT = Path(__file__).resolve().parents[1]
_VOICE_REFERENCE = _ROOT / "utils/xr-ai-voicegate/xr_ai_voicegate/voice_gate.yaml"
_MODELS_REFERENCE = _ROOT / "agent-sdk/xr-ai-models/models.yaml.sample"


def test_voice_reference_preserves_omitted_configuration_defaults():
    config = load_voice_gate_config(_VOICE_REFERENCE)
    assert config == VoiceGateConfig()
    assert config._speaker is None
    assert config._conversation is None

    raw = yaml.safe_load(_VOICE_REFERENCE.read_text())
    assert set(raw) == {
        field.name for field in fields(VoiceGateConfig) if not field.name.startswith("_")
    } | {"conversation", "speaker"}
    assert set(raw["conversation"]) == set(asdict(_ConversationConfig())) | {"enabled"}
    # Conversation controls are documented only once, not under both mappings.
    assert set(raw["speaker"]) == set(asdict(_SpeakerConfig())) - set(asdict(_ConversationConfig())) | {"enabled"}


def test_voice_reference_optional_sections_use_real_parser(tmp_path):
    raw = yaml.safe_load(_VOICE_REFERENCE.read_text())
    raw["conversation"]["enabled"] = True
    raw["speaker"]["enabled"] = True
    path = tmp_path / "voice_gate.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_voice_gate_config(path)
    assert config._conversation == _ConversationConfig()
    assert config._speaker == _SpeakerConfig()

    raw["speaker"]["backend"] = "required"
    path.write_text(yaml.safe_dump(raw))
    config = load_voice_gate_config(path)
    assert config._speaker.backend == "required"

    # Explicitly disabling conversation also disables speaker ASR.
    raw["conversation"]["enabled"] = False
    path.write_text(yaml.safe_dump(raw))
    assert load_voice_gate_config(path)._speaker is None


def test_models_reference_loads_consumer_roles_with_existing_local_ports(tmp_path):
    working = tmp_path / "models.yaml"
    working.write_text(_MODELS_REFERENCE.read_text())
    config = load_models_config(working)
    role_services = {
        "llm": "nemotron-omni-llm/nemotron_omni_llm_server.yaml",
        "vlm": "vlm-server/vlm_server.yaml",
        "stt": "stt-server/stt_server.yaml",
        "tts": "pocket-tts/pocket_tts_server.yaml",
        "embedding": "embedding-server/embedding_server.yaml",
    }
    assert set(config.entries) == set(role_services) | {"decision"}
    assert config.required_credentials == ()
    for role, spec in config.entries.items():
        assert getattr(config, role)(role) is spec
        assert spec.deployment.ownership == "external"
        assert spec.deployment.service is None
        assert spec.endpoint.api_key_env is None
        assert spec.endpoint.timeout == (60.0 if role in {"llm", "vlm", "embedding"} else 30.0)
        assert spec.endpoint.health_path == "/health"
        assert spec.endpoint.readiness == "health"
        if role in role_services:
            service = yaml.safe_load((_ROOT / "services" / role_services[role]).read_text())
            assert urlsplit(spec.endpoint.base_url).port == service["port"]
    assert config.decision("decision").model_name == "Cloudflare/clef-flash"


@pytest.mark.parametrize("content", [None, "", "{}", "null"])
def test_packaged_voice_defaults_for_missing_or_empty_override(tmp_path, content):
    path = tmp_path / "voice_gate.yaml"
    if content is not None:
        path.write_text(content)
    assert load_voice_gate_config(path) == VoiceGateConfig()


@pytest.mark.parametrize("conversation", [{}, {"enabled": True}, {"enabled": False}, None])
def test_optional_voice_activation_preserves_explicit_controls(tmp_path, conversation):
    raw = {"speaker": {"enabled": True}, "conversation": conversation}
    path = tmp_path / "voice_gate.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_voice_gate_config(path)
    enabled = isinstance(conversation, dict) and conversation.get("enabled", True)
    assert (config._conversation is not None) is enabled
    assert (config._speaker is not None) is enabled


def test_packaged_defaults_are_used_by_partial_overrides(tmp_path, monkeypatch):
    defaults = yaml.safe_load(_VOICE_REFERENCE.read_text())
    defaults["followup_grace_s"] = 8.0
    defaults["speaker"]["timeout_s"] = 15.0
    defaults["conversation"]["start_phrase"] = "Begin conversation"
    package = tmp_path / "package"
    package.mkdir()
    (package / "voice_gate.yaml").write_text(yaml.safe_dump(defaults))
    monkeypatch.setattr(voice_config, "files", lambda _package: package)
    path = tmp_path / "override.yaml"
    path.write_text("magic_phrases: [hello]\nspeaker: {enabled: true, backend: required}\n")
    config = load_voice_gate_config(path)
    assert config.magic_phrases == ("hello",)
    assert config.followup_grace_s == 8.0
    assert config._speaker.timeout_s == 15.0
    # Speaker-only input still infers conversation, including packaged controls.
    assert config._conversation.start_phrase == "Begin conversation"
    assert config._speaker.backend == "required"


@pytest.mark.parametrize("raw", [
    ["invalid"], {"conversation": False}, {"speaker": []},
    {"conversation": {"typo": True}}, {"speaker": {"typo": True}},
    {"speaker": {"enabled": False, "backend": "invalid"}},
    {"conversation": {"enabled": False, "phrase_window_s": -1}},
])
def test_packaged_defaults_do_not_hide_invalid_overrides(tmp_path, raw):
    path = tmp_path / "voice_gate.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_voice_gate_config(path)


def test_packaged_defaults_preserve_null_and_unknown_top_level_policy(tmp_path):
    path = tmp_path / "voice_gate.yaml"
    path.write_text("magic_phrases: null\nconversation: null\nspeaker: null\nunknown: ignored\n")
    assert load_voice_gate_config(path) == VoiceGateConfig()


def test_corrupt_packaged_voice_defaults_fail_clearly(tmp_path, monkeypatch):
    (tmp_path / "voice_gate.yaml").write_text("- not a mapping\n")
    monkeypatch.setattr(voice_config, "files", lambda _package: tmp_path)
    with pytest.raises(ValueError, match="packaged voice_gate.yaml: top-level"):
        load_voice_gate_config(tmp_path / "missing.yaml")
