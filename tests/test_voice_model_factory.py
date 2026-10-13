# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configured voice construction, real adapters, and executable NIM selection."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from xr_ai_models import ModelsConfig, load_models_config, load_models_config_from_dict
from xr_ai_models._openai_compat import OpenAICompatSTT
from xr_ai_runtime import Topic
from xr_ai_voice import UserQuery, VoiceAgent, _runtime, _session
from xr_ai_voice._processors import VadConfig
from xr_ai_voicegate import load_voice_gate_config

ROOT = Path(__file__).resolve().parents[1]


def gate(tmp_path, enabled, backend):
    path = tmp_path / "voice.yaml"
    path.write_text(yaml.safe_dump({"speaker": {"enabled": enabled, "backend": backend}}))
    return load_voice_gate_config(path)


def options(voice_gate):
    return dict(query_topic=Topic("test.factory", UserQuery),
                tts=SimpleNamespace(close=AsyncMock()), vad=VadConfig(), voice_gate=voice_gate)


@pytest.mark.parametrize("with_batch_role", [False, True])
async def test_required_factory_skips_batch_dependency(tmp_path, monkeypatch, with_batch_role):
    models = load_models_config(ROOT / "deployment/nim/models.yaml") if with_batch_role else ModelsConfig({})
    monkeypatch.setattr(_runtime, "make_stt", lambda *_a: pytest.fail("batch dependency constructed"))
    monkeypatch.setattr(_session, "HubVoiceTransport", lambda: pytest.fail("media allocated"))
    voice = VoiceAgent.from_models(models=models, **options(gate(tmp_path, True, "required")))
    assert voice._session.stt is None
    assert voice._session._transport is None
    await voice._session.close()


@pytest.mark.parametrize("enabled, backend", [(False, "auto"), (False, "required"), (True, "auto")])
async def test_factory_ordinary_and_auto_construct_real_batch_client(tmp_path, enabled, backend):
    models = load_models_config_from_dict({"stt": {
        "adapter": {"preset": "parakeet_stt"}, "endpoint": {"base_url": "http://localhost:8103"},
    }})
    voice = VoiceAgent.from_models(models=models, **options(gate(tmp_path, enabled, backend)))
    assert isinstance(voice._session.stt, OpenAICompatSTT)
    await voice._session.close()


@pytest.mark.parametrize("enabled, backend", [(False, "auto"), (False, "required"), (True, "auto")])
def test_factory_missing_batch_role_fails_before_media(tmp_path, monkeypatch, enabled, backend):
    monkeypatch.setattr(_session, "HubVoiceTransport", lambda: pytest.fail("media allocated"))
    with pytest.raises(KeyError, match="stt"):
        VoiceAgent.from_models(models=ModelsConfig({}), **options(gate(tmp_path, enabled, backend)))


def test_factory_wrong_role_and_invalid_capacity_fail_before_client(tmp_path, monkeypatch):
    models = load_models_config(ROOT / "deployment/nim/models.yaml")
    models.entries["stt"] = models.entries["tts"]
    with pytest.raises(TypeError, match="STTSpec"):
        VoiceAgent.from_models(models=models, **options(gate(tmp_path, False, "auto")))
    monkeypatch.setattr(_runtime, "make_stt", lambda *_a: pytest.fail("client allocated"))
    with pytest.raises(ValueError, match="positive"):
        VoiceAgent.from_models(models=models, response_capacity=0, **options(gate(tmp_path, False, "auto")))


async def test_documented_nim_switch_and_restore_select_real_factory(tmp_path):
    import re
    import shutil
    import subprocess

    from xr_ai_models._riva_grpc import RivaSTT

    relative = Path("agent-samples/simple-vlm-example/yaml")
    target = tmp_path / relative
    shutil.copytree(ROOT / relative, target)
    model_backup = (target / "models.json").read_bytes()
    gate_backup = (target / "voice_gate.yaml").read_bytes()
    source = tmp_path / "models.nim.yaml"
    shutil.copy(ROOT / "deployment/nim/models.yaml", source)
    docs = (ROOT / "docs/source/guides/deploying-with-nim.md").read_text()
    code = re.search(r"python - <<'PYTHON'\n(.*?)\nPYTHON", docs, re.S).group(1)
    code = code.replace("/tmp/models.nim.yaml", str(source))
    # Execute the actual documented transformation, with only its input path redirected.
    subprocess.run([__import__("sys").executable, "-c", code], cwd=tmp_path, check=True)
    models = load_models_config(target / "models.json")
    selected = load_voice_gate_config(target / "voice_gate.yaml")
    assert selected._speaker is None
    voice = VoiceAgent.from_models(models=models, **options(selected))
    assert isinstance(voice._session.stt, RivaSTT)
    await voice._session.close()
    (target / "models.json").write_bytes(model_backup)
    (target / "voice_gate.yaml").write_bytes(gate_backup)
    restored = VoiceAgent.from_models(models=load_models_config(target / "models.json"),
                                     **options(load_voice_gate_config(target / "voice_gate.yaml")))
    assert restored._session.stt is None
    await restored._session.close()
