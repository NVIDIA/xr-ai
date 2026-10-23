# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configured voice construction, real adapters, and executable NIM selection."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from xr_ai_interaction import UserQuery
from xr_ai_models import ModelsConfig, load_models_config, load_models_config_from_dict
from xr_ai_models._openai_compat import OpenAICompatSTT
from xr_ai_runtime import Topic
from xr_ai_voice import VoiceAgent, _runtime, _session
from xr_ai_voice._processors import VadConfig
from xr_ai_voicegate import load_voice_gate_config

ROOT = Path(__file__).resolve().parents[1]


def test_factory_preserves_constructor_option_signature():
    import inspect

    constructor = dict(inspect.signature(VoiceAgent.__init__).parameters)
    factory = dict(inspect.signature(VoiceAgent.from_models).parameters)
    constructor.pop("self")
    constructor.pop("stt")
    factory.pop("models")
    assert list(factory) == list(constructor)
    assert factory == constructor
    assert inspect.signature(VoiceAgent.__init__).return_annotation == "None"
    assert inspect.signature(VoiceAgent.from_models).return_annotation == "VoiceAgent"


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


@pytest.mark.parametrize("sample, model_file, preset", [
    ("simple-vlm-example", "models.json", "voice_gate.yaml"),
    ("tea-making-sample", "models.local.json", "voice_gate.yaml"),
])
async def test_documented_nim_switch_preserves_first_backup_and_restores_active_preset(
    tmp_path, sample, model_file, preset,
):
    import json
    import re
    import shutil
    import subprocess
    import sys

    from xr_ai_models._riva_grpc import RivaSTT

    relative = Path("agent-samples") / sample / "yaml"
    target = tmp_path / relative
    shutil.copytree(ROOT / relative, target)
    local_models = json.loads((target / model_file).read_text())
    local_models["models"]["vlm"]["endpoint"]["base_url"] = "http://localhost:9990"
    (target / model_file).write_text(json.dumps(local_models))
    local_gate = yaml.safe_load((target / preset).read_text())
    local_gate["followup_grace_s"] = 7.25
    (target / preset).write_text(yaml.safe_dump(local_gate))
    model_original = (target / model_file).read_bytes()
    gate_original = (target / preset).read_bytes()
    source = tmp_path / "models.nim.yaml"
    shutil.copy(ROOT / "deployment/nim/models.yaml", source)
    backup_dir = tmp_path / "local-backup"
    docs = (ROOT / "docs/source/guides/deploying-with-nim.md").read_text()
    code = re.search(r"python - <<'PYTHON'\n(.*?)\nPYTHON", docs, re.S).group(1)
    restore = re.search(r"```bash\n(cp /tmp/simple-vlm.local-backup/.*?)\nuv run", docs, re.S).group(1)
    def adapt(text):
        return (text.replace("/tmp/models.nim.yaml", str(source))
                .replace("/tmp/simple-vlm.local-backup", str(backup_dir))
                .replace("agent-samples/simple-vlm-example/yaml/models.json", str(relative / model_file))
                .replace("agent-samples/simple-vlm-example/yaml/voice_gate.yaml", str(relative / preset))
                .replace(str(backup_dir / "models.json"), str(backup_dir / model_file))
                .replace(str(backup_dir / "voice_gate.yaml"), str(backup_dir / preset)))
    code, restore = adapt(code), adapt(restore)
    # Run the actual recipe, adapting only sample selection and temporary paths.
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
    models = load_models_config(target / model_file)
    selected = load_voice_gate_config(target / preset)
    assert selected._speaker is None
    voice = VoiceAgent.from_models(models=models, **options(selected))
    assert isinstance(voice._session.stt, RivaSTT)
    await voice._session.close()
    switched_models = (target / model_file).read_bytes()
    switched_gate = (target / preset).read_bytes()
    repeated = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True)
    assert repeated.returncode != 0 and b"FileExistsError" in repeated.stderr
    assert (backup_dir / model_file).read_bytes() == model_original
    assert (backup_dir / preset).read_bytes() == gate_original
    assert (target / model_file).read_bytes() == switched_models
    assert (target / preset).read_bytes() == switched_gate
    # Execute the documented restore copies, without starting inference services.
    subprocess.run(["bash", "-eu", "-c", restore], cwd=tmp_path, check=True)
    assert (target / model_file).read_bytes() == model_original
    assert (target / preset).read_bytes() == gate_original
    restored = VoiceAgent.from_models(models=load_models_config(target / model_file),
                                     **options(load_voice_gate_config(target / preset)))
    assert restored._session.stt is None
    await restored._session.close()
