# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer composition retains only explicit warmups and capability probes."""
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from xr_ai_voicegate import VoiceGateConfig


@pytest.mark.parametrize("sample, package, expected_probes", [
    ("simple-vlm-example", "simple_vlm_example_worker", {"vlm"}),
    ("lab-instrument-monitoring", "lab_instrument_monitoring_worker", set()),
    ("tea-making-sample", "tea_making_worker", {"rag"}),
    ("xr-render-demo", "xr_render_demo_worker", {"agent-llm"}),
])
async def test_sample_does_not_register_model_health_probes(
    sample, package, expected_probes, monkeypatch, tmp_path,
):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "agent-samples" / sample / "worker"))
    monkeypatch.syspath_prepend(str(root / "agent-samples/xr-render-demo/scene"))
    app = importlib.import_module(f"{package}.app")
    service = SimpleNamespace(
        health=AsyncMock(side_effect=AssertionError("unexpected model health probe")),
        chat=AsyncMock(),
    )
    for factory in ("make_llm", "make_vlm", "make_stt", "make_tts"):
        if hasattr(app, factory):
            monkeypatch.setattr(app, factory, lambda *_args: service)
    monkeypatch.setattr(app, "setup_logging", Mock())
    monkeypatch.setattr(app, "load_models_config", Mock(return_value=object()))
    monkeypatch.setattr(app, "load_voice_gate_config", Mock(return_value=VoiceGateConfig()))
    monkeypatch.setattr(app, "HubVoiceTransport", Mock())
    config = SimpleNamespace(
        models_config=tmp_path / "models.json", voice_gate_yaml=tmp_path / "gate.yaml",
        silence_duration=0.5, min_speech=0.2, silero_threshold=0.5, idle_timeout_secs=None,
        rag_endpoint="unused", scene_endpoint="unused", openxr_endpoint="unused",
        text_memory_dir=tmp_path, video_history_enabled=False,
    )
    rag = SimpleNamespace(health=AsyncMock(return_value=True))
    if sample == "tea-making-sample":
        monkeypatch.setattr(app, "RAGTools", Mock(return_value=rag))
    if sample == "simple-vlm-example":
        monkeypatch.setattr(app, "_warm_vlm", AsyncMock(return_value=True))
    if sample == "xr-render-demo":
        for constructor in (
            "SceneTools", "TrackingTools", "TextMemoryTools", "ImageRegistry",
            "CurrentFrameTool", "SceneSupervisor", "RenderAgent",
        ):
            monkeypatch.setattr(app, constructor, Mock())
        monkeypatch.setattr(app, "close_clients", AsyncMock())

    class WiringComplete(Exception):
        pass

    options = {}

    def capture_voice(**kwargs):
        options.update(kwargs)
        raise WiringComplete

    monkeypatch.setattr(app, "VoiceAgent", SimpleNamespace(from_models=capture_voice))
    with pytest.raises(WiringComplete):
        await app.run_app(config)
    probes = options.get("probes", {})
    assert set(probes) == expected_probes
    for probe in probes.values():
        assert await probe() is True
    service.health.assert_not_awaited()
    if sample == "xr-render-demo":
        service.chat.assert_awaited_once()
        assert service.chat.call_args.kwargs["tools"][0].name == "warmup_noop"
        service.chat.side_effect = RuntimeError("still loading")
        assert await probes["agent-llm"]() is False
    if sample == "tea-making-sample":
        rag.health.assert_awaited_once()


@pytest.mark.parametrize("sample, package, preset", [
    (sample, package, preset.name)
    for sample, package in (
        ("simple-vlm-example", "simple_vlm_example_worker"),
        ("lab-instrument-monitoring", "lab_instrument_monitoring_worker"),
        ("tea-making-sample", "tea_making_worker"),
        ("xr-render-demo", "xr_render_demo_worker"),
    )
    for preset in sorted(
        (Path(__file__).resolve().parents[1] / "agent-samples" / sample / "yaml").glob("voice_gate*.yaml")
    )
])
@pytest.mark.parametrize("optional_batch_entry", [False, True])
async def test_shipped_speaker_mode_never_constructs_batch_client(
    sample, package, preset, optional_batch_entry, monkeypatch, tmp_path,
):
    from xr_ai_models import load_models_config
    from xr_ai_voice import VoiceAgent, _runtime

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "agent-samples" / sample / "worker"))
    monkeypatch.syspath_prepend(str(root / "agent-samples/xr-render-demo/scene"))
    app = importlib.import_module(f"{package}.app")
    model_file = "models.local.json" if sample == "tea-making-sample" else "models.json"
    profile = load_models_config(root / "agent-samples" / sample / "yaml" / model_file)
    assert "stt" not in profile.entries
    if optional_batch_entry:
        profile.entries["stt"] = load_models_config(root / "deployment/nim/models.yaml").entries["stt"]
    monkeypatch.setattr(app, "load_models_config", lambda *_args: profile)
    monkeypatch.setattr(app, "setup_logging", Mock())
    monkeypatch.setattr(_runtime, "make_stt", lambda *_args: pytest.fail("unused batch client constructed"))
    service = SimpleNamespace(chat=AsyncMock(), close=AsyncMock())
    for factory in ("make_llm", "make_vlm", "make_tts"):
        if hasattr(app, factory):
            monkeypatch.setattr(app, factory, lambda *_args: service)
    monkeypatch.setattr(app, "HubVoiceTransport", Mock())
    if sample == "tea-making-sample":
        monkeypatch.setattr(app, "RAGTools", Mock())
    if sample == "xr-render-demo":
        for constructor in ("SceneTools", "TrackingTools", "TextMemoryTools", "ImageRegistry",
                            "CurrentFrameTool", "SceneSupervisor", "RenderAgent"):
            monkeypatch.setattr(app, constructor, Mock())
        monkeypatch.setattr(app, "close_clients", AsyncMock())

    class WiringComplete(Exception):
        pass

    class CaptureVoice(VoiceAgent):
        def __init__(self, **kwargs):
            assert kwargs["stt"] is None
            assert kwargs["voice_gate"]._speaker.backend == "required"
            raise WiringComplete

    monkeypatch.setattr(app, "VoiceAgent", CaptureVoice)
    config = SimpleNamespace(
        models_config=tmp_path / model_file,
        voice_gate_yaml=root / "agent-samples" / sample / "yaml" / preset,
        silence_duration=0.5, min_speech=0.2, silero_threshold=0.5, idle_timeout_secs=None,
        rag_endpoint="unused", scene_endpoint="unused", openxr_endpoint="unused",
        text_memory_dir=tmp_path, video_history_enabled=False,
    )
    with pytest.raises(WiringComplete):
        await app.run_app(config)
