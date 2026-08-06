# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``riva_grpc`` STT/TTS coverage against a stubbed ``riva.client`` module.

The real ``nvidia-riva-client`` is an optional extra (``xr-ai-models[riva]``)
and needs no GPU or network here — the stub is installed in ``sys.modules``
before the deferred import in ``_riva_grpc.py`` runs.
"""
from __future__ import annotations

import io
import sys
import types
import wave

import pytest
from xr_ai_models import STTService, TTSService, load_models_config, make_stt, make_tts

# ── riva.client stub ──────────────────────────────────────────────────────


class _FakeChannel:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeAuth:
    def __init__(self, ssl_cert=None, use_ssl=False, uri="localhost:50051",
                 metadata_args=None) -> None:
        self.uri = uri
        self.use_ssl = use_ssl
        self.metadata_args = metadata_args or []
        self.channel = _FakeChannel()


class _FakeRecognitionConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeASRService:
    def __init__(self, auth) -> None:
        self.auth = auth
        self.calls: list[tuple[bytes, _FakeRecognitionConfig]] = []
        self.transcripts = ["hello ", "world"]

    def offline_recognize(self, audio, config):
        self.calls.append((audio, config))
        results = [
            types.SimpleNamespace(
                alternatives=[types.SimpleNamespace(transcript=t)],
            )
            for t in self.transcripts
        ]
        results.append(types.SimpleNamespace(alternatives=[]))
        return types.SimpleNamespace(results=results)


class _FakeSynthesisService:
    def __init__(self, auth) -> None:
        self.auth = auth
        self.calls: list[dict] = []
        self.pcm = b"\x01\x02" * 32

    def synthesize_online(self, text, voice_name=None, language_code="en-US",
                          encoding=None, sample_rate_hz=44100):
        self.calls.append({
            "text": text, "voice_name": voice_name,
            "language_code": language_code, "encoding": encoding,
            "sample_rate_hz": sample_rate_hz,
        })
        half = len(self.pcm) // 2
        yield types.SimpleNamespace(audio=self.pcm[:half])
        yield types.SimpleNamespace(audio=self.pcm[half:])


@pytest.fixture
def riva_stub(monkeypatch):
    stub = types.ModuleType("riva.client")
    stub.Auth = _FakeAuth
    stub.ASRService = _FakeASRService
    stub.SpeechSynthesisService = _FakeSynthesisService
    stub.RecognitionConfig = _FakeRecognitionConfig
    stub.AudioEncoding = types.SimpleNamespace(LINEAR_PCM=1)
    riva_pkg = types.ModuleType("riva")
    riva_pkg.client = stub
    monkeypatch.setitem(sys.modules, "riva", riva_pkg)
    monkeypatch.setitem(sys.modules, "riva.client", stub)
    return stub


def _write(tmp_path, text: str):
    p = tmp_path / "models.yaml"
    p.write_text(text)
    return p


_NIM_SPEECH_YAML = """
stt:
  kind:         riva_grpc
  category:     stt
  base_url:     grpc.nvcf.nvidia.com:443
  use_ssl:      true
  api_key_env:  NGC_API_KEY
  function_id:  "asr-func-id"
  language:     en-US
  health_check: false

tts:
  kind:         riva_grpc
  category:     tts
  base_url:     grpc.nvcf.nvidia.com:443
  use_ssl:      true
  api_key_env:  NGC_API_KEY
  function_id:  "tts-func-id"
  voice:        "Magpie-Multilingual.EN-US.Sofia"
  sample_rate:  22050
  health_check: false
"""


# ── config parsing ────────────────────────────────────────────────────────


def test_riva_spec_fields_parse(tmp_path) -> None:
    cfg = load_models_config(_write(tmp_path, _NIM_SPEECH_YAML))
    stt = cfg.stt("stt")
    assert stt.kind == "riva_grpc"
    assert stt.base_url == "grpc.nvcf.nvidia.com:443"
    assert stt.use_ssl is True
    assert stt.function_id == "asr-func-id"
    assert stt.language == "en-US"
    assert stt.health_check is False
    tts = cfg.tts("tts")
    assert tts.kind == "riva_grpc"
    assert tts.function_id == "tts-func-id"
    assert tts.voice == "Magpie-Multilingual.EN-US.Sofia"
    assert tts.sample_rate == 22050


# ── factory dispatch + auth wiring ────────────────────────────────────────


async def test_factory_builds_riva_clients_with_auth_metadata(
    tmp_path, riva_stub, monkeypatch,
) -> None:
    monkeypatch.setenv("NGC_API_KEY", "nvapi-test")
    cfg = load_models_config(_write(tmp_path, _NIM_SPEECH_YAML))
    stt = make_stt(cfg, "stt")
    tts = make_tts(cfg, "tts")
    try:
        assert isinstance(stt, STTService)
        assert isinstance(tts, TTSService)
        auth = stt._asr.auth
        assert auth.uri == "grpc.nvcf.nvidia.com:443"
        assert auth.use_ssl is True
        assert ["authorization", "Bearer nvapi-test"] in auth.metadata_args
        assert ["function-id", "asr-func-id"] in auth.metadata_args
        assert ["function-id", "tts-func-id"] in tts._tts.auth.metadata_args
    finally:
        await stt.close()
        await tts.close()


async def test_factory_raises_helpful_error_without_riva_extra(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "riva", None)
    monkeypatch.setitem(sys.modules, "riva.client", None)
    cfg = load_models_config(_write(tmp_path, _NIM_SPEECH_YAML))
    with pytest.raises(ImportError, match=r"xr-ai-models\[riva\]"):
        make_stt(cfg, "stt")


# ── STT ───────────────────────────────────────────────────────────────────


async def test_stt_transcribe_pcm_builds_config_and_joins_results(
    tmp_path, riva_stub,
) -> None:
    cfg = load_models_config(_write(tmp_path, _NIM_SPEECH_YAML))
    async with make_stt(cfg, "stt") as stt:
        text = await stt.transcribe(b"\x00\x01" * 480, sample_rate=16000, channels=1)
    assert text == "hello world"
    audio, rec_config = stt._asr.calls[0]
    assert audio == b"\x00\x01" * 480
    assert rec_config.kwargs["sample_rate_hertz"] == 16000
    assert rec_config.kwargs["audio_channel_count"] == 1
    assert rec_config.kwargs["language_code"] == "en-US"
    assert rec_config.kwargs["encoding"] == 1


async def test_stt_transcribe_wav_bytes_parses_container(
    tmp_path, riva_stub,
) -> None:
    pcm = b"\x02\x03" * 240
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(pcm)
    cfg = load_models_config(_write(tmp_path, _NIM_SPEECH_YAML))
    async with make_stt(cfg, "stt") as stt:
        await stt.transcribe(buf.getvalue())
    audio, rec_config = stt._asr.calls[0]
    assert audio == pcm
    assert rec_config.kwargs["sample_rate_hertz"] == 48000
    assert rec_config.kwargs["audio_channel_count"] == 2


async def test_stt_transcribe_rejects_non_16bit_wav(tmp_path, riva_stub) -> None:
    # An 8-bit WAV mislabeled as LINEAR_PCM (16-bit) would transcribe as
    # garbage with no error from Riva, so it must be rejected client-side.
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(16000)
        wf.writeframes(b"\x80" * 160)
    cfg = load_models_config(_write(tmp_path, _NIM_SPEECH_YAML))
    async with make_stt(cfg, "stt") as stt:
        with pytest.raises(ValueError, match="16-bit"):
            await stt.transcribe(buf.getvalue())
        assert stt._asr.calls == []


# ── TTS ───────────────────────────────────────────────────────────────────


async def test_tts_synthesize_wraps_pcm_in_wav(tmp_path, riva_stub) -> None:
    cfg = load_models_config(_write(tmp_path, _NIM_SPEECH_YAML))
    async with make_tts(cfg, "tts") as tts:
        data = await tts.synthesize("hello")
    call = tts._tts.calls[0]
    assert call["text"] == "hello"
    assert call["voice_name"] == "Magpie-Multilingual.EN-US.Sofia"
    assert call["language_code"] == "en-US"
    assert call["sample_rate_hz"] == 22050
    with wave.open(io.BytesIO(data), "rb") as wf:
        assert wf.getframerate() == 22050
        assert wf.getnchannels() == 1
        assert wf.readframes(wf.getnframes()) == tts._tts.pcm


async def test_tts_synthesize_pcm_passthrough_and_bad_format(
    tmp_path, riva_stub,
) -> None:
    cfg = load_models_config(_write(tmp_path, _NIM_SPEECH_YAML))
    async with make_tts(cfg, "tts") as tts:
        data = await tts.synthesize("hi", response_format="pcm")
        assert data == tts._tts.pcm
        with pytest.raises(ValueError, match="response_format"):
            await tts.synthesize("hi", response_format="mp3")


# ── health / lifecycle ────────────────────────────────────────────────────


async def test_health_true_when_health_check_disabled(tmp_path, riva_stub) -> None:
    cfg = load_models_config(_write(tmp_path, _NIM_SPEECH_YAML))
    async with make_stt(cfg, "stt") as stt:
        assert (await stt.health()) is True


async def test_close_closes_grpc_channel(tmp_path, riva_stub) -> None:
    cfg = load_models_config(_write(tmp_path, _NIM_SPEECH_YAML))
    stt = make_stt(cfg, "stt")
    await stt.close()
    assert stt._auth.channel.closed is True


_LOCAL_RIVA_YAML = """
stt:
  kind:     riva_grpc
  category: stt
  base_url: localhost:50051
"""


def _grpc_stub(ready: bool) -> types.ModuleType:
    class _Future:
        def result(self, timeout=None):
            if not ready:
                raise TimeoutError("channel not ready")

    stub = types.ModuleType("grpc")
    stub.channel_ready_future = lambda channel: _Future()
    return stub


async def test_health_probes_channel_when_enabled(
    tmp_path, riva_stub, monkeypatch,
) -> None:
    cfg = load_models_config(_write(tmp_path, _LOCAL_RIVA_YAML))
    async with make_stt(cfg, "stt") as stt:
        monkeypatch.setitem(sys.modules, "grpc", _grpc_stub(ready=True))
        assert (await stt.health()) is True
        monkeypatch.setitem(sys.modules, "grpc", _grpc_stub(ready=False))
        assert (await stt.health()) is False


# ── cleartext-key warning ─────────────────────────────────────────────────


def _capture_warnings():
    from loguru import logger

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    return logger, handler_id, messages


_CLEARTEXT_YAML = """
stt:
  kind:         riva_grpc
  category:     stt
  base_url:     {base_url}
  use_ssl:      false
  api_key_env:  TEST_RIVA_KEY
  health_check: false
"""


async def test_cleartext_key_to_remote_host_warns(
    tmp_path, riva_stub, monkeypatch,
) -> None:
    monkeypatch.setenv("TEST_RIVA_KEY", "sekrit")
    cfg = load_models_config(_write(
        tmp_path, _CLEARTEXT_YAML.format(base_url="riva.example.com:50051"),
    ))
    logger, handler_id, messages = _capture_warnings()
    try:
        async with make_stt(cfg, "stt"):
            pass
    finally:
        logger.remove(handler_id)
    assert any("cleartext" in m for m in messages)


async def test_cleartext_key_to_loopback_does_not_warn(
    tmp_path, riva_stub, monkeypatch,
) -> None:
    monkeypatch.setenv("TEST_RIVA_KEY", "sekrit")
    cfg = load_models_config(_write(
        tmp_path, _CLEARTEXT_YAML.format(base_url="localhost:50051"),
    ))
    logger, handler_id, messages = _capture_warnings()
    try:
        async with make_stt(cfg, "stt"):
            pass
    finally:
        logger.remove(handler_id)
    assert not any("cleartext" in m for m in messages)
