# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech/noise power, setup baselines and capture notification integration."""


# ruff: noqa: F811
import asyncio
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_procedure_capture_guide import announcements, demo, joined, speak  # noqa: F401
from test_procedure_capture_quality import Scene, install_quality
from xr_ai_voice import VoiceAudioUtterance, VoiceAudioWindow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide/worker"))
from procedure_capture_guide_worker.capture.audio_quality import AudioQuality  # noqa: E402
from procedure_capture_guide_worker.capture.quality import QualityConfig, QualityFeedback  # noqa: E402


class AudioScene:
    def __init__(self, config=None):
        self.now = 1.0
        self.config = config or QualityConfig()
        self.audio = AudioQuality(self.config, clock=lambda: self.now)
        self.observer = self.audio.observe
        self.finish = self.audio.finish

    def windows(self, count, power, speech):
        for _ in range(count):
            self.observer(VoiceAudioWindow("user", round(self.now * 1e6), 0.032, power, speech))
            self.now += 0.032

    def utterance(self, ratio=100, *, count=32, noise=0.000001):
        self.windows(40, noise, False)
        onset = round(self.now * 1e6)
        self.windows(count, noise * (1 + ratio), True)
        self.windows(26, noise, False)
        self.finish(VoiceAudioUtterance("user", onset + 128_000))
        return onset + 128_000  # Accepted timestamp follows the VAD minimum speech delay.


def test_setup_snr_uses_power_and_relative_threshold_not_half_the_db_value():
    scene = AudioScene()
    scene.audio.accept(scene.utterance(), setup=True)
    low, metrics = scene.audio.evaluate()
    assert not low
    assert metrics["baseline_audio_snr_db"] == pytest.approx(20)
    for _ in range(3):
        scene.audio.accept(scene.utterance(ratio=49))
    low, metrics = scene.audio.evaluate()
    assert low
    assert metrics["audio_snr_db"] == pytest.approx(10 * math.log10(49))
    assert metrics["audio_snr_drop_percent"] == pytest.approx(51)
    assert metrics["audio_snr_drop_db"] == pytest.approx(3.098, abs=0.001)
    scene.audio.accept(scene.utterance(ratio=80))
    assert not scene.audio.evaluate()[0]


@pytest.mark.parametrize("ratio,warns", [(51, False), (50, True), (49, True)])
def test_exact_half_linear_snr_threshold(ratio, warns):
    scene = AudioScene()
    scene.audio.accept(scene.utterance(), setup=True)
    for _ in range(3):
        scene.audio.accept(scene.utterance(ratio=ratio))
    assert scene.audio.evaluate()[0] == warns


def test_audio_needs_accepted_speech_and_never_uses_silence_as_zero_snr():
    scene = AudioScene()
    scene.audio.accept(scene.utterance(), setup=True)
    scene.utterance(ratio=10, count=128)  # Other speech / TTS has no accepted query.
    assert not scene.audio.evaluate()[0]
    scene.audio.accept(scene.utterance(ratio=10))
    assert not scene.audio.evaluate()[0]  # Less than three seconds of accepted speech.
    scene.now += scene.config.audio_max_age_s + 1
    low, metrics = scene.audio.evaluate()
    assert not low and metrics["audio_snr_db"] is None
    scene.audio.accept(scene.utterance(ratio=10))
    assert not scene.audio.evaluate()[0]


@pytest.mark.parametrize("case", ["zero_noise", "missing_noise", "too_short", "no_audio"])
def test_unreliable_setup_does_not_establish_a_baseline(case):
    scene = AudioScene()
    if case == "zero_noise":
        stamp = scene.utterance(noise=0)
    elif case == "too_short":
        stamp = scene.utterance(count=2)
    elif case == "missing_noise":
        stamp = round(scene.now * 1e6)
        scene.windows(32, 0.001, True)
        scene.windows(26, 0.000001, False)
    else:
        stamp = 123
    scene.audio.accept(stamp, setup=True)
    assert scene.audio.baseline is None
    assert not scene.audio.evaluate()[0]


def test_thresholds_are_independent_and_configurable():
    scene = Scene()
    scene.state.config = QualityConfig(frame_rate_drop_ratio=0.25, resolution_drop_ratio=0.75)
    for _ in range(12):
        scene.second()
    for _ in range(12):
        conditions = scene.second(fps=10, width=960, height=720)
    assert conditions == {"low_resolution"}
    audio = AudioScene(QualityConfig(audio_snr_drop_ratio=0.25))
    audio.audio.accept(audio.utterance(), setup=True)
    for _ in range(3):
        audio.audio.accept(audio.utterance(ratio=40))
    assert not audio.audio.evaluate()[0]


async def test_setup_and_narration_use_real_agent_wiring_and_mic_change_resets(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    quality = demo.capture._connections["user"].quality
    scene = AudioScene()
    quality.clock = quality.audio.clock = lambda: scene.now
    scene.observer = demo.capture.on_audio_window
    scene.finish = demo.capture.on_audio_utterance
    await demo.capture.on_quality_audio(SimpleNamespace(
        participant_id="user", track_id="mic1", sample_rate=16000, channels=1, pts_us=round(scene.now * 1e6),
    ))
    await speak(demo, "check setup", timestamp_us=scene.utterance())
    assert "Estimated audio signal-to-noise ratio is 20 decibels." in announcements(demo)[-1]
    await speak(demo, "start capture", timestamp_us=round(scene.now * 1e6))
    assert quality.audio.baseline == pytest.approx(100)
    for _ in range(3):
        await speak(demo, "Narration", timestamp_us=scene.utterance(ratio=40))
    async with asyncio.timeout(2):
        while not any("a decrease of 4 decibels" in text for text in announcements(demo)):
            await asyncio.sleep(0.005)
    assert demo.capture_backend.is_capturing("user")
    await demo.capture.on_quality_audio(SimpleNamespace(
        participant_id="user", track_id="mic2", sample_rate=16000, channels=1, pts_us=round(scene.now * 1e6),
    ))
    assert quality.audio.baseline is None
    assert not quality.audio.evaluate()[0]
    await demo.capture.on_quality_audio(SimpleNamespace(
        participant_id="user", track_id="mic1", sample_rate=16000, channels=1, pts_us=1,
    ))
    assert quality.audio_track[0] == "mic2"
    await speak(demo, "stop capture", timestamp_us=round(scene.now * 1e6))
    assert quality.task is None


async def test_audio_warning_combines_with_video_and_repeats_without_accumulating():
    scene = AudioScene()
    monitor = QualityFeedback(
        "user", SimpleNamespace(capture_quality=AsyncMock()), scene.config, clock=lambda: scene.now,
    )
    monitor.audio = scene.audio
    monitor.audio.accept(scene.utterance(), setup=True)
    ticks, resume = asyncio.Queue(), asyncio.Queue()
    messages = []

    async def sleep(_):
        ticks.put_nowait(True)
        await resume.get()

    async def publish(_topic, message, **_):
        messages.append(message)

    monitor.sleep = sleep
    monitor.start(SimpleNamespace(publish=publish))
    await ticks.get()
    for _ in range(3):
        monitor.audio.accept(scene.utterance(ratio=40))
    monitor.state.evaluate = lambda: {"low_frame_rate", "low_resolution"}
    try:
        resume.put_nowait(True)
        await ticks.get()
        updates = [m.text for m in messages if m.kind == "update"]
        assert len(updates) == 1
        assert all(word in updates[0] for word in ("frame rate", "resolution", "signal-to-noise ratio"))
        assert "at least 50 percent since setup, a decrease of 4 decibels" in updates[0]
        scene.now += 10
        resume.put_nowait(True)
        await ticks.get()
        assert [m.kind for m in messages] == ["update", "withdraw", "update"]
    finally:
        await monitor.stop()


@pytest.mark.parametrize("field", ["frame_rate_drop_ratio", "resolution_drop_ratio", "audio_snr_drop_ratio"])
@pytest.mark.parametrize("value", [0, 1, -0.1, float("nan"), float("inf")])
def test_ratio_thresholds_reject_invalid_values(field, value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QualityConfig.model_validate({field: value})


async def test_audio_baselines_are_per_participant_and_reconnect_clears_them(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    await demo.capture.receive(joined("bob", "bob-session"))
    scene = AudioScene()
    quality = demo.capture._connections["user"].quality
    quality.audio.clock = lambda: scene.now
    scene.observer = demo.capture.on_audio_window
    scene.finish = demo.capture.on_audio_utterance
    await speak(demo, "check setup", timestamp_us=scene.utterance())
    assert quality.audio.baseline == pytest.approx(100)
    assert demo.capture._connections["bob"].quality.audio.baseline is None
    await demo.capture.receive(joined(session="new-session", pts_us=round(scene.now * 1e6)))
    assert demo.capture._connections["user"].quality.audio.baseline is None


@pytest.fixture
def amplitude_vad(monkeypatch):
    from xr_ai_vad import detector

    class Silero:
        def __call__(self, tensor, sample_rate):
            return 0.9 if tensor.abs().max().item() > 0.01 else 0.0

        def reset_states(self):
            pass

    monkeypatch.setattr(detector, "load_silero_vad", lambda **kwargs: Silero())


async def test_long_narration_snr_is_ready_before_real_vad_transcription(amplitude_vad):
    import numpy as np
    from pipecat.frames.frames import InputAudioRawFrame
    from test_voice_pipeline import _FakeStt, _run_chain
    from xr_ai_voice._processors.vad_stt import VadConfig, VadSttProcessor

    scene = AudioScene()
    scene.audio.baseline = 1000
    accepted = []

    async def accept(pid, text, timestamp_us):
        scene.audio.accept(timestamp_us)
        accepted.append(scene.audio.evaluate())

    proc = VadSttProcessor(stt=_FakeStt(text="narration"),
                          vad_cfg=VadConfig(stop_probe_after_s=0, on_audio_window=scene.audio.observe,
                                            on_audio_utterance=scene.audio.finish),
                          on_final_transcript=accept)
    frames = []
    # The real VAD's 30-second limit includes its ten pre-roll windows.
    for i in range(1000):
        frame = InputAudioRawFrame(audio=np.full(512, 1638 if i >= 40 else 163, np.int16).tobytes(),
                                   sample_rate=16000, num_channels=1)
        frame.transport_source = "user"
        frame.pts = 1_000_000_000 + i * 32_000_000
        frames.append(frame)
    await _run_chain(proc, sends=frames)
    assert len(accepted) == 1
    assert accepted[0][0]  # Accepted speech must already have its low-SNR measurement.
    assert accepted[0][1]["audio_snr_db"] == pytest.approx(20, abs=.1)


async def test_setup_after_mid_utterance_mic_change_uses_only_its_own_pcm(demo, amplitude_vad, monkeypatch):
    import numpy as np
    from pipecat.frames.frames import InputAudioRawFrame
    from test_voice_pipeline import _FakeStt
    from xr_ai_voice._processors.vad_stt import VadConfig, VadSttProcessor

    install_quality(demo)
    await demo.capture.receive(joined())
    baselines = []

    async def accept(pid, text, timestamp_us):
        await speak(demo, text, pid=pid, timestamp_us=timestamp_us)
        baselines.append(demo.capture._connections[pid].quality.audio.baseline)

    proc = VadSttProcessor(
        stt=_FakeStt(text="check setup"),
        vad_cfg=VadConfig(stop_probe_after_s=0, on_audio_window=demo.capture.on_audio_window,
                          on_audio_utterance=demo.capture.on_audio_utterance),
        on_final_transcript=accept,
    )
    monkeypatch.setattr(proc, "push_frame", AsyncMock())
    pts_us = 1_000_000
    # The 736 ms pause after switching microphones is shorter than VAD finalization.
    # Both sides of that pause therefore belong to the first utterance.
    for track, count, amplitude in [
        ("mic1", 40, 160), ("mic1", 20, 1600),
        ("mic2", 23, 160), ("mic2", 30, 5000), ("mic2", 40, 160),
        ("mic2", 30, 480), ("mic2", 30, 160),
    ]:
        chunk = SimpleNamespace(participant_id="user", track_id=track, sample_rate=16000, channels=1)
        pcm = np.full(512, amplitude, np.int16).tobytes()
        for _ in range(count):
            chunk.pts_us = pts_us
            await demo.capture.on_quality_audio(chunk)
            frame = InputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)
            frame.transport_source, frame.pts = "user", pts_us * 1000
            await proc._handle_audio(frame)
            pts_us += 32_000

    assert baselines == [None, pytest.approx(8)]  # (480 / 160) squared, minus noise power
    assert "9 decibels" in announcements(demo)[-1]
