# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local, hash-pinned audio and generated scenes for sample replays."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import wave
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import resample_poly

from ._scenario import _ROOT, _Scenario


def _local_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    ignore = path / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n")


def _audio_path(directory: Path, text: str) -> Path:
    return directory / (hashlib.sha256(text.encode()).hexdigest() + ".wav")


def _decode_audio(raw: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(raw)) as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() not in (1, 2) or wav.getcomptype() != "NONE":
            raise ValueError("fixtures must be uncompressed mono/stereo 16-bit PCM WAV")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
        samples = samples.reshape(-1, wav.getnchannels()).mean(axis=1)
        rate = wav.getframerate()
    if not len(samples) or not 1 <= rate <= 192000:
        raise ValueError("fixture must contain audio at a supported sample rate")
    divisor = math.gcd(rate, 16000)
    return resample_poly(samples, 16000 // divisor, rate // divisor).astype("<f4")


def _audio_manifest(directory: Path) -> dict:
    path = directory / "audio.json"
    if not path.exists():
        return {"version": 1, "files": {}}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) != {"version", "files"} or data["version"] != 1:
        raise ValueError("invalid audio.json manifest")
    if not isinstance(data["files"], dict):
        raise ValueError("audio.json files must map utterance text to WAV SHA256")
    return data


async def _prepare(scenario: _Scenario, directory: Path, models_path: Path) -> None:
    from xr_ai_models import load_models_config, make_tts

    _local_directory(directory)
    manifest = _audio_manifest(directory)
    tts = make_tts(load_models_config(models_path), "tts")
    try:
        for text in scenario.texts:
            path = _audio_path(directory, text)
            expected = manifest["files"].get(text)
            if expected is not None and not path.exists():
                raise ValueError(f"pinned audio is missing: {path}; use a new fixture directory")
            if not path.exists():
                raw = await tts.synthesize(text, timeout=120)
                _decode_audio(raw)
                path.write_bytes(raw)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected is not None and digest != expected:
                raise ValueError(f"pinned audio changed: {path}")
            _decode_audio(path.read_bytes())
            manifest["files"][text] = digest
        (directory / "audio.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        await tts.close()


def _load_audio(scenario: _Scenario, directory: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    manifest = _audio_manifest(directory)
    audio, hashes = {}, {}
    for index, event in enumerate(scenario.events):
        text = event.get("say")
        if text is None:
            continue
        path = _audio_path(directory, text)
        expected = manifest["files"].get(text)
        if expected is None or not path.exists():
            raise ValueError(f"missing pinned audio for {text!r}; run prepare first")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise ValueError(f"audio hash mismatch: {path}")
        audio[text], hashes[text] = _decode_audio(raw), digest
        boundary = next(
            (other["at_s"] for other in scenario.events[index + 1 :] if "say" in other),
            scenario.duration_s,
        )
        if event["at_s"] + len(audio[text]) / 16000 > boundary:
            raise ValueError(f"utterance overlaps the next input or replay end: {text!r}")
    return audio, hashes


def _renderer():
    path = _ROOT / "agent-samples/procedure-capture-guide/example-guides/tea-making/evaluation/_scene_rendering.py"
    spec = importlib.util.spec_from_file_location("_sample_eval_scenes", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scene renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_scene


def _load_images(scenario: _Scenario, bindings: dict[str, Path]) -> tuple[dict[str, Image.Image], dict[str, str]]:
    if set(bindings) != scenario.images:
        raise ValueError(f"image bindings must be exactly {sorted(scenario.images)}")
    images, hashes = {}, {}
    for name, path in bindings.items():
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
        image.thumbnail((1280, 1280))
        images[name], hashes[name] = image, hashlib.sha256(raw).hexdigest()
    return images, hashes
