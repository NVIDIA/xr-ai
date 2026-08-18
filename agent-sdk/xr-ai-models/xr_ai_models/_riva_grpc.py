# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Riva gRPC clients for the STT and TTS service protocols.

Hosted NIM speech (build.nvidia.com) speaks Riva's gRPC API (not OpenAI
``/v1/audio``) at ``grpc.nvcf.nvidia.com:443``, selecting the model by
NVCF ``function-id`` metadata. A self-hosted Riva/NIM speech container is
the same API on a local port with no function id. Requires the ``riva``
extra (``nvidia-riva-client``); the import is deferred so the base install
stays gRPC-free.

The ``riva.client`` SDK is synchronous, so calls run in a worker thread
via ``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
import io
import os
import wave
from typing import Any

from loguru import logger

from ._openai_compat import _pcm_to_wav

_LOOPBACK_PREFIXES = ("localhost:", "127.0.0.1:", "[::1]:")


def _import_riva() -> Any:
    try:
        import riva.client as riva_client
    except ImportError as exc:
        raise ImportError(
            "kind: riva_grpc requires the optional 'riva' extra: "
            "install xr-ai-models[riva] (nvidia-riva-client)"
        ) from exc
    return riva_client


def _make_auth(
    riva_client: Any,
    base_url: str,
    *,
    use_ssl: bool,
    api_key: str | None,
    function_id: str | None,
) -> Any:
    if api_key and not use_ssl and not base_url.startswith(_LOOPBACK_PREFIXES):
        logger.warning(
            "API key set for {!r}: a bearer token over an unencrypted gRPC "
            "channel to a non-loopback host is sent cleartext. Set "
            "use_ssl: true for remote endpoints.",
            base_url,
        )
    metadata: list[list[str]] = []
    if api_key:
        metadata.append(["authorization", f"Bearer {api_key}"])
    if function_id:
        metadata.append(["function-id", function_id])
    return riva_client.Auth(uri=base_url, use_ssl=use_ssl, metadata_args=metadata)


def _parse_wav(data: bytes) -> tuple[int, int, bytes]:
    """Split a WAV container into (sample_rate, channels, raw PCM frames).

    Only 16-bit PCM is accepted: the frames go to Riva labelled LINEAR_PCM
    (signed 16-bit), so any other sample width would transcribe as garbage
    with no error from the service.
    """
    with wave.open(io.BytesIO(data), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(
                f"riva_grpc requires 16-bit PCM WAV audio, got sample width "
                f"{wf.getsampwidth() * 8} bits"
            )
        return wf.getframerate(), wf.getnchannels(), wf.readframes(wf.getnframes())


async def _channel_ready(auth: Any, enabled: bool) -> bool:
    # Hosted NVCF exposes no health surface; the spec sets health_check=false
    # and readiness is assumed. Self-hosted Riva answers a channel-ready probe.
    if not enabled:
        return True

    def probe() -> bool:
        import grpc
        try:
            grpc.channel_ready_future(auth.channel).result(timeout=3.0)
            return True
        except Exception:
            return False

    return await asyncio.to_thread(probe)


class RivaSTT:
    """Riva ASR client: offline (batch) recognition over gRPC."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key_env: str | None = None,
        function_id: str | None = None,
        use_ssl: bool = False,
        language: str = "en-US",
        timeout: float = 30.0,
        health_check: bool = True,
    ) -> None:
        self._rc = _import_riva()
        self._language = language
        self._timeout = timeout
        self._health_check = health_check
        api_key = os.environ.get(api_key_env) if api_key_env else None
        self._auth = _make_auth(
            self._rc, base_url,
            use_ssl=use_ssl, api_key=api_key, function_id=function_id,
        )
        self._asr = self._rc.ASRService(self._auth)

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int | None = None,
        channels: int = 1,
        timeout: float | None = None,
    ) -> str:
        """Recognize *audio* (WAV or raw 16-bit PCM) and return the transcript."""
        if sample_rate is None:
            sample_rate, channels, audio = _parse_wav(audio)
        config = self._rc.RecognitionConfig(
            encoding=self._rc.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=sample_rate,
            language_code=self._language,
            audio_channel_count=channels,
            max_alternatives=1,
        )
        response = await asyncio.wait_for(
            asyncio.to_thread(self._asr.offline_recognize, audio, config),
            timeout or self._timeout,
        )
        return "".join(
            r.alternatives[0].transcript
            for r in response.results
            if r.alternatives
        )

    async def health(self) -> bool:
        """Whether the Riva gRPC channel is ready (assumed when probing is off)."""
        return await _channel_ready(self._auth, self._health_check)

    async def close(self) -> None:
        """Close the underlying gRPC channel."""
        self._auth.channel.close()

    async def __aenter__(self) -> "RivaSTT":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


class RivaTTS:
    """Riva TTS client: streaming synthesis over gRPC, concatenated to one
    WAV (or raw PCM) result."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key_env: str | None = None,
        function_id: str | None = None,
        use_ssl: bool = False,
        voice: str = "",
        language: str = "en-US",
        sample_rate: int = 44100,
        timeout: float = 30.0,
        health_check: bool = True,
    ) -> None:
        self._rc = _import_riva()
        self._voice = voice
        self._language = language
        self._sample_rate = sample_rate
        self._timeout = timeout
        self._health_check = health_check
        api_key = os.environ.get(api_key_env) if api_key_env else None
        self._auth = _make_auth(
            self._rc, base_url,
            use_ssl=use_ssl, api_key=api_key, function_id=function_id,
        )
        self._tts = self._rc.SpeechSynthesisService(self._auth)

    async def synthesize(
        self,
        text: str,
        *,
        response_format: str = "wav",
        timeout: float | None = None,
    ) -> bytes:
        """Synthesize *text* and return WAV bytes (or raw 16-bit PCM)."""
        if response_format not in ("wav", "pcm"):
            raise ValueError(
                f"riva_grpc TTS supports response_format 'wav' or 'pcm', "
                f"got {response_format!r}"
            )
        # Streaming synthesis, concatenated: the batch Synthesize RPC
        # segfaults some Riva NIM builds (magpie-tts-multilingual) after
        # producing its response; synthesize_online works on hosted NVCF and
        # self-hosted NIMs alike.
        def _collect() -> bytes:
            return b"".join(
                resp.audio
                for resp in self._tts.synthesize_online(
                    text,
                    voice_name=self._voice,
                    language_code=self._language,
                    encoding=self._rc.AudioEncoding.LINEAR_PCM,
                    sample_rate_hz=self._sample_rate,
                )
            )

        pcm = await asyncio.wait_for(
            asyncio.to_thread(_collect), timeout or self._timeout,
        )
        if response_format == "pcm":
            return pcm
        return _pcm_to_wav(pcm, self._sample_rate, 1)

    async def health(self) -> bool:
        """Whether the Riva gRPC channel is ready (assumed when probing is off)."""
        return await _channel_ready(self._auth, self._health_check)

    async def close(self) -> None:
        """Close the underlying gRPC channel."""
        self._auth.channel.close()

    async def __aenter__(self) -> "RivaTTS":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
