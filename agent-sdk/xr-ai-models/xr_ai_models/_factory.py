# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``make_*`` constructors that dispatch a :class:`Spec` to a concrete client."""
from __future__ import annotations

from ._config import (
    KIND_NVIDIA_OCR,
    KIND_OPENAI_COMPAT,
    KIND_RIVA_GRPC,
    ModelsConfig,
)
from ._ocr import VLMOCR, NvidiaOCR
from ._openai_compat import (
    OpenAICompatEmbedding,
    OpenAICompatLLM,
    OpenAICompatSTT,
    OpenAICompatTTS,
    OpenAICompatVLM,
)
from ._protocols import (
    Capabilities,
    EmbeddingService,
    LLMService,
    OCRService,
    STTService,
    TTSService,
    VLMService,
)


def make_embedding(config: ModelsConfig, name: str) -> EmbeddingService:
    """Construct the embedding service for the named configuration entry.

    Raises :class:`KeyError` when *name* is absent, :class:`TypeError` when it
    names a different model role, and :class:`ValueError` for an unsupported
    adapter kind.
    """

    spec = config.embedding(name)
    if spec.kind == KIND_OPENAI_COMPAT:
        return OpenAICompatEmbedding(
            base_url=spec.base_url,
            model_name=spec.model_name,
            api_key_env=spec.api_key_env,
            timeout=spec.timeout,
            health_check=spec.health_check,
            health_path=spec.health_path,
        )
    raise ValueError(f"unsupported embedding kind: {spec.kind!r}")


def make_llm(config: ModelsConfig, name: str) -> LLMService:
    """Construct the text chat service for the named configuration entry.

    Raises :class:`KeyError` when *name* is absent, :class:`TypeError` when it
    names a different model role, and :class:`ValueError` for an unsupported
    adapter kind.
    """

    spec = config.llm(name)
    adapter = spec.adapter
    endpoint = spec.endpoint
    if adapter.kind == KIND_OPENAI_COMPAT:
        return OpenAICompatLLM(
            base_url=endpoint.base_url,
            model_name=adapter.model_name,
            capabilities=Capabilities(**adapter.capabilities),
            reasoning_field=adapter.reasoning_field,
            default_extras=adapter.default_extras,
            api_key_env=endpoint.api_key_env,
            timeout=endpoint.timeout,
            health_check=endpoint.health_check,
            health_path=endpoint.health_path,
        )
    raise ValueError(f"unsupported LLM kind: {adapter.kind!r}")


def make_vlm(config: ModelsConfig, name: str) -> VLMService:
    """Construct the visual chat service for the named configuration entry.

    Raises :class:`KeyError` when *name* is absent, :class:`TypeError` when it
    names a different model role, and :class:`ValueError` for an unsupported
    adapter kind.
    """

    spec = config.vlm(name)
    adapter = spec.adapter
    endpoint = spec.endpoint
    if adapter.kind == KIND_OPENAI_COMPAT:
        return OpenAICompatVLM(
            base_url=endpoint.base_url,
            model_name=adapter.model_name,
            capabilities=Capabilities(**adapter.capabilities),
            default_extras=adapter.default_extras,
            api_key_env=endpoint.api_key_env,
            timeout=endpoint.timeout,
            health_check=endpoint.health_check,
            health_path=endpoint.health_path,
        )
    raise ValueError(f"unsupported VLM kind: {adapter.kind!r}")


def make_ocr(config: ModelsConfig, name: str) -> OCRService:
    """Construct the OCR service for the named configuration entry.

    NVIDIA Image OCR NIM roles use :class:`NvidiaOCR`; OpenAI-compatible OCR
    roles adapt a configured VLM behind the same :class:`OCRService` protocol.
    """

    spec = config.ocr(name)
    adapter = spec.adapter
    endpoint = spec.endpoint
    if adapter.kind == KIND_NVIDIA_OCR:
        return NvidiaOCR(
            base_url=endpoint.base_url,
            model_name=adapter.model_name,
            request_path=adapter.request_path,
            health_path=endpoint.health_path,
            api_key_env=endpoint.api_key_env,
            timeout=endpoint.timeout,
            health_check=endpoint.health_check,
        )
    if adapter.kind == KIND_OPENAI_COMPAT:
        vlm = OpenAICompatVLM(
            base_url=endpoint.base_url,
            model_name=adapter.model_name,
            capabilities=Capabilities(**adapter.capabilities),
            default_extras=adapter.default_extras,
            api_key_env=endpoint.api_key_env,
            timeout=endpoint.timeout,
            health_check=endpoint.health_check,
            health_path=endpoint.health_path,
        )
        if adapter.prompt:
            return VLMOCR(vlm, prompt=adapter.prompt)
        return VLMOCR(vlm)
    raise ValueError(f"unsupported OCR kind: {adapter.kind!r}")


def make_stt(config: ModelsConfig, name: str) -> STTService:
    """Construct the speech-to-text service for the named configuration entry.

    Raises :class:`KeyError` when *name* is absent, :class:`TypeError` when it
    names a different model role, and :class:`ValueError` for an unsupported
    adapter kind.
    """

    spec = config.stt(name)
    adapter = spec.adapter
    endpoint = spec.endpoint
    if adapter.kind == KIND_OPENAI_COMPAT:
        return OpenAICompatSTT(
            base_url=endpoint.base_url,
            api_key_env=endpoint.api_key_env,
            timeout=endpoint.timeout,
            health_check=endpoint.health_check,
            health_path=endpoint.health_path,
        )
    if adapter.kind == KIND_RIVA_GRPC:
        # Deferred: RivaSTT needs the optional nvidia-riva-client (riva extra).
        from ._riva_grpc import RivaSTT
        return RivaSTT(
            base_url=endpoint.base_url,
            api_key_env=endpoint.api_key_env,
            function_id=adapter.function_id,
            use_ssl=adapter.use_ssl,
            language=adapter.language,
            timeout=endpoint.timeout,
            health_check=endpoint.health_check,
        )
    raise ValueError(f"unsupported STT kind: {adapter.kind!r}")


def make_tts(config: ModelsConfig, name: str) -> TTSService:
    """Construct the text-to-speech service for the named configuration entry.

    Raises :class:`KeyError` when *name* is absent, :class:`TypeError` when it
    names a different model role, and :class:`ValueError` for an unsupported
    adapter kind.
    """

    spec = config.tts(name)
    adapter = spec.adapter
    endpoint = spec.endpoint
    if adapter.kind == KIND_OPENAI_COMPAT:
        return OpenAICompatTTS(
            base_url=endpoint.base_url,
            api_key_env=endpoint.api_key_env,
            timeout=endpoint.timeout,
            health_check=endpoint.health_check,
            health_path=endpoint.health_path,
        )
    if adapter.kind == KIND_RIVA_GRPC:
        from ._riva_grpc import RivaTTS
        return RivaTTS(
            base_url=endpoint.base_url,
            api_key_env=endpoint.api_key_env,
            function_id=adapter.function_id,
            use_ssl=adapter.use_ssl,
            voice=adapter.voice,
            language=adapter.language,
            sample_rate=adapter.sample_rate,
            timeout=endpoint.timeout,
            health_check=endpoint.health_check,
        )
    raise ValueError(f"unsupported TTS kind: {adapter.kind!r}")
