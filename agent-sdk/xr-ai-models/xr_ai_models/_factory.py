# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``make_*`` constructors that dispatch a :class:`Spec` to a concrete client."""
from __future__ import annotations

from ._config import KIND_OPENAI_COMPAT, ModelsConfig
from ._openai_compat import (
    OpenAICompatEmbedding,
    OpenAICompatLLM,
    OpenAICompatSTT,
    OpenAICompatTTS,
    OpenAICompatVLM,
)
from ._protocols import Capabilities, EmbeddingService, LLMService, STTService, TTSService, VLMService


def make_embedding(config: ModelsConfig, name: str) -> EmbeddingService:
    spec = config.embedding(name)
    if spec.kind == KIND_OPENAI_COMPAT:
        return OpenAICompatEmbedding(
            base_url=spec.base_url,
            model_name=spec.model_name,
            api_key_env=spec.api_key_env,
            timeout=spec.timeout,
            health_check=spec.health_check,
        )
    raise ValueError(f"unsupported embedding kind: {spec.kind!r}")


def make_llm(config: ModelsConfig, name: str) -> LLMService:
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
        )
    raise ValueError(f"unsupported LLM kind: {adapter.kind!r}")


def make_vlm(config: ModelsConfig, name: str) -> VLMService:
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
        )
    raise ValueError(f"unsupported VLM kind: {adapter.kind!r}")


def make_stt(config: ModelsConfig, name: str) -> STTService:
    spec = config.stt(name)
    adapter = spec.adapter
    endpoint = spec.endpoint
    if adapter.kind == KIND_OPENAI_COMPAT:
        return OpenAICompatSTT(
            base_url=endpoint.base_url,
            api_key_env=endpoint.api_key_env,
            timeout=endpoint.timeout,
            health_check=endpoint.health_check,
        )
    raise ValueError(f"unsupported STT kind: {adapter.kind!r}")


def make_tts(config: ModelsConfig, name: str) -> TTSService:
    spec = config.tts(name)
    adapter = spec.adapter
    endpoint = spec.endpoint
    if adapter.kind == KIND_OPENAI_COMPAT:
        return OpenAICompatTTS(
            base_url=endpoint.base_url,
            api_key_env=endpoint.api_key_env,
            timeout=endpoint.timeout,
            health_check=endpoint.health_check,
        )
    raise ValueError(f"unsupported TTS kind: {adapter.kind!r}")
