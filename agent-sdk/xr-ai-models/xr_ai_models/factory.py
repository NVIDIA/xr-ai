# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``make_*`` constructors that dispatch a :class:`Spec` to a concrete client."""
from __future__ import annotations

from .config import KIND_OPENAI_COMPAT, ModelsConfig
from .openai_compat import (
    OpenAICompatLLM,
    OpenAICompatSTT,
    OpenAICompatTTS,
    OpenAICompatVLM,
)
from .protocols import Capabilities, LLMService, STTService, TTSService, VLMService


def make_llm(config: ModelsConfig, name: str) -> LLMService:
    spec = config.llm(name)
    if spec.kind == KIND_OPENAI_COMPAT:
        return OpenAICompatLLM(
            base_url=spec.base_url,
            model_name=spec.model_name,
            capabilities=Capabilities(**spec.capabilities),
            reasoning_field=spec.reasoning_field,
            default_extras=spec.default_extras,
            api_key_env=spec.api_key_env,
            timeout=spec.timeout,
            health_check=spec.health_check,
        )
    raise ValueError(f"unsupported LLM kind: {spec.kind!r}")


def make_vlm(config: ModelsConfig, name: str) -> VLMService:
    spec = config.vlm(name)
    if spec.kind == KIND_OPENAI_COMPAT:
        return OpenAICompatVLM(
            base_url=spec.base_url,
            model_name=spec.model_name,
            capabilities=Capabilities(**spec.capabilities),
            default_extras=spec.default_extras,
            api_key_env=spec.api_key_env,
            timeout=spec.timeout,
            health_check=spec.health_check,
        )
    raise ValueError(f"unsupported VLM kind: {spec.kind!r}")


def make_stt(config: ModelsConfig, name: str) -> STTService:
    spec = config.stt(name)
    if spec.kind == KIND_OPENAI_COMPAT:
        return OpenAICompatSTT(
            base_url=spec.base_url,
            api_key_env=spec.api_key_env,
            timeout=spec.timeout,
            health_check=spec.health_check,
        )
    raise ValueError(f"unsupported STT kind: {spec.kind!r}")


def make_tts(config: ModelsConfig, name: str) -> TTSService:
    spec = config.tts(name)
    if spec.kind == KIND_OPENAI_COMPAT:
        return OpenAICompatTTS(
            base_url=spec.base_url,
            api_key_env=spec.api_key_env,
            timeout=spec.timeout,
            health_check=spec.health_check,
        )
    raise ValueError(f"unsupported TTS kind: {spec.kind!r}")
