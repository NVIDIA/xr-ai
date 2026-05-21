# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural-typing checks for the four xr-ai-models protocols."""
from __future__ import annotations

from xr_ai_models import (
    Capabilities,
    LLMService,
    OpenAICompatLLM,
    OpenAICompatSTT,
    OpenAICompatTTS,
    OpenAICompatVLM,
    STTService,
    TTSService,
    VLMService,
)


def test_openai_compat_llm_satisfies_llm_service() -> None:
    llm = OpenAICompatLLM("http://stub", "llm")
    assert isinstance(llm, LLMService)


def test_openai_compat_vlm_satisfies_vlm_service() -> None:
    vlm = OpenAICompatVLM("http://stub", "vlm")
    assert isinstance(vlm, VLMService)


def test_openai_compat_stt_satisfies_stt_service() -> None:
    stt = OpenAICompatSTT("http://stub")
    assert isinstance(stt, STTService)


def test_openai_compat_tts_satisfies_tts_service() -> None:
    tts = OpenAICompatTTS("http://stub")
    assert isinstance(tts, TTSService)


def test_capabilities_defaults_to_text_streaming_only() -> None:
    cap = Capabilities()
    assert cap.streaming is True
    assert cap.tool_calls is False
    assert cap.vision is False
    assert cap.video is False
    assert cap.reasoning is False


def test_llm_capabilities_propagate_from_constructor() -> None:
    llm = OpenAICompatLLM(
        "http://stub", "llm",
        capabilities=Capabilities(tool_calls=True, reasoning=True),
    )
    assert llm.capabilities.tool_calls is True
    assert llm.capabilities.reasoning is True


def test_vlm_defaults_to_vision_capability() -> None:
    vlm = OpenAICompatVLM("http://stub", "vlm")
    assert vlm.capabilities.vision is True
