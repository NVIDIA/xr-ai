# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr-ai-models — unified LLM / VLM / STT / TTS service protocols and clients.

Worker code talks to the four ``*Service`` protocols.  The concrete
``OpenAICompat*`` clients cover every in-tree backend (vLLM, in-process
NeMo/Piper) and any external OpenAI-compatible endpoint.  Additional backend
kinds (LiteLLM, vendor SDKs) slot in as new ``kind``s in ``factory.py``
without changing the protocols or callers.
"""
from .protocols import (
    Capabilities,
    ChatMessage,
    ChatResponse,
    ContentPart,
    ImageInput,
    ImagePart,
    LLMService,
    STTService,
    TextPart,
    ToolCall,
    ToolDef,
    TTSService,
    VideoInput,
    VideoPart,
    VLMService,
)
from .openai_compat import (
    OpenAICompatLLM,
    OpenAICompatSTT,
    OpenAICompatTTS,
    OpenAICompatVLM,
)
from .config import (
    LLMSpec,
    ModelsConfig,
    STTSpec,
    TTSSpec,
    VLMSpec,
    load_models_config,
    load_models_config_from_dict,
)
from .factory import make_llm, make_stt, make_tts, make_vlm

__all__ = [
    "Capabilities",
    "ChatMessage",
    "ChatResponse",
    "ContentPart",
    "ImageInput",
    "ImagePart",
    "LLMService",
    "STTService",
    "TextPart",
    "ToolCall",
    "ToolDef",
    "TTSService",
    "VideoInput",
    "VideoPart",
    "VLMService",
    "OpenAICompatLLM",
    "OpenAICompatSTT",
    "OpenAICompatTTS",
    "OpenAICompatVLM",
    "LLMSpec",
    "ModelsConfig",
    "STTSpec",
    "TTSSpec",
    "VLMSpec",
    "load_models_config",
    "load_models_config_from_dict",
    "make_llm",
    "make_stt",
    "make_tts",
    "make_vlm",
]
