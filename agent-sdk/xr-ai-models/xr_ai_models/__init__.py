# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified service protocols and OpenAI-compatible clients for XR AI models.

Repository code talks to the typed ``*Service`` protocols. The concrete
``OpenAICompat*`` clients cover every in-tree backend and external
OpenAI-compatible endpoints.
"""
from ._protocols import (
    Capabilities,
    ChatMessage,
    ChatResponse,
    ContentPart,
    EmbeddingService,
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
from ._openai_compat import (
    OpenAICompatLLM,
    OpenAICompatEmbedding,
    OpenAICompatSTT,
    OpenAICompatTTS,
    OpenAICompatVLM,
)
from ._config import (
    AdapterSpec,
    DeploymentSpec,
    EmbeddingSpec,
    EndpointSpec,
    LLMSpec,
    ModelsConfig,
    STTSpec,
    TTSSpec,
    VLMSpec,
    load_models_config,
    load_models_config_from_dict,
)
from ._factory import make_embedding, make_llm, make_stt, make_tts, make_vlm

__all__ = [
    "Capabilities",
    "ChatMessage",
    "ChatResponse",
    "ContentPart",
    "EmbeddingService",
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
    "OpenAICompatEmbedding",
    "OpenAICompatSTT",
    "OpenAICompatTTS",
    "OpenAICompatVLM",
    "AdapterSpec",
    "DeploymentSpec",
    "EmbeddingSpec",
    "EndpointSpec",
    "LLMSpec",
    "ModelsConfig",
    "STTSpec",
    "TTSSpec",
    "VLMSpec",
    "load_models_config",
    "load_models_config_from_dict",
    "make_llm",
    "make_embedding",
    "make_stt",
    "make_tts",
    "make_vlm",
]
