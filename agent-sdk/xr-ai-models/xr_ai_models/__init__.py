# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified service protocols and OpenAI-compatible clients for XR AI models.

Repository code talks to the typed ``*Service`` protocols. The concrete
``OpenAICompat*`` clients cover every in-tree backend (vLLM, in-process
NeMo/Piper) and any external OpenAI-compatible endpoint.  Additional backend
kinds (LiteLLM, vendor SDKs) slot in as new ``kind``s in the private factory
implementation without changing the protocols or callers.
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
    DeploymentSpec,
    EmbeddingSpec,
    LLMSpec,
    ModelsConfig,
    STTSpec,
    TTSSpec,
    VLMSpec,
    load_models_config,
    load_models_config_from_dict,
)
from ._factory import make_embedding, make_llm, make_stt, make_tts, make_vlm
from ._riva_grpc import RivaSTT, RivaTTS

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
    "DeploymentSpec",
    "EmbeddingSpec",
    "RivaSTT",
    "RivaTTS",
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
