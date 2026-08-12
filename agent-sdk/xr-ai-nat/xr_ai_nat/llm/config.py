# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT provider registration for the repository model-service boundary."""

from __future__ import annotations

from typing import Any

from nat.plugin_api import (
    Builder,
    LLMBaseConfig,
    LLMFrameworkEnum,
    LLMProviderInfo,
    register_llm_client,
    register_llm_provider,
)
from pydantic import ConfigDict, Field, PrivateAttr, model_validator


class ModelsLLMConfig(LLMBaseConfig, name="xr_ai_models"):
    """Select an ``xr-ai-models`` LLM by profile or bind an existing service."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile_path: str | None = Field(
        default=None,
        description="Path to an xr-ai-models deployment profile.",
    )
    role: str = Field(default="agent_llm", description="LLM role within the deployment profile.")
    service: Any | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Already-created LLMService for programmatic applications.",
    )
    model_name: str = Field(default="xr-ai-model", description="Model name reported to the agent framework.")
    temperature: float = 0.0
    max_tokens: int = 1024
    enable_thinking: bool = False
    thinking_budget: int | None = None

    _resolved_service: Any | None = PrivateAttr(default=None)
    _owns_service: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _select_source(self) -> "ModelsLLMConfig":
        if (self.profile_path is None) == (self.service is None):
            raise ValueError("set exactly one of profile_path or service")
        return self


@register_llm_provider(config_type=ModelsLLMConfig)
async def models_llm_provider(config: ModelsLLMConfig, _builder: Builder):
    """Resolve one shared repository model client for every framework wrapper."""
    if config.service is not None:
        config._resolved_service = config.service
    else:
        from xr_ai_models import load_models_config, make_llm

        assert config.profile_path is not None
        config._resolved_service = make_llm(load_models_config(config.profile_path), config.role)
        config._owns_service = True

    try:
        yield LLMProviderInfo(
            config=config,
            description="An xr-ai-models deployment-profile LLM.",
        )
    finally:
        if config._owns_service and config._resolved_service is not None:
            await config._resolved_service.close()
        config._resolved_service = None
        config._owns_service = False


@register_llm_client(config_type=ModelsLLMConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def models_langchain_client(config: ModelsLLMConfig, _builder: Builder):
    """Adapt the shared model service to LangChain-backed NAT agents."""
    # NAT registers its Function-to-LangChain tool conversion in this module.
    from nat.plugins.langchain import tool_wrapper as _tool_wrapper  # noqa: F401

    from ._langchain import LangChainChatModel

    if config._resolved_service is None:
        raise RuntimeError("ModelsLLMConfig must be added to a NAT Builder before use")
    yield LangChainChatModel(
        service=config._resolved_service,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        enable_thinking=config.enable_thinking,
        thinking_budget=config.thinking_budget,
    )


__all__ = ["ModelsLLMConfig"]
