# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose retrieval as one individually selectable NAT tool."""

from time import perf_counter
from typing import Any

from nat.builder.function import Function
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import ConfigDict, Field
from xr_ai_nat.functions.rag import RetrieveRequest, RetrieveResult

from ..runtime.events import emit
from ..runtime.scope import current_invocation


class RAGLookupConfig(FunctionBaseConfig, name="tea_guidance_rag_lookup"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Any = Field(exclude=True, repr=False)
    max_results: int = Field(default=2, ge=1, le=20)


@register_function(config_type=RAGLookupConfig)
async def rag_lookup(config: RAGLookupConfig, _builder: Builder):
    source: Function = config.source

    async def retrieve(request: RetrieveRequest) -> RetrieveResult:
        call = current_invocation()
        started = perf_counter()
        request = request.model_copy(update={"top_k": min(request.top_k, config.max_results)})
        emit(
            "rag.lookup.request",
            participant_id=call.session.participant_id,
            step=call.session.step_id,
            trace_id=call.trace_id,
            query=request.query,
            top_k=request.top_k,
        )
        result = await source.ainvoke(request, to_type=RetrieveResult)
        emit(
            "rag.lookup.response",
            participant_id=call.session.participant_id,
            step=call.session.step_id,
            trace_id=call.trace_id,
            latency_ms=round((perf_counter() - started) * 1_000, 1),
            results=[chunk.model_dump() for chunk in result.results],
        )
        return result

    yield FunctionInfo.from_fn(
        retrieve,
        description=(
            "Retrieve missing brewing values after reading an exact tea name. Query for that name, water "
            "temperature, and steep time. Results support values only, never identity; require the same variety."
        ),
    )


__all__ = ["RAGLookupConfig"]
