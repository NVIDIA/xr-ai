# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT document-retrieval functions backed by the RAG service."""

from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import Field

from ._client import ListDocumentsRequest, ListDocumentsResult, RAGClient


class RAGFunctionsConfig(FunctionGroupBaseConfig, name="xr_rag"):
    endpoint: str = Field(description="Private msgpack/ZMQ endpoint of rag-service.")
    timeout_s: float = Field(default=30.0, gt=0.0)


@register_function_group(config_type=RAGFunctionsConfig)
async def rag_functions(config: RAGFunctionsConfig, _builder: Builder):
    client = RAGClient(config.endpoint, timeout_s=config.timeout_s)

    async def list_documents(request: ListDocumentsRequest) -> ListDocumentsResult:
        return await client.list_documents(request)

    group = FunctionGroup(config=config)
    group.add_function(
        "retrieve",
        client.retrieve,
        description="Retrieve the most relevant passages from the configured document collection.",
    )
    group.add_function(
        "list_documents",
        list_documents,
        description="List the documents available to the retrieval service.",
    )
    try:
        yield group
    finally:
        await client.close()


__all__ = ["RAGFunctionsConfig"]
