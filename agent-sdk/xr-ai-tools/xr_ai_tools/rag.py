# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native tools backed by the typed document-retrieval service."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .rpc import RPCClient
from .tools import Tool
from .types import EmptyRequest, StrictRequest


class RetrieveRequest(StrictRequest):
    """Search parameters for retrieving relevant document passages."""

    query: str = Field(
        min_length=1,
        description="Question or search phrase to retrieve context for.",
    )
    """Question or search phrase to retrieve context for."""

    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum matching chunks to return.",
    )
    """Maximum number of matching passages to return."""

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        """Require visible text in the retrieval query."""

        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class RetrievedChunk(BaseModel):
    """A document passage returned by similarity search."""

    text: str = Field(description="Retrieved document passage.")
    """Retrieved document passage."""

    source: str = Field(
        description="Source path relative to the configured document directory."
    )
    """Source path relative to the configured document directory."""

    score: float = Field(
        description="Cosine similarity between the query and passage."
    )
    """Cosine similarity between the query and passage."""


class RetrieveResult(BaseModel):
    """Ranked document passages returned for a retrieval request."""

    results: list[RetrievedChunk]
    """Matching passages in service-defined relevance order."""


class ListDocumentsResult(BaseModel):
    """Documents currently indexed by the retrieval service."""

    documents: list[str]
    """Source paths relative to the configured document directory."""


class RAGHealthResult(BaseModel):
    """Readiness and index statistics reported by the RAG service."""

    ready: bool
    """Whether the service is ready to answer retrieval requests."""

    document_count: int
    """Number of indexed documents."""

    chunk_count: int
    """Number of indexed document passages."""


class RAGTools:
    """Own the RAG-service client and document retrieval tools."""

    def __init__(self, endpoint: str, *, timeout_s: float = 30.0) -> None:
        self._rpc = RPCClient(endpoint, timeout_s=timeout_s)
        self.retrieve = Tool(
            "retrieve",
            "Retrieve the most relevant passages from the configured document collection.",
            RetrieveRequest,
            RetrieveResult,
            self._retrieve,
        )
        self.list_documents = Tool(
            "list_documents",
            "List the documents available to the retrieval service.",
            EmptyRequest,
            ListDocumentsResult,
            self._list_documents,
        )
        self.tools = (self.retrieve, self.list_documents)

    async def _retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        return RetrieveResult.model_validate(
            await self._rpc.call("retrieve", request.model_dump())
        )

    async def _list_documents(
        self,
        request: EmptyRequest,
    ) -> ListDocumentsResult:
        return ListDocumentsResult.model_validate(
            await self._rpc.call("list_documents", request.model_dump())
        )

    async def get_health(self) -> RAGHealthResult:
        """Return detailed readiness and index statistics."""

        return RAGHealthResult.model_validate(
            await self._rpc.call("get_health", {}, timeout_s=2.0)
        )

    async def health(self) -> bool:
        """Return whether the RAG service is reachable and ready."""

        try:
            return (await self.get_health()).ready
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying service connection."""

        await self._rpc.close()


__all__ = [
    "ListDocumentsResult",
    "RAGHealthResult",
    "RAGTools",
    "RetrieveRequest",
    "RetrieveResult",
    "RetrievedChunk",
]
