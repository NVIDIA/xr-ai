# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts and private client for document retrieval."""

from pydantic import BaseModel, Field, field_validator

from .._models import _StrictRequest
from .._service.rpc import RPCClient


class RetrieveRequest(_StrictRequest):
    query: str = Field(min_length=1, description="Question or search phrase to retrieve context for.")
    top_k: int = Field(default=4, ge=1, le=20, description="Maximum matching chunks to return.")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class RetrievedChunk(BaseModel):
    text: str = Field(description="Retrieved document passage.")
    source: str = Field(description="Source path relative to the configured document directory.")
    score: float = Field(description="Cosine similarity between the query and passage.")


class RetrieveResult(BaseModel):
    results: list[RetrievedChunk]


class ListDocumentsRequest(_StrictRequest):
    """Discovery operation that does not require arguments."""


class ListDocumentsResult(BaseModel):
    documents: list[str]


class RAGHealthRequest(_StrictRequest):
    """Readiness probe that does not require arguments."""


class RAGHealthResult(BaseModel):
    ready: bool
    document_count: int
    chunk_count: int


class RAGClient:
    def __init__(self, endpoint: str, *, timeout_s: float = 30.0) -> None:
        self._rpc = RPCClient(endpoint, timeout_s=timeout_s)

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        return RetrieveResult.model_validate(
            await self._rpc.call("retrieve", request.model_dump())
        )

    async def list_documents(
        self, request: ListDocumentsRequest | None = None
    ) -> ListDocumentsResult:
        arguments = (request or ListDocumentsRequest()).model_dump()
        return ListDocumentsResult.model_validate(
            await self._rpc.call("list_documents", arguments)
        )

    async def get_health(
        self, request: RAGHealthRequest | None = None
    ) -> RAGHealthResult:
        arguments = (request or RAGHealthRequest()).model_dump()
        return RAGHealthResult.model_validate(
            await self._rpc.call("get_health", arguments, timeout_s=2.0)
        )

    async def health(self) -> bool:
        try:
            return (await self.get_health()).ready
        except Exception:
            return False

    async def close(self) -> None:
        await self._rpc.close()


__all__ = [
    "ListDocumentsRequest",
    "ListDocumentsResult",
    "RAGClient",
    "RAGHealthRequest",
    "RAGHealthResult",
    "RetrieveRequest",
    "RetrieveResult",
    "RetrievedChunk",
]
