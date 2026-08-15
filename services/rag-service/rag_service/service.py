# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RPC dispatch for document retrieval."""

from loguru import logger
from pydantic import ValidationError
from xr_ai_tools.rag import RetrieveRequest
from xr_ai_tools.rpc import RPCError
from xr_ai_tools.types import EmptyRequest

from .index import DenseIndex


class RAGService:
    def __init__(self, index: DenseIndex) -> None:
        self._index = index

    async def dispatch(self, method: str, arguments: dict) -> dict:
        if method == "retrieve":
            try:
                request = RetrieveRequest.model_validate(arguments)
            except ValidationError as exc:
                raise RPCError(str(exc), code="invalid_request") from exc
            results = await self._index.retrieve(request.query, top_k=request.top_k)
            logger.info(
                "rag retrieval results={}",
                [
                    {"source": result["source"], "score": round(result["score"], 3)}
                    for result in results
                ],
            )
            return {"results": results}
        if method == "list_documents":
            self._validate_empty(arguments)
            return {"documents": self._index.documents}
        if method == "get_health":
            self._validate_empty(arguments)
            return {
                "ready": await self._index.health(),
                "document_count": len(self._index.documents),
                "chunk_count": len(self._index.chunks),
            }
        raise RPCError(f"unknown operation: {method}", code="unknown_operation")

    @staticmethod
    def _validate_empty(arguments: dict) -> None:
        try:
            EmptyRequest.model_validate(arguments)
        except ValidationError as exc:
            raise RPCError(str(exc), code="invalid_request") from exc
