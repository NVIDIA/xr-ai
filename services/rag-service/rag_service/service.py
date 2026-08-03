# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RPC dispatch for document retrieval."""

from loguru import logger

from .index import DenseIndex


class RAGService:
    def __init__(self, index: DenseIndex) -> None:
        self._index = index

    async def dispatch(self, method: str, arguments: dict) -> dict:
        if method == "retrieve":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 4)
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string")
            if not isinstance(top_k, int) or not 1 <= top_k <= 20:
                raise ValueError("top_k must be between 1 and 20")
            results = await self._index.retrieve(query, top_k=top_k)
            logger.info(
                "rag retrieval query={!r} results={}",
                query,
                [
                    {"source": result["source"], "score": round(result["score"], 3)}
                    for result in results
                ],
            )
            return {"results": results}
        if method == "list_documents":
            return {"documents": self._index.documents}
        if method == "get_health":
            return {
                "ready": True,
                "document_count": len(self._index.documents),
                "chunk_count": len(self._index.chunks),
            }
        raise ValueError(f"unknown method: {method}")
