# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RPC dispatch for document retrieval."""

from loguru import logger

from .index import DenseIndex


class RAGService:
    def __init__(self, index: DenseIndex, *, corpus_id: str | None = None) -> None:
        self._index = index
        self._corpus_id = corpus_id

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
                "rag retrieval results={}",
                [
                    {"source": result["source"], "score": round(result["score"], 3)}
                    for result in results
                ],
            )
            return {"results": results}
        if method == "list_documents":
            return {"documents": self._index.documents}
        if method == "get_health":
            result: dict[str, object] = {
                "ready": await self._index.health(),
                "document_count": len(self._index.documents),
                "chunk_count": len(self._index.chunks),
            }
            if self._corpus_id is not None:
                result["corpus_id"] = self._corpus_id
            return result
        raise ValueError(f"unknown method: {method}")
