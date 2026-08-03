# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only coverage for dense document indexing and retrieval."""

import asyncio
import contextlib
import uuid

from nat.builder.workflow_builder import WorkflowBuilder
from rag_service import DenseIndex, RAGService
from xr_ai_nat.functions._service.rpc import RPCServer
from xr_ai_nat.functions.rag import RAGClient, RAGFunctionsConfig, RetrieveRequest


class _Embedding:
    async def embed(self, texts, *, timeout=None):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([
                float("reset" in lowered),
                float("count" in lowered),
                0.1,
            ])
        return vectors

    async def health(self):
        return True

    async def close(self):
        return None


async def test_build_retrieve_list_and_health(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "workflow.md").write_text("Reset the task before starting again.")
    (documents / "counting.txt").write_text("Count objects carefully from left to right.")
    index = await DenseIndex.build(
        documents,
        _Embedding(),
        cache_dir=tmp_path / "cache",
        chunk_size=100,
        overlap=10,
        embedding_dim=3,
    )
    service = RAGService(index)

    result = await service.dispatch("retrieve", {"query": "how do I reset?", "top_k": 1})
    assert result["results"][0]["source"] == "workflow.md"
    assert await service.dispatch("list_documents", {}) == {
        "documents": ["counting.txt", "workflow.md"]
    }
    health = await service.dispatch("get_health", {})
    assert health == {"ready": True, "document_count": 2, "chunk_count": 2}


async def test_native_client_and_function_group_use_typed_contracts(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("Reset the task before starting again.")
    index = await DenseIndex.build(
        documents,
        _Embedding(),
        cache_dir=tmp_path / "cache",
        embedding_dim=3,
    )
    endpoint = f"ipc:///tmp/rag-{uuid.uuid4().hex}"
    server = RPCServer(endpoint, RAGService(index).dispatch)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.02)
    client = RAGClient(endpoint)
    try:
        result = await client.retrieve(RetrieveRequest(query="reset", top_k=1))
        assert result.results[0].source == "guide.md"
        assert (await client.get_health()).ready is True
    finally:
        await client.close()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async with WorkflowBuilder() as builder:
        await builder.add_function_group("rag", RAGFunctionsConfig(endpoint=endpoint))
        functions = await (await builder.get_function_group("rag")).get_all_functions()
        schemas = {
            name: function.input_schema.model_json_schema()
            for name, function in functions.items()
        }
    assert set(schemas) == {"rag__retrieve", "rag__list_documents"}
    assert schemas["rag__retrieve"]["properties"]["query"]["minLength"] == 1
    assert schemas["rag__list_documents"].get("properties", {}) == {}
