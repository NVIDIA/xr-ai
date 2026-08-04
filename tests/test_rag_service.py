# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only coverage for dense document indexing and retrieval."""

import asyncio
import uuid

import numpy as np
import pytest
from nat.builder.workflow_builder import WorkflowBuilder
from rag_service import DenseIndex, RAGService
from rag_service.startup import connect_endpoint, corpus_metadata, reusable_client
from xr_ai_nat.functions._service.rpc import RPCServer
from xr_ai_nat.functions.rag import RAGClient, RAGFunctionsConfig, RetrieveRequest


class _Embedding:
    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy
        self.calls: list[list[str]] = []

    async def embed(self, texts, *, timeout=None):
        self.calls.append(list(texts))
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
        return self.healthy

    async def close(self):
        return None


async def test_build_retrieve_list_and_health(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "workflow.md").write_text("Reset the task before starting again.")
    (documents / "counting.txt").write_text("Count objects carefully from left to right.")
    embedder = _Embedding()
    index = await DenseIndex.build(
        documents,
        embedder,
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
    embedder.healthy = False
    assert (await service.dispatch("get_health", {}))["ready"] is False


async def test_cache_reuse_and_invalidation(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    document = documents / "guide.md"
    document.write_text("Reset the task before starting again.")
    cache = tmp_path / "cache"

    initial = _Embedding()
    await DenseIndex.build(documents, initial, cache_dir=cache, embedding_dim=3)
    assert initial.calls

    reused = _Embedding()
    await DenseIndex.build(documents, reused, cache_dir=cache, embedding_dim=3)
    assert reused.calls == []

    document.write_text("Count objects carefully from left to right.")
    changed_document = _Embedding()
    await DenseIndex.build(
        documents,
        changed_document,
        cache_dir=cache,
        embedding_dim=3,
    )
    assert changed_document.calls

    changed_key = _Embedding()
    await DenseIndex.build(
        documents,
        changed_key,
        cache_dir=cache,
        embedding_dim=3,
        cache_key="model-revision-2",
    )
    assert changed_key.calls


async def test_invalid_cached_shape_is_rebuilt(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("Reset the task before starting again.")
    cache = tmp_path / "cache"
    await DenseIndex.build(documents, _Embedding(), cache_dir=cache, embedding_dim=3)
    cache_path = next(cache.glob("*.npz"))
    np.savez_compressed(cache_path, vectors=np.zeros((1, 2), dtype=np.float32))

    rebuilt = _Embedding()
    index = await DenseIndex.build(
        documents,
        rebuilt,
        cache_dir=cache,
        embedding_dim=3,
    )
    assert rebuilt.calls
    assert index.vectors.shape == (1, 3)


@pytest.mark.parametrize("vector", [[1.0, 2.0], [1.0, float("nan"), 3.0]])
async def test_invalid_generated_vectors_are_rejected(tmp_path, vector) -> None:
    class _InvalidEmbedding(_Embedding):
        async def embed(self, texts, *, timeout=None):
            return [vector for _ in texts]

    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("Reset the task before starting again.")
    with pytest.raises(ValueError, match="embedding"):
        await DenseIndex.build(
            documents,
            _InvalidEmbedding(),
            cache_dir=tmp_path / "cache",
            embedding_dim=3,
        )


async def test_min_score_filters_weak_matches(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("Reset the task before starting again.")
    index = await DenseIndex.build(
        documents,
        _Embedding(),
        cache_dir=tmp_path / "cache",
        embedding_dim=3,
        min_score=0.3,
    )
    assert await index.retrieve("unrelated topic", top_k=1) == []


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
        await asyncio.gather(task, return_exceptions=True)

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


def test_startup_metadata_and_wildcard_endpoint(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("Make tea carefully.")

    names, first_id = corpus_metadata(documents)
    assert names == ["guide.md"]
    assert connect_endpoint("tcp://0.0.0.0:8340") == "tcp://127.0.0.1:8340"
    assert connect_endpoint("ipc:///tmp/rag") == "ipc:///tmp/rag"

    (documents / "guide.md").write_text("Updated tea guidance.")
    assert corpus_metadata(documents)[1] != first_id


@pytest.mark.parametrize("publish_corpus_id", [False, True])
async def test_reuses_compatible_running_service(tmp_path, publish_corpus_id) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("Reset the task before starting again.")
    corpus_id = corpus_metadata(documents)[1]
    index = await DenseIndex.build(
        documents,
        _Embedding(),
        cache_dir=tmp_path / "cache",
        embedding_dim=3,
    )
    endpoint = f"ipc:///tmp/rag-reuse-{uuid.uuid4().hex}"
    server = RPCServer(
        endpoint,
        RAGService(
            index,
            corpus_id=corpus_id if publish_corpus_id else None,
        ).dispatch,
    )
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.02)
    client = None
    try:
        client = await reusable_client(endpoint, documents)
        assert client is not None
        assert await client.health()
    finally:
        if client is not None:
            await client.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
