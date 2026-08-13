# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only coverage for dense document indexing and retrieval."""

import asyncio
import uuid

import numpy as np
import pytest
from pydantic import ValidationError
from rag_service import DenseIndex, RAGService
from rag_service.index import _chunk_text
from xr_ai_tools.rag import RAGTools, RetrieveRequest
from xr_ai_tools.rpc import RPCServer
from xr_ai_tools.types import EmptyRequest


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


def test_chunk_text_splits_oversized_unbroken_token() -> None:
    chunks = _chunk_text("x" * 2_500, chunk_size=900, overlap=120)

    assert [len(chunk) for chunk in chunks] == [900, 900, 700]
    assert "".join(chunks) == "x" * 2_500


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


async def test_native_rag_tools_use_typed_contracts(tmp_path) -> None:
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
    try:
        tools = RAGTools(endpoint)
        try:
            schemas = {
                tool.name: tool.request_model.model_json_schema()
                for tool in tools.tools
            }
            result = await tools.retrieve.execute(
                RetrieveRequest(query="reset", top_k=1)
            )
            documents = await tools.list_documents.execute(EmptyRequest())
            health = await tools.get_health()
        finally:
            await tools.close()
        assert result.results[0].source == "guide.md"
        assert documents.documents == ["guide.md"]
        assert health.ready is True
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert set(schemas) == {"retrieve", "list_documents"}
    assert schemas["retrieve"]["properties"]["query"]["minLength"] == 1
    assert schemas["list_documents"].get("properties", {}) == {}


def test_retrieve_request_rejects_blank_query() -> None:
    with pytest.raises(ValidationError, match="query must not be blank"):
        RetrieveRequest(query="  ")
