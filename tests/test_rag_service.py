# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only coverage for dense document indexing and retrieval."""

import asyncio
import hashlib
import uuid
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from rag_service import DenseIndex, RAGService
from rag_service import __main__ as main
from rag_service.index import _chunk_text
from xr_ai_tools.rag import RAGTools, RetrieveRequest
from xr_ai_tools.rpc import RPCServer
from xr_ai_tools.types import EmptyRequest


class _Embedding:
    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy
        self.calls: list[list[str]] = []
        self.identity: str | None = "test-model-v1"

    async def model_identity(self):
        return self.identity

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

    changed_dimension = _Embedding()
    await DenseIndex.build(documents, changed_dimension, cache_dir=cache, embedding_dim=2)
    assert changed_dimension.calls
    assert len(list(cache.glob("*.npz"))) == 2

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


async def test_cache_tracks_backing_model_and_skips_unknown_identity(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("Reset the task.")
    cache = tmp_path / "cache"
    embedder = _Embedding()
    await DenseIndex.build(documents, embedder, cache_dir=cache, embedding_dim=3)
    original = next(cache.glob("*.npz")).read_bytes()

    embedder.identity = "different-weights-same-alias"
    embedder.calls.clear()
    await DenseIndex.build(documents, embedder, cache_dir=cache, embedding_dim=3)
    assert embedder.calls
    assert len(list(cache.glob("*.npz"))) == 2

    embedder.identity = None
    for _ in range(2):
        embedder.calls.clear()
        await DenseIndex.build(documents, embedder, cache_dir=cache, embedding_dim=3)
        assert embedder.calls
        assert len(list(cache.glob("*.npz"))) == 2
    assert original in [path.read_bytes() for path in cache.glob("*.npz")]


async def test_legacy_cache_without_model_identity_is_not_reused(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("Reset the task.")
    cache = tmp_path / "cache"
    cache.mkdir()
    # The previous format can contain valid-shaped vectors from different weights.
    digest = hashlib.sha256(b"900:120:3:passage: :guide.mdReset the task.")
    legacy = cache / f"{digest.hexdigest()}.npz"
    np.savez_compressed(legacy, vectors=np.ones((1, 3), dtype=np.float32))
    embedder = _Embedding()
    await DenseIndex.build(documents, embedder, cache_dir=cache, embedding_dim=3)
    assert embedder.calls
    assert len(list(cache.glob("*.npz"))) == 2


@pytest.mark.parametrize("final_identity", ["different-weights", None])
async def test_model_change_during_indexing_never_persists_or_serves_vectors(tmp_path, final_identity):
    class ChangingEmbedding(_Embedding):
        async def embed(self, texts, *, timeout=None):
            rows = await super().embed(texts, timeout=timeout)
            self.identity = final_identity
            return rows

    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("Reset the task.")
    (documents / "count.md").write_text("Count the tasks.")
    cache = tmp_path / "cache"
    embedder = ChangingEmbedding()
    with pytest.raises(RuntimeError, match="identity changed.*restart RAG"):
        await DenseIndex.build(documents, embedder, cache_dir=cache, embedding_dim=3, batch_size=1)
    assert len(embedder.calls) == 2
    assert not list(cache.glob("*.npz"))

    # Retrying against the original model must embed again, never reuse the failed build.
    stable = _Embedding()
    await DenseIndex.build(documents, stable, cache_dir=cache, embedding_dim=3)
    assert stable.calls
    assert len(list(cache.glob("*.npz"))) == 1


async def test_profile_identity_is_not_replaced_by_manual_cache_key(tmp_path, monkeypatch) -> None:
    profile = tmp_path / "models.json"
    profile.write_text("profile-v1")
    monkeypatch.setattr(main, "load_models_config", lambda _path: object())
    monkeypatch.setattr(main, "make_embedding", lambda *_args: _Embedding())
    captured = []

    async def build(*args, **kwargs):
        captured.append(kwargs)
        raise RuntimeError("stop before serving")

    monkeypatch.setattr(main.DenseIndex, "build", build)
    config = {"models_config": "models.json", "documents_dir": ".", "cache_key": "manual-v1"}
    for content in ("profile-v1", "profile-v2"):
        profile.write_text(content)
        with pytest.raises(RuntimeError, match="stop before serving"):
            await main._serve(config, tmp_path / "rag.yaml", None)
    assert captured[0]["embedding_dim"] == captured[1]["embedding_dim"] == 2048
    assert captured[0]["cache_key"] == captured[1]["cache_key"] == "manual-v1"
    assert captured[0]["profile_key"] != captured[1]["profile_key"]


async def test_profile_change_invalidates_cache_with_same_manual_key(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("Reset the task.")
    cache = tmp_path / "cache"
    embedder = _Embedding()
    for profile in ("profile-v1", "profile-v2"):
        embedder.calls.clear()
        await DenseIndex.build(
            documents, embedder, cache_dir=cache, embedding_dim=3,
            cache_key="manual-v1", profile_key=profile,
        )
        assert embedder.calls
    assert len(list(cache.glob("*.npz"))) == 2


async def test_default_dimension_is_native_in_index_and_standalone_yaml(tmp_path) -> None:
    class NativeEmbedding(_Embedding):
        async def embed(self, texts, *, timeout=None):
            return [[0.1] * 2048 for _ in texts]

    config = yaml.safe_load((Path(__file__).resolve().parents[1] /
                             "services/rag-service/rag_service.yaml").read_text())
    assert config["embedding_dim"] == 2048
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("Reset the task.")
    index = await DenseIndex.build(documents, NativeEmbedding(), cache_dir=tmp_path / "cache")
    assert index.vectors.shape == (1, 2048)


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
