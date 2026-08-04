# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Idempotent startup helpers for a local RAG service."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlsplit

from loguru import logger
from xr_ai_nat.functions.rag import RAGClient

from .index import find_document_paths


def connect_endpoint(bind_endpoint: str) -> str:
    """Convert a wildcard TCP bind endpoint into a local connect endpoint."""
    parsed = urlsplit(bind_endpoint)
    if parsed.scheme != "tcp" or parsed.hostname not in {"0.0.0.0", "*", "::"}:
        return bind_endpoint
    if parsed.port is None:
        raise ValueError(f"TCP endpoint does not include a port: {bind_endpoint}")
    return f"tcp://127.0.0.1:{parsed.port}"


def corpus_metadata(documents_dir: Path) -> tuple[list[str], str]:
    """Return source names and a stable content fingerprint for a corpus."""
    paths = find_document_paths(documents_dir)
    names = [str(path.relative_to(documents_dir)) for path in paths]
    digest = hashlib.sha256()
    for path, name in zip(paths, names, strict=True):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return names, digest.hexdigest()


async def reusable_client(
    bind_endpoint: str,
    documents_dir: Path,
) -> RAGClient | None:
    """Return a client when a compatible RAG service already owns the endpoint."""
    endpoint = connect_endpoint(bind_endpoint)
    client = RAGClient(endpoint, timeout_s=2.0)
    try:
        health = await client.get_health()
        remote_documents = (await client.list_documents()).documents
    except Exception as exc:
        logger.debug("no reusable RAG service at {}: {}", endpoint, exc)
        await client.close()
        return None

    expected_documents, expected_corpus_id = corpus_metadata(documents_dir)
    if remote_documents != expected_documents:
        await client.close()
        raise RuntimeError(
            f"RAG endpoint {bind_endpoint} is already serving a different document set; "
            "stop that service or configure a different endpoint"
        )
    if health.corpus_id is not None and health.corpus_id != expected_corpus_id:
        await client.close()
        raise RuntimeError(
            f"RAG endpoint {bind_endpoint} is already serving older document contents; "
            "stop that service so the index can be rebuilt"
        )
    if not health.ready:
        await client.close()
        raise RuntimeError(
            f"RAG endpoint {bind_endpoint} is already occupied by an unhealthy RAG service; "
            "stop that service before restarting"
        )
    return client
