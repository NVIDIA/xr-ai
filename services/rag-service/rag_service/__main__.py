# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the RAG service."""

import argparse
import asyncio
import hashlib
from pathlib import Path

import yaml
from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_embedding
from xr_ai_nat.functions._service.rpc import RPCServer

from .index import DenseIndex
from .service import RAGService
from .startup import corpus_metadata, reusable_client

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "rag_service.yaml"


def _resolve(path: str, config_path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()


async def _serve(config: dict, config_path: Path, ready_file: Path | None) -> None:
    address = str(config.get("endpoint", "tcp://0.0.0.0:8340"))
    documents_dir = _resolve(str(config["documents_dir"]), config_path)
    existing = await reusable_client(address, documents_dir)
    if existing is not None:
        logger.info("rag-service already running at {} - reusing", address)
        if ready_file is not None:
            ready_file.touch()
        try:
            while True:
                await asyncio.sleep(2.0)
                if not await existing.health():
                    raise RuntimeError(f"reused RAG service at {address} became unavailable")
        finally:
            await existing.close()

    models_path = _resolve(str(config["models_config"]), config_path)
    models = load_models_config(models_path)
    embedder = make_embedding(models, str(config.get("embedding_role", "embedding")))
    try:
        if not await embedder.health():
            raise RuntimeError("embedding service is not healthy")
        index = await DenseIndex.build(
            documents_dir,
            embedder,
            cache_dir=_resolve(str(config.get("cache_dir", ".rag-cache")), config_path),
            chunk_size=int(config.get("chunk_size", 900)),
            overlap=int(config.get("overlap", 120)),
            embedding_dim=int(config.get("embedding_dim", 768)),
            batch_size=int(config.get("batch_size", 32)),
            passage_prefix=str(config.get("passage_prefix", "passage: ")),
            query_prefix=str(config.get("query_prefix", "query: ")),
            cache_key=str(
                config.get("cache_key")
                or hashlib.sha256(models_path.read_bytes()).hexdigest()
            ),
            min_score=float(config.get("min_score", 0.3)),
        )
        _, corpus_id = corpus_metadata(documents_dir)
        logger.info(
            "rag-service rpc={} documents={} chunks={}",
            address,
            len(index.documents),
            len(index.chunks),
        )
        await RPCServer(address, RAGService(index, corpus_id=corpus_id).dispatch).serve(
            ready=ready_file.touch if ready_file else None
        )
    finally:
        await embedder.close()


def run() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    args, _ = parser.parse_known_args()
    setup_logging("rag-service")
    config_path = (args.config or _DEFAULT_CONFIG).resolve()
    config = yaml.safe_load(config_path.read_text()) or {}
    asyncio.run(_serve(config, config_path, args.ready_file))


if __name__ == "__main__":
    run()
