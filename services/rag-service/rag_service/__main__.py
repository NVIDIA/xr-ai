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

def _resolve(path: str, config_path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()


async def _serve(config: dict, config_path: Path, ready_file: Path | None) -> None:
    models_path = _resolve(str(config["models_config"]), config_path)
    models = load_models_config(models_path)
    embedder = make_embedding(models, str(config.get("embedding_role", "embedding")))
    try:
        if not await embedder.health():
            raise RuntimeError("embedding service is not healthy")
        index = await DenseIndex.build(
            _resolve(str(config["documents_dir"]), config_path),
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
        address = str(config.get("endpoint", "tcp://0.0.0.0:8340"))
        logger.info(
            "rag-service rpc={} documents={} chunks={}",
            address,
            len(index.documents),
            len(index.chunks),
        )
        await RPCServer(address, RAGService(index).dispatch).serve(
            ready=ready_file.touch if ready_file else None
        )
    finally:
        await embedder.close()


def run() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, default=None)
    args, _ = parser.parse_known_args()
    setup_logging("rag-service")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text()) or {}
    asyncio.run(_serve(config, config_path, args.ready_file))


if __name__ == "__main__":
    run()
