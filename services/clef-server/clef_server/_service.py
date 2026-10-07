# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP and native inference lifecycle for the Clef joint-schema model."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from ._config import ServerConfig, identity


class ClefBackend:
    """Own the downloaded code, model, processor, and serialized GPU calls."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.revision = config.model_revision
        self.model: Any = None
        self.processor: Any = None
        self.systemone: Callable[..., dict[str, Any]] | None = None
        self.encode_record: Callable[..., Any] | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clef-inference")

    def load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download

        if self.config.model_path is not None:
            model_dir = self.config.model_path
            if not model_dir.is_dir():
                raise ValueError(f"model_path is not a directory: {model_dir}")
        else:
            self.config.model_cache.mkdir(parents=True, exist_ok=True)
            model_dir = Path(
                snapshot_download(
                    repo_id=self.config.model_name,
                    revision=self.config.model_revision,
                    cache_dir=self.config.model_cache,
                )
            )
        source = model_dir / "joint_schema_model.py"
        if not source.is_file():
            raise ValueError(f"missing upstream model implementation: {source}")
        module_spec = importlib.util.spec_from_file_location("_clef_release_model", source)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"could not load upstream model implementation: {source}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.config.dtype]
        self.model, self.processor = module.load_release_model(
            model_dir,
            device=self.config.device,
            dtype=dtype,
        )
        self.systemone = module.systemone
        self.encode_record = module.encode_record
        logger.info(
            "Loaded Clef model {} at revision {} from {}",
            self.config.model_name,
            self.revision,
            model_dir,
        )

    def validate_length(self, body: dict[str, Any]) -> None:
        if self.encode_record is None or self.processor is None:
            raise RuntimeError("Clef model is not loaded")
        try:
            encoded = self.encode_record(
                self.processor.tokenizer,
                body,
                # A token needs at least one UTF-8 byte, so the request-size
                # limit bounds token count. Leave room for fixed prompt text.
                max_length=max(self.config.max_length, self.config.max_body_bytes + 4096),
                processor=self.processor,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if len(encoded.input_ids) > self.config.max_length:
            raise ValueError(f"request requires {len(encoded.input_ids)} tokens; maximum is {self.config.max_length}")

    def answer(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.systemone is None or self.model is None or self.processor is None:
            raise RuntimeError("Clef model is not loaded")
        return self.systemone(self.model, self.processor, body, max_length=self.config.max_length)

    async def close(self) -> None:
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=False)


def create_app(
    config: ServerConfig,
    backend: ClefBackend | None = None,
) -> FastAPI:
    """Build the HTTP application; injection keeps route tests independent of CUDA."""
    instance = backend or ClefBackend(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = instance
        app.state.inference_lock = asyncio.Lock()
        try:
            if instance.model is None:
                await _await_without_abandoning(asyncio.create_task(asyncio.to_thread(instance.load)))
            await _warm(instance)
            yield
        finally:
            await instance.close()

    app = FastAPI(title="Clef SystemOne", lifespan=lifespan)
    app.state.backend = instance
    app.state.inference_lock = asyncio.Lock()

    @app.exception_handler(ValueError)
    async def invalid_request(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return identity(config)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model_name,
                    "object": "model",
                    "model_revision": instance.revision,
                }
            ],
        }

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> dict[str, Any]:
        length = request.headers.get("content-length")
        if length is not None:
            try:
                if int(length) > config.max_body_bytes:
                    raise HTTPException(status_code=413, detail="request body is too large")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid Content-Length") from exc
        raw_parts: list[bytes] = []
        size = 0
        async for part in request.stream():
            size += len(part)
            if size > config.max_body_bytes:
                raise HTTPException(status_code=413, detail="request body is too large")
            raw_parts.append(part)
        raw = b"".join(raw_parts)
        try:
            body = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        _validate_request(body, config)
        loop = asyncio.get_running_loop()
        async with app.state.inference_lock:
            # The executor has one worker, and shielding keeps the request lock
            # held until a CUDA call finishes even if the client disconnects.
            await _await_without_abandoning(loop.run_in_executor(instance.executor, instance.validate_length, body))
            future = loop.run_in_executor(instance.executor, instance.answer, body)
            result = await _await_without_abandoning(future)
        result["model"] = config.model_name
        result["model_revision"] = instance.revision
        return result

    return app


async def _await_without_abandoning(future):
    """Keep serialized executor work accounted for after client cancellation."""
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(future)
        except Exception:
            pass
        raise


async def _warm(backend: ClefBackend) -> None:
    request = {
        "model": backend.config.model_name,
        "state": "warmup",
        "questions": {
            "ready": {
                "type": "choice",
                "instructions": "Choose the warmup option.",
                "criteria": {"a": "A", "b": "B"},
            }
        },
    }
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(backend.executor, backend.answer, request)
    result = await _await_without_abandoning(future)
    if "ready" not in result.get("answers", {}):
        raise RuntimeError("Clef warmup returned no answer")


def _validate_request(body: Any, config: ServerConfig) -> None:
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    if body.get("model") != config.model_name:
        raise ValueError(f"model must be {config.model_name!r}")
    if "state" not in body or not isinstance(body["state"], (str, dict)):
        raise ValueError("state must be text or a JSON object")
    if "images" in body or "videos" in body:
        raise ValueError("this endpoint accepts text-only choice requests")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("question IDs must be non-empty strings")
        if not isinstance(question, dict) or question.get("type") != "choice":
            raise ValueError(f"{question_id}: only choice questions are supported")
        if not isinstance(question.get("instructions"), str):
            raise ValueError(f"{question_id}: instructions must be a string")
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError(f"{question_id}: criteria must contain at least two choices")
        if any(not isinstance(key, str) or not key.strip() for key in criteria):
            raise ValueError(f"{question_id}: choice labels must be non-empty strings")
        if any(not isinstance(value, str) or not value.strip() for value in criteria.values()):
            raise ValueError(f"{question_id}: choice descriptions must be non-empty strings")
