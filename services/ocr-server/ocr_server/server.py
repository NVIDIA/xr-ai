# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-worker multilingual OCR with the NIM v2 image request/response shape."""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

MODEL_ID = "nvidia/nemotron-ocr-v2"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image_url"]
    url: str = Field(max_length=4 * ((MAX_IMAGE_BYTES + 2) // 3) + 64)


class OCRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: list[ImageInput] = Field(min_length=1, max_length=1)
    merge_levels: list[Literal["word", "sentence", "paragraph"]] = Field(
        default=["word"], min_length=1, max_length=1,
    )


def decode_image(url: str) -> Image.Image:
    """Decode bounded inline PNG/JPEG only; never fetch caller-controlled URLs."""
    header, separator, encoded = url.partition(",")
    if not separator or header not in {"data:image/png;base64", "data:image/jpeg;base64"}:
        raise HTTPException(422, "Use a base64 PNG or JPEG data URL")
    try:
        data = base64.b64decode(encoded, validate=True)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Image exceeds 10 MiB")
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image exceeds 20 million pixels")
            if image.format not in {"PNG", "JPEG"}:
                raise HTTPException(422, "Only PNG and JPEG images are supported")
            return image.convert("RGB")
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError,
            Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Invalid image") from exc


def load_model():
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

    if not torch.cuda.is_available():
        raise RuntimeError("Nemotron OCR requires a CUDA GPU")
    torch.set_num_threads(4)
    length = int(os.environ.get("OCR_INFER_LENGTH", "1024"))
    if length not in (640, 1024):
        raise ValueError("OCR_INFER_LENGTH must be 640 or 1024")
    snapshot = snapshot_download(
        MODEL_ID, revision=os.environ["OCR_MODEL_REVISION"],
        allow_patterns=["v2_multilingual/*"],
    )
    model = NemotronOCRV2(
        model_dir=f"{snapshot}/v2_multilingual", lang="multi", infer_length=length,
        detector_max_batch_size=1, recognizer_chunk_size=32, relational_chunk_size=32,
    )
    # Warm the detector before readiness; recognition buffers remain input-dependent.
    model(np.full((length, length, 3), 255, dtype=np.uint8), merge_level="word")
    torch.cuda.synchronize()
    return model


def transcribe(model, request: OCRRequest) -> dict:
    import numpy as np
    import torch

    image = decode_image(request.input[0].url)
    with torch.inference_mode():
        predictions = model(np.asarray(image), merge_level=request.merge_levels[0])
    detections = []
    for prediction in predictions:
        left, right = sorted((prediction["left"], prediction["right"]))
        # Upstream's 'upper' is max-y and 'lower' is min-y.
        top, bottom = sorted((prediction["upper"], prediction["lower"]))
        detections.append({
            "text_prediction": {
                "text": prediction["text"], "confidence": prediction["confidence"],
            },
            "bounding_box": {"points": [
                {"x": left, "y": top}, {"x": right, "y": top},
                {"x": right, "y": bottom}, {"x": left, "y": bottom},
            ]},
        })
    return {"model": MODEL_ID, "data": [{"index": 0, "text_detections": detections}]}


def build_app(loader=load_model, inference=transcribe) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.model = await asyncio.to_thread(loader)
        yield
        del app.state.model

    app = FastAPI(lifespan=lifespan)
    busy = threading.Lock()

    @app.get("/v1/health/ready")
    def health():
        if not hasattr(app.state, "model"):
            raise HTTPException(503, "Model is loading")
        return {"ready": True}

    @app.post("/v1/ocr")
    def ocr(request: OCRRequest):
        # FastAPI runs sync endpoints in its thread pool. Keep ownership in
        # that thread so client cancellation cannot unlock an active GPU call.
        if not busy.acquire(blocking=False):
            raise HTTPException(503, "OCR is busy; retry later")
        try:
            return inference(app.state.model, request)
        finally:
            busy.release()

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host="0.0.0.0", port=8000, workers=1)
