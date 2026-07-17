# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image normalization for VLM inputs."""

import base64
import io
from pathlib import Path


def load_jpeg_data_url(image_path: str | Path, quality: int = 85) -> str:
    """Convert a local image to an RGB JPEG data URL."""

    from PIL import Image

    with Image.open(image_path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
