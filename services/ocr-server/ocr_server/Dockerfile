# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" CUDA_HOME=/usr/local/cuda
# Upstream compiles with C++17; keep a matching, qualified PyTorch/CUDA pair.
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
RUN pip install --no-cache-dir hatchling editables setuptools ninja huggingface-hub==0.34.4 \
    pillow==12.3.0 numpy==2.2.6 shapely==2.1.2 fastapi==0.116.1 uvicorn==0.35.0
ENV OCR_MODEL_REVISION=0e83e83f17943524b90afa6c0fd82ac2bc1a40ca
RUN python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download("nvidia/nemotron-ocr-v2", revision=os.environ["OCR_MODEL_REVISION"], local_dir="/opt/upstream", allow_patterns=["nemotron-ocr/**", "LICENSE"])'
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=4
RUN test -n "$TORCH_CUDA_ARCH_LIST" && pip install --no-cache-dir --no-build-isolation /opt/upstream/nemotron-ocr
COPY __init__.py server.py /app/ocr_server/
ENV PYTHONPATH=/app HF_HOME=/models/huggingface TORCH_HOME=/models/torch
ENTRYPOINT ["python", "-m", "ocr_server.server"]
