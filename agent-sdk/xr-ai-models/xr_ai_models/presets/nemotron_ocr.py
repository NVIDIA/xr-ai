# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared local Hugging Face and self-hosted NIM OCR v2 wire contract."""

NEMOTRON_OCR = {
    "category": "ocr",
    "kind": "nemotron_ocr",
    "health_path": "/v1/health/ready",
    "timeout": 120.0,
}
