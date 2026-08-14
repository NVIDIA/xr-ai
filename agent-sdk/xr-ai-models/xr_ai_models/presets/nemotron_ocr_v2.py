# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for Nemotron OCR v2 through the NVIDIA Image OCR NIM API."""

NEMOTRON_OCR_V2 = {
    "category": "ocr",
    "kind": "nvidia_ocr",
    "model_name": "nvidia/nemotron-ocr-v2",
    "request_path": "/v1/ocr",
    "health_path": "/v1/health/ready",
}
