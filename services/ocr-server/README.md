<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# OCR server

Persistent GPU inference for the Hugging Face Nemotron OCR v2 multilingual
model. The launcher builds an isolated CUDA container; the model stays local.

Refer to [OCR serving](https://nvidia.github.io/xr-ai/latest/components/ai-services.html#ocr-serving)
for setup, client usage, request limits, and hardware qualification.
Hardware settings live in
`model-server-samples/model-servers/yaml/<gpu-profile>/ocr_server.yaml`.
