<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# OCR server

Persistent GPU inference for the Hugging Face Nemotron OCR v2 multilingual
model in an operator-managed CUDA container; the model stays local.

Refer to [OCR serving](https://nvidia.github.io/xr-ai/latest/components/ai-services.html#ocr-serving)
for setup, client usage, request limits, and hardware qualification.
The Dockerfile and inference app live in `ocr_server/`. Configure inference
through the container environment and connect through `xr_ai_models`.
There is no host launcher or sample-stack registration. No supplied sample uses OCR.
