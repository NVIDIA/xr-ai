<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Video memory service

Private typed RPC service for selecting and decoding recorded XR camera
history. DeviceIOHub writes H.264 chunks; this process exports requested clips
or timestamped PNG frames for `VideoMemoryTools`.

Refer to [AI inference servers](https://nvidia.github.io/xr-ai/latest/components/ai-services.html#per-server-notes)
for timing, sampling, configuration, and VLM-integration guidance. The adjacent
`video_memory_service.yaml` and its comments define the standalone
configuration.
