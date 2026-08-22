<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Video memory service

Private typed RPC service for selecting and decoding recorded XR camera
history. DeviceIOHub writes H.264 chunks; this process exports requested clips
or timestamped PNG frames for `VideoMemoryTools`.

See [AI inference servers](../../docs/source/components/ai-services.md#per-server-notes)
for timing, sampling, configuration, and VLM-integration guidance. Exact fields
are generated from `video_memory_service.yaml`.
