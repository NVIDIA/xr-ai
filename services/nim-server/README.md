<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# nim-server

Generic launcher for a self-hosted [NVIDIA NIM](https://build.nvidia.com)
container. One `nim_server` command serves any NIM image; the YAML picks
which. Orchestrators list one `Process` row per NIM with a distinct
`config=`.

```yaml
# yaml/nim_llm_server.yaml
image:     nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
http_port: 8106
```

On first start the container pulls from `nvcr.io` and downloads the
GPU-matched optimized engine from NGC (multi-GB; expect a long cold start)
into the `nim_cache` volume; later starts reuse it. Requires `NGC_API_KEY`.
Readiness gates on `/v1/health/ready`.

Riva speech NIMs (parakeet ASR, magpie TTS) additionally set `grpc_port:`,
the gRPC endpoint workers reach via the `riva_grpc` model kind; `http_port`
then only serves health.

The samples launch one of these per managed NIM service in the
`models.vlm_llm_nim.json` deployment profile (the worker reads the same
profile's model entries). See `docs/source/components/ai-services.md` "Hosting models on
NVIDIA NIM".
