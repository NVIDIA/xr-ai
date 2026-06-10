<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Requirements

## Hardware

XR-AI samples are designed for a single NVIDIA RTX PRO 6000 Blackwell workstation
GPU or an NVIDIA DGX Spark. Both provide enough VRAM to run the full model stack
locally. If you prefer not to run models on local hardware, model endpoints are
plain URLs — point the worker config at a cloud NIM or model endpoint and no
local GPU is required for the agent or hub.

| Sample | Local VRAM needed |
|---|---|
| model-servers (all 4 models) | ~70 GB |
| simple-vlm-example (standalone) | ~23 GB |
| xr-render-demo (requires model-servers) | ~70 GB (models) + ~2 GB (hub/TTS) |
| Hub only | none |

## Software

| Requirement | Version | Notes |
|---|---|---|
| OS | Linux | Ubuntu 22.04 / 24.04 recommended |
| Python | 3.11 or 3.12 | 3.10 and 3.13 are not supported |
| [uv](https://docs.astral.sh/uv/) | latest | dependency manager used by all samples |
| NVIDIA driver | 570+ | required for local model inference |
| Docker | 24+ | required: all vLLM-backed services (LLM, VLM) run in `nvcr.io/nvidia/vllm` containers |
| NVIDIA Container Toolkit | latest | required: gives Docker access to the GPU. Without it, `model_servers` fails with `failed to discover GPU vendor from CDI: no known GPU vendor found` |
| npm | 18+ | required for xr-render-demo: the orchestrator builds the web vendor bundle on first run |

`uv` handles all Python dependencies per-sample — no global `pip install` or
virtual-environment setup needed. If you do not have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The NVIDIA Container Toolkit install is one-time per host. Follow the official
install guide and run the CDI / runtime-configure steps from there:

> https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Quick smoke-test once installed:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu24.04 nvidia-smi
```

## GPU-profile prerequisites

Install before `uv sync` for these targets:

- **DGX Spark** (`xr-render-demo/yaml/spark/`): `sudo apt install python3-dev`

All GPU profiles default to `vllm_backend: docker`, so the vLLM container ships
nvcc + FlashInfer. If you switch a profile to `vllm_backend: pip`, see the
troubleshooting guide for the host CUDA toolchain prerequisite.

If `uv sync` or the VLM fails on first run, see the troubleshooting guide.

## Network

Open the firewall ports listed in the networking guide before connecting from
another machine. UDP 7882 is a silent-failure path: signaling succeeds but media
frames are dropped if it is closed.
