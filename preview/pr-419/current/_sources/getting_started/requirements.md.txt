<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Requirements

## Hardware

The bundled GPU profiles target a single NVIDIA RTX PRO 6000 Blackwell workstation
GPU or an NVIDIA DGX Spark, both of which have enough VRAM to run the full model
stack locally. These profiles are turnkey presets, not a hardware allowlist: you
can run on other NVIDIA GPUs by tuning the per-server GPU-memory split. Refer to
[Running on other GPUs](#running-on-other-gpus) below.

If you prefer not to run models on local hardware, model endpoints are plain
URLs: point the worker configuration at a cloud NIM or model endpoint and no
local GPU is required for the agent or DeviceIOHub.

| Sample | Local VRAM needed |
|---|---|
| model-servers (all models) | ~55 GB |
| simple-vlm-example (requires model services) | Uses the model-services allocation |
| lab-instrument-monitoring (requires model services) | Uses the model-services allocation + Piper TTS |
| tea-making-sample (requires model services) | Uses the model-services allocation + Piper TTS |
| xr-render-demo (requires model-servers) | ~55 GB (models) + ~2 GB (hub/TTS) |
| Hub only | none |

## Software

| Requirement | Version | Notes |
|---|---|---|
| OS | Linux | Ubuntu 22.04 / 24.04 recommended; WSL2 is not officially supported (refer to [Windows (WSL2)](#windows-wsl2) below) |
| Python | 3.11 or 3.12 | 3.10 and 3.13 are not supported |
| [uv](https://docs.astral.sh/uv/) | latest | dependency manager used by all samples |
| NVIDIA driver | 570+ | required for local model inference |
| Docker | 24+ | required by the checked-in model-server profiles: their LLM, VLM, and embedding services run in `nvcr.io/nvidia/vllm` containers |
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
nvcc + FlashInfer. If you switch a profile to `vllm_backend: pip`, refer to the
troubleshooting guide for the host CUDA toolchain prerequisite.

If `uv sync` or the VLM fails on first run, refer to the troubleshooting guide.

## Windows (WSL2)

WSL2 is not an officially supported or tested platform. The notes below come
from a single field report (Windows 11, RTX PRO 6000 Blackwell, Ubuntu WSL2
distribution with in-distro Docker Engine) and may not generalize to other
setups:

- **model-servers and simple-vlm-example ran end-to-end** in that
  configuration. Docker Desktop's WSL integration did not work for this
  stack: `--network host` attaches containers to the Docker Desktop VM's
  network namespace, not the distribution's, so LiveKit signaling succeeds
  but WebRTC media never flows (clients drop after ~18 s).
- **xr-render-demo cannot run under WSL2.** The WSL2 GPU stack is compute-only
  (CUDA, NVENC, NVML) with no Vulkan ICD, so Vulkan falls back to the llvmpipe
  software rasterizer and the `VK_KHR_external_semaphore_fd` /
  `VK_KHR_external_fence_fd` device extensions CloudXR Runtime requires are
  unavailable. This sample needs bare-metal Linux.
- **NAT networking (the WSL default) limits the stack to a browser on the
  same Windows machine.** The WSL `eth0` address is on a host-internal
  virtual subnet that other devices on the LAN cannot reach, and Windows'
  NAT port forwarding (`netsh portproxy`) is TCP-only, so external clients
  (headset, phone) have no WebRTC media path into a NAT-mode WSL VM.
  Reaching them would require mirrored networking (untested with this stack,
  and subject to the port-8000 collision below).
- **For that same-machine browser under NAT, localhost forwarding is TCP
  only**: signaling works via `localhost` but WebRTC media silently fails.
  Open the web client at the WSL distribution's `eth0` address instead (note
  it can change across reboots). Microphone capture needs a secure context.
  Prefer the hub's default HTTPS web server (`https://<eth0-ip>:8080`): an HTTPS
  origin is a secure context once you trust the hub's self-signed cert
  (download it from `https://<eth0-ip>:8080/cert`, or copy
  `~/.local/share/xr-ai/web-server.crt` out of the WSL filesystem via
  `\\wsl$\`, then install it into the Windows cert store) or click through
  the browser warning. On a plain-HTTP path (the legacy token server, or
  `web_server_tls: false`), the report's verified workaround is
  whitelisting the exact origin in
  `chrome://flags/#unsafely-treat-insecure-origin-as-secure`; the HTTPS route
  was not exercised in the report.
- **Mirrored networking collides with the token server**: Windows' IP Helper
  service occupies port 8000. Refer to the `token_server_port` note in
  `services/device-io-hub/device_io_hub.yaml`.

## Running on other GPUs

A profile (`agent-samples/model-servers/yaml/<profile>/`) is a convenience preset
that pins two knobs per model server so the stack fits a known configuration:

- `cuda_visible_devices` — which physical GPU each server runs on (for example,
  the `dual_48G_ada` profile places some servers on GPU `0` and others on GPU `1`).
- `gpu_memory_utilization` — the fraction of that GPU's VRAM the server may use.
  Several servers share one GPU, so each takes a slice (for example, `0.43`), and
  the slices on a given GPU must sum to less than `1.0`.

To run on a GPU that is not one of the presets, copy the closest profile directory
and adjust those knobs to your hardware:

1. Set `cuda_visible_devices` in each server's YAML to your GPU index, or spread
   the servers across the GPUs you have.
2. Tune `gpu_memory_utilization` per server so the slices on each GPU fit its VRAM.
   Lower the values if a server fails to start with an out-of-memory error; raise
   them if you have spare VRAM.
3. On lower-VRAM GPUs, run fewer models concurrently, or lower `max_model_len` on
   the LLM and VLM servers to reduce the KV-cache footprint.

Then select the reviewed profile explicitly:

```bash
uv run --project agent-samples/model-servers model_servers \
  --gpu-profile <profile-directory-name>
```

Automatic detection intentionally accepts only bundled profiles whose hardware
requirements are known to match. `--gpu-profile` bypasses that selection for a
profile you have explicitly copied and tuned; it does not validate that the model
servers fit the selected devices.

```{note}
The model weights are independent of the GPU. Any NVIDIA GPU with enough VRAM for
the models you load will run the stack; the profiles only encode where each server
lands and how much memory it claims.
```

## Network

Open the firewall ports listed in the networking guide before connecting from
another machine.

```{warning}
UDP 7882 is a silent-failure path: signaling succeeds but media frames are
dropped if it is closed.
```
