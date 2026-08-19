<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Requirements

## Hardware

The bundled GPU profiles target a single NVIDIA RTX PRO 6000 Blackwell workstation
GPU or an NVIDIA DGX Spark, both of which have enough GPU memory to run the full model
stack locally. These profiles are turnkey presets, not a hardware allowlist: you
can run on other NVIDIA GPUs by tuning the per-server GPU-memory split. Refer to
[Running on other GPUs](#running-on-other-gpus) below.

If you prefer not to run models on local hardware, model endpoints are plain
URLs: point the worker configuration at a cloud NIM or model endpoint and no
local GPU is required for the agent or XR-Media-Hub.

XR-AI inventories total and currently free GPU memory on every physical GPU before
starting a local model stack. It prints the service reservations, device safety
reserve, existing compute processes, and pass/fail result per GPU. The checked-in
reservations are initial estimates until validated by three-run
measurement on each supported host.

Here, GPU memory means the GPU-visible capacity reported by `nvidia-smi`. It is
dedicated memory on discrete GPUs and shared unified memory on DGX Spark. Because
CPU workloads can consume that shared pool, Spark measurements must include
representative whole-system load and retain a larger safety reserve.

## Software

| Requirement | Version | Notes |
|---|---|---|
| OS | Linux | Ubuntu 22.04 / 24.04 recommended; WSL2 is not officially supported (refer to [Windows (WSL2)](#windows-wsl2) below) |
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
  `services/xr-media-hub/xr_media_hub.yaml`.

## GPU profiles and reservations

A hardware profile
(`agent-samples/model-servers/yaml/<profile>/gpu_profile.yaml`) describes a
specific topology and its per-device safety reserve, not a loose GPU family.
Automatic matching verifies GPU count, compute capability, and minimum GPU memory
independently on every device. Failure to run or parse `nvidia-smi` is fatal;
XR-AI never falls back to an assumed profile.

Each existing service YAML in that directory owns its port, GPU placement, and
absolute `gpu_memory_reservation_gib`. The deployment JSON only selects services.
At runtime XR-AI sums the selected service reservations and derives vLLM's
`gpu_memory_utilization` as:

```text
service reservation GiB / physical GPU total GiB
```

Current non-XR processes remain part of used GPU memory; they do not reduce that
denominator. Preflight requires current usage, incremental service reservations,
and the device safety reserve to fit together before model downloads start.

To intentionally use the closest bundled placement on reviewed custom hardware,
pass `model_servers --gpu-profile <name>`. Preflight still validates actual free
GPU memory. For a durable custom profile, measure it rather than editing utilization
percentages by hand.

## Measuring and certifying GPU memory

The stdlib-only measurement tool samples `nvidia-smi` every 250 ms from baseline
through startup and the final stable window. Run a representative workload before
stopping a long-running sample; the artifact records the overall peak and stable
median/p95. Its recommendation is the observed peak delta plus 10% and a fixed
1 GiB service margin.

```bash
cd xr-ai
uv run --project agent-samples/model-servers \
  python -m xr_ai_launcher.gpu_memory_measure measure \
  --label omni-run-1 --gpu 1 --output omni-run-1.json -- \
  uv run --project services/nemotron-omni-llm \
  nemotron_omni_llm_server \
  --config agent-samples/model-servers/yaml/dual_48G_ada/nemotron_omni_llm_server.yaml
```

Repeat each service measurement at least three times with the same Git revision,
driver, command, and hashed config. Certify that service YAML from the artifacts;
certification rejects inconsistent signatures, uses the largest recommended
reservation, and records the config fingerprint in the same file:

```bash
uv run --project agent-samples/model-servers \
  python -m xr_ai_launcher.gpu_memory_measure certify \
  --config agent-samples/model-servers/yaml/dual_48G_ada/nemotron_omni_llm_server.yaml \
  --gpu 1 \
  --measurement omni-run-1.json \
  --measurement omni-run-2.json \
  --measurement omni-run-3.json
```

Repeat this for every GPU service in the stack. Measure the complete
sample as an additional aggregate check, especially under configured concurrency
and maximum image/video inputs.

```{note}
The model weights are independent of the GPU, but a certification signature is
not: model/runtime/config or driver changes require new measurements.
```

## Network

Open the firewall ports listed in the networking guide before connecting from
another machine.

```{warning}
UDP 7882 is a silent-failure path: signaling succeeds but media frames are
dropped if it is closed.
```
