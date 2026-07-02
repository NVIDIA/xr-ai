<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Adding CloudXR to a sample

`cloudxr-runtime/` is a shared top-level component, like `server-runtime/`.
Any sample can stream XR content to a device by adding one line to its
orchestrator and a config file in the sample root. For the broader
orchestrator pattern see `docs/adding-a-sample.md`.

## 1 — Add the process to the orchestrator

```python
PROCESSES = [
    Process("hub",     "../../server-runtime",  "xr_media_hub"),
    Process("cloudxr", "../../cloudxr-runtime", "cloudxr_runtime"),  # ← add this
    Process("worker",  "worker",                "my_agent_worker"),
]
```

## 2 — Add `cloudxr_runtime.yaml` to the sample root

The launcher auto-discovers this file and passes it as `--config`.

```yaml
# CloudXR runtime configuration.
cloudxr_install_dir: ~/.cloudxr

# Accept the NVIDIA CloudXR EULA non-interactively.
# View: https://github.com/NVIDIA/IsaacTeleop/blob/main/deps/cloudxr/CLOUDXR_LICENSE
# Written once to <cloudxr_install_dir>/run/eula_accepted; ignored on subsequent runs.
accept_eula: true

# Device profile: selects the client type and XR device defaults.
#   auto-webrtc: WebRTC / web XR clients (default)
#   auto-native: native iOS / visionOS clients
# Device-specific values also accepted: apple-vision-pro | ipad-pro | quest3
cloudxr_env:
  NV_DEVICE_PROFILE: auto-webrtc

# ── Ports (do not conflict with LiveKit) ──────────────────────────────────────
# CloudXR native service:  localhost:49100  (internal)
# WSS proxy (TLS):         0.0.0.0:48322   (XR clients connect here; auto-webrtc only)
```

## Per-client profile

The transport CloudXR negotiates with the connecting client is selected by
`NV_DEVICE_PROFILE`. Pick the value that matches the client you intend to
run against the same sample. The choice is **exclusive per run**, so you
can run web clients or native Apple clients, not both at once.

| Client | `NV_DEVICE_PROFILE` | WSS proxy on 48322 |
|---|---|---|
| `client-samples/web-xr/` (browser + `@nvidia/cloudxr` JS) | `auto-webrtc` | yes |
| `client-samples/ios-visionos/` (CloudXRKit native) | `auto-native` | unused |
| Other native devices (Quest 3, etc.) | `auto-native` | unused |

Switching audiences is a one-line YAML edit + a stack restart. No
orchestrator or sample-code changes are needed.

## Notes

- CloudXR and the hub are **independent stacks**. CloudXR streams sim/render
  content directly to XR devices; the hub handles agent media via LiveKit.
  They share no ports.
- `auto-webrtc` profile starts a WSS proxy on port 48322 for WebRTC signaling.
  `auto-native` uses a direct native transport and does not need the proxy.
- After CloudXR is ready, activate its environment in a separate terminal to
  run an OpenXR app against it:
  ```bash
  source ~/.cloudxr/run/cloudxr.env
  ```
- Full list of supported `NV_*` env vars: `cloudxr-openxr-runtime` source,
  `env_config` / `nv_config.h`.
