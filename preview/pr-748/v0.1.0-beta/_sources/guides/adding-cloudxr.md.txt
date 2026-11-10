<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Adding CloudXR to a sample

`cloudxr-runtime/` is a shared top-level component, like `server-runtime/`.
Any sample can stream XR content to a device by adding one line to its
orchestrator and a configuration file in the sample root. For the broader
orchestrator pattern, refer to {doc}`adding-a-sample <adding-a-sample>`.

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

# Device profile — controls transport and XR device defaults.
# Valid: auto-native | auto-webrtc | apple-vision-pro | ipad-pro | quest3
cloudxr_env:
  NV_DEVICE_PROFILE: auto-webrtc

# ── Ports (do not conflict with LiveKit) ──────────────────────────────────────
# CloudXR native service:  localhost:49100  (internal)
# WSS proxy (TLS):         0.0.0.0:48322   (XR clients connect here; auto-webrtc only)
```

## Notes

- CloudXR and the XR-Media-Hub are **independent stacks**. CloudXR streams simulation
  and render content directly to XR devices over WebRTC; the hub handles agent
  media via LiveKit. They share no ports.
- The `auto-webrtc` profile starts a WSS proxy on port 48322 for WebRTC
  signaling. The `auto-native` profile uses a direct native transport and does
  not need the proxy.
- After CloudXR is ready, activate its environment to run an OpenXR app against
  it. Run this in a separate terminal each time you start a new shell against a
  running stack:

  ```bash
  source ~/.cloudxr/run/cloudxr.env
  ```

- For the full list of supported `NV_*` environment variables, refer to the
  CloudXR runtime documentation.
