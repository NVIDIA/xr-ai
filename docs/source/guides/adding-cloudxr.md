<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Adding CloudXR to a sample

`services/cloudxr-runtime/` is the shared CloudXR service.
Any sample can stream XR content to a device by adding one line to its
orchestrator and a configuration file under the sample's `yaml/` directory. For the broader
orchestrator pattern, refer to {doc}`adding-a-sample <adding-a-sample>`.

## 1 — Add the process to the orchestrator

```python
PROCESSES = [
    Process("hub",     "../../services/device-io-hub",  "device_io_hub"),
    Process("cloudxr", "../../services/cloudxr-runtime", "cloudxr_runtime",
            config="yaml/cloudxr_runtime.yaml"),  # ← add this
    Process("worker",  "worker",                "my_agent_worker"),
]
```

(add-cloudxr-runtime-yaml-to-the-sample-root)=
## 2 — Add `cloudxr_runtime.yaml` to the sample's `yaml/` directory

The `Process.config` value above passes this file as `--config`.

```yaml
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
# WSS proxy (TLS):         0.0.0.0:48322   (XR clients connect here; WebRTC profiles only)
```

## Select one client profile

`NV_DEVICE_PROFILE` selects both the client type and CloudXR transport. The
choice is exclusive per run: one stack serves either WebRTC clients or native
clients, not both. Change the profile and restart the stack when switching.

| Client | `NV_DEVICE_PROFILE` | WSS proxy on 48322 |
|---|---|---|
| `client-samples/web-xr/` | `auto-webrtc` | used |
| `client-samples/ios-visionos/` | `auto-native` | unused |
| Other native CloudXR clients | `auto-native` | unused |
| Meta Quest 3 with CloudXR.js | `quest3` | used |

The runtime also accepts the device-specific profiles `apple-vision-pro`,
`ipad-pro`, and `quest3`. The static `quest3` profile is a valid alternative
when fixed Quest 3 defaults are preferred. Use `auto-webrtc` for the general
Quest 2/3/3S path so the runtime waits for the CloudXR.js client and discovers
its device settings dynamically.

## Notes

- CloudXR and the DeviceIOHub are **independent stacks**. CloudXR streams
  simulation and render content through the transport selected by the device
  profile; the hub handles agent media via LiveKit. They share no ports.
- WebRTC profiles (`auto-webrtc` and `quest3`) start a WSS proxy on port 48322
  for signaling. Native profiles use a direct native transport and do not need
  the proxy.
- After CloudXR is ready, activate its environment to run an OpenXR app against
  it. Run this in a separate terminal each time you start a new shell against a
  running stack:

  ```bash
  source ~/.cloudxr/run/cloudxr.env
  ```

- For the full list of supported `NV_*` environment variables, refer to the
  CloudXR runtime documentation.
