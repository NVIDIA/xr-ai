<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Notices

This file lists the third-party open-source software distributed with or
required by `xr-ai`. Each entry gives the upstream project, version, SPDX
license identifier, and a link to the upstream source repository, where the
canonical license text is available.

`xr-ai` itself is licensed under Apache-2.0 — see [`LICENSE`](LICENSE).

## Python (server-side and AI services)

Used by `server-runtime/`, `agent-sdk/`, `utils/`, `ai-services/`,
`agent-mcp-servers/`, `agent-samples/`, `cloudxr-runtime/`, and `tests/`.
For the per-package dependency mapping, see [`DEPENDENCIES.md`](DEPENDENCIES.md).

| Package        | Version  | License       | Upstream |
|---             |---       |---            |---|
| `msgpack`      | 1.0.0    | Apache-2.0    | https://github.com/msgpack/msgpack-python |
| `pyzmq`        | 27.0.0   | BSD-3-Clause  | https://github.com/zeromq/pyzmq |
| `uvicorn`      | 0.29.0   | BSD-3-Clause  | https://github.com/encode/uvicorn |
| `fastapi`      | 0.111.0  | MIT           | https://github.com/fastapi/fastapi |
| `httpx`        | 0.27.0   | BSD-3-Clause  | https://github.com/encode/httpx |
| `livekit`      | 0.17.0   | Apache-2.0    | https://github.com/livekit/python-sdks |
| `livekit-api`  | 0.7.0    | Apache-2.0    | https://github.com/livekit/python-sdks |
| `numpy`        | 1.24.0   | BSD-3-Clause  | https://github.com/numpy/numpy |
| `websockets`   | 12.0     | BSD-3-Clause  | https://github.com/python-websockets/websockets |
| `fastmcp`      | >=2.0    | Apache-2.0    | https://github.com/jlowin/fastmcp |
| `loguru`       | >=0.7    | MIT           | https://github.com/Delgan/loguru |

## Python — `experimental/semantic-slam-source`

This branch carries a legacy semantic-SLAM source snapshot for evaluation and
migration reference. It is not the MCP integration on
`feat/semantic-slam-module`, and it does not distribute third-party model
repositories, model checkpoints, CUDA, TensorRT, or a Python environment.

The snapshot's direct dependency, legacy model-component, binary-runtime, and
source-attribution inventory is maintained with the snapshot at
[`experimental/semantic-slam-source/THIRD_PARTY_NOTICES.md`](experimental/semantic-slam-source/THIRD_PARTY_NOTICES.md).
That inventory reconciles shared components with `feat/semantic-slam-module`
and records the additional GroundingDINO, RAM, gRPC, and video-codec surface
required by the legacy code.

## Swift (iOS / visionOS client)

Used by `client-samples/ios-visionos/`. Resolved via Swift Package Manager.

| Package | Version | License | Upstream |
|---|---|---|---|
| `LiveKitClient` (`livekit/client-sdk-swift`)            | 2.13.0       | Apache-2.0   | https://github.com/livekit/client-sdk-swift |
| `livekit/webrtc-xcframework`                            | 144.7559.01  | MIT          | https://github.com/livekit/webrtc-xcframework |
| `livekit/livekit-uniffi-xcframework`                    | 0.0.5        | Apache-2.0   | https://github.com/livekit/livekit-uniffi-xcframework |
| `swift-protobuf` (`apple/swift-protobuf`)               | 1.36.1       | Apache-2.0   | https://github.com/apple/swift-protobuf |

## License texts

The full text of each SPDX license identifier referenced above is available at:

- **Apache-2.0**: https://www.apache.org/licenses/LICENSE-2.0 — also bundled
  with this repository as [`LICENSE`](LICENSE).
- **BSD-2-Clause**: https://opensource.org/license/bsd-2-clause
- **BSD-3-Clause**: https://opensource.org/license/bsd-3-clause
- **HPND**: https://opensource.org/license/historical-permission-notice-and-disclaimer (Pillow / PIL)
- **MIT**: https://opensource.org/license/mit
- **MPL-2.0**: https://www.mozilla.org/en-US/MPL/2.0/ (tqdm dual-license)
- **PSF**: https://docs.python.org/3/license.html (matplotlib)

Each upstream project repository linked above includes its own canonical
license file (typically `LICENSE`, `LICENSE.txt`, or `COPYING`).

## Updating this file

When adding, removing, or upgrading a third-party dependency:

1. Update the relevant manifest — `pyproject.toml` (Python),
   `Package.swift` (Swift), `gradle/libs.versions.toml` (Android), or the
   web client's manifest.
2. Update [`DEPENDENCIES.md`](DEPENDENCIES.md) — the internal/external
   dependency map.
3. Update this file with the new package name, version, license, and upstream
   URL.

All three changes belong in the same commit.
