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

Used by `agent-sdk/`, `utils/`, `services/`,
`agent-samples/` and `tests/`.
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
| `nemo-relay`   | >=0.7.2,<0.8 | Apache-2.0  | https://github.com/NVIDIA/NeMo-Relay |
| `Pillow`       | 10.0.0   | HPND          | https://github.com/python-pillow/Pillow |
| `pipecat-ai`   | >=1.3    | BSD-2-Clause  | https://github.com/pipecat-ai/pipecat |
| `pydantic`     | >=2.10   | MIT           | https://github.com/pydantic/pydantic |
| `python-multipart` | >=0.0.9 | Apache-2.0 | https://github.com/Kludex/python-multipart |
| `websockets`   | 12.0     | BSD-3-Clause  | https://github.com/python-websockets/websockets |

## Swift (iOS / visionOS client)

Used by `client-samples/ios-visionos/`. Resolved via Swift Package Manager.

| Package | Version | License | Upstream |
|---|---|---|---|
| `LiveKitClient` (`livekit/client-sdk-swift`)            | 2.13.0       | Apache-2.0   | https://github.com/livekit/client-sdk-swift |
| `livekit/webrtc-xcframework`                            | 144.7559.01  | MIT          | https://github.com/livekit/webrtc-xcframework |
| `livekit/livekit-uniffi-xcframework`                    | 0.0.5        | Apache-2.0   | https://github.com/livekit/livekit-uniffi-xcframework |
| `swift-protobuf` (`apple/swift-protobuf`)               | 1.36.1       | Apache-2.0   | https://github.com/apple/swift-protobuf |

## Android build toolchain (`client-samples/android/`)

Building the Android sample with the Android Gradle Plugin (declared in
`client-samples/android/gradle/libs.versions.toml`) places dual-licensed
artifacts on the build classpath, including the following, for which `xr-ai`
records a license election:

| Package | Version | Dual license | Elected | Upstream |
|---|---|---|---|---|
| Java Native Access (`net.java.dev.jna:jna`, `jna-platform`) | 5.6.0 | LGPL-2.1-or-later OR Apache-2.0 | Apache-2.0 | https://github.com/java-native-access/jna |
| `javax.annotation:javax.annotation-api`                 | 1.3.2 | CDDL-1.1 OR GPL-2.0-with-classpath-exception | CDDL-1.1 | https://github.com/eclipse-ee4j/common-annotations-api |

Neither artifact is bundled into the APK produced by the current build
configuration (verified against the resolved `releaseRuntimeClasspath`), and
`xr-ai` does not redistribute them; the election records the terms under
which each artifact is used at build time.

## License texts

The full text of each SPDX license identifier that applies to `xr-ai`'s use
of the software above (for dual-licensed artifacts, the elected license) is
available at:

- **Apache-2.0**: https://www.apache.org/licenses/LICENSE-2.0 — also bundled
  with this repository as [`LICENSE`](LICENSE).
- **BSD-3-Clause**: https://opensource.org/license/bsd-3-clause
- **BSD-2-Clause**: https://opensource.org/license/bsd-2-clause
- **HPND**: https://opensource.org/license/historical-ntu-disclaimer
- **MIT**: https://opensource.org/license/mit
- **CDDL-1.1**: https://spdx.org/licenses/CDDL-1.1.html

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
