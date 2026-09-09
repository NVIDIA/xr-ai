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
| `pocket-tts`   | 3.0.2    | MIT           | https://github.com/kyutai-labs/pocket-tts |
| `pydantic`     | >=2.10   | MIT           | https://github.com/pydantic/pydantic |
| `python-multipart` | >=0.0.9 | Apache-2.0 | https://github.com/Kludex/python-multipart |
| `websockets`   | 12.0     | BSD-3-Clause  | https://github.com/python-websockets/websockets |

### Pocket TTS model and voice

Pocket TTS 3.0.2 pins the following downloaded artifacts for the default
English configuration. XR AI does not redistribute them.

- Gated weights from `kyutai/pocket-tts` at revision
  `39592ff23c9ef80098bb74895d104c26275fe2c9`.
- Ungated fallback weights and the tokenizer from
  `kyutai/pocket-tts-without-voice-cloning` at revision
  `d29db7978e464fb90cb3359ee0c69a273b9142cc`.
- The predefined `bill_boerst` voice embedding from
  `kyutai/pocket-tts-without-voice-cloning` at revision
  `e81d79e8194ad4c7ce879c87a4258ef20cbf2487`.

Both model repositories are licensed under CC-BY-4.0. The `bill_boerst`
embedding derives from the CC0-1.0 Voice-Zero recording identified in the
[`kyutai/tts-voices` model card](https://huggingface.co/kyutai/tts-voices), but
the default service downloads the precomputed embedding rather than that
recording.

The model records reviewed at repository revisions
`492522650173a0653b7575cdc25ae09810e5d741` (`kyutai/pocket-tts`) and
`e81d79e8194ad4c7ce879c87a4258ef20cbf2487`
(`kyutai/pocket-tts-without-voice-cloning`) carried the following
acceptable-use text on 2026-09-09:

> Prohibited use: Use of our model must comply with all applicable laws and
> regulations and must not result in, involve, or facilitate any illegal,
> harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses
> include, without limitation, voice impersonation or cloning without explicit
> and lawful consent; misinformation, disinformation, or deception (including
> fake news, fraudulent calls, or presenting generated content as genuine
> recordings of real people or events); and the generation of unlawful,
> harmful, libelous, abusive, harassing, discriminatory, hateful, or
> privacy-invasive content. We disclaim all liability for any non-compliant use.

The exact UTF-8 text, without a trailing newline, has SHA-256
`8febd058c61bdeebf64b22d5a7bd3a78081e9b136a845fd98a2daf1de9d6381e`.
Review the live [gated model record](https://huggingface.co/api/models/kyutai/pocket-tts)
before accepting the terms because upstream may revise them. This release
accepts only the reviewed `bill_boerst` voice; it does not expose Pocket TTS's
other predefined voices or arbitrary voice-cloning inputs.

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
- **CC-BY-4.0**: https://creativecommons.org/licenses/by/4.0/
- **CC0-1.0**: https://creativecommons.org/publicdomain/zero/1.0/
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
