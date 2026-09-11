<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# StreamKit cross-platform API parity

The four StreamKit client SDKs (web `web/StreamKit/`, iOS/visionOS
`ios-visionos/StreamKit/`, Android `android/.../streamkit/`, and native C++) are intended to
mirror one another. The table tracks selected shared surfaces. **parity** means
the capability is equivalent, **idiom** marks an intentional platform-specific
shape, and **to-fix** identifies a current gap.

| Surface | web | iOS / visionOS | Android | Native C++ | Status | Notes |
|---|---|---|---|---|---|---|
| File transfer | `sendBytes`, `sendFile` | `sendBytes`, `sendFile` | `sendBytes`, `sendFile` | `SendBytes`, `SendFile` | **parity** | All use the reserved LiveKit byte-stream mapping and return local completion metadata. |
| `send()` topic param | `send(data, { reliable, topic })` | `send(_ data:, reliable:topic:)` | `send(data, reliable)` | `Send(data, reliable, topic)` | **to-fix** | Android packet sending has no application-topic parameter. |
| `AudioConfig` fields | `mode`, `echoCancellation` | `mode`, `highpassFilter`, `typingNoiseDetection` | `mode` only | `mode` | **to-fix** | Android retains the shared mode names, but every enabled value currently uses LiveKit's default microphone capture. |
| `CameraConfig.default` facing | front-facing | front-facing | back-facing | capture-source owned | **idiom** | Each sample follows its platform's primary capture path. |
| `stopAudio()` error contract | does not throw | `throws` | does not throw | throws | **idiom** | Each platform follows its language and transport teardown behavior. |
| Network telemetry | callback | callback + published snapshot | callback | callback | **idiom** | All expose quality, RTT, and receive jitter from LiveKit-native stats. |
