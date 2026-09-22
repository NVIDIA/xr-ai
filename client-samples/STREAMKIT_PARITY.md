<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# StreamKit cross-platform API parity

The three StreamKit client SDKs (web `web/StreamKit/`, iOS/visionOS
`ios-visionos/StreamKit/`, Android `android/.../streamkit/`) are intended to
mirror one another. A full-repo audit found the divergences below. Each is
tagged **idiom** (intentional — follows the platform's idiom; leave as-is) or
**to-fix** (a real gap to close in future native-client work).

| Surface | web | iOS / visionOS | Android | Status | Notes |
|---|---|---|---|---|---|
| `send()` topic param | `send(data, { reliable, topic })` — supports a per-message `topic` | `send(_ data:, reliable:topic:)` — topic-aware | `send(data, reliable)` — no topic | **idiom** | iOS/visionOS threads an optional `topic` into `DataPublishOptions`; the visionOS XR client publishes `xr.session.started` on it for the render worker. |
| `AudioConfig` fields | `mode`, `echoCancellation` | `mode`, `highpassFilter`, `typingNoiseDetection` | `mode` only | **to-fix** | Android retains the shared mode names, but every enabled value currently uses LiveKit's default microphone capture. The sample UI therefore exposes no DSP-mode picker. |
| `CameraConfig.default` facing | `default` → front-facing | `default` → front-facing | `DEFAULT` → back-facing | **idiom** | Mobile/desktop default to the selfie camera; Android's sample is built around the rear/primary camera (and a synthetic provider), so its default differs deliberately. The sample apps select a camera explicitly, so the differing default does not change observed behavior. |
| `stopAudio()` error contract | `async stopAudio()` — does not throw | `func stopAudio() async throws` | `suspend fun stopAudio()` — does not throw | **idiom** | Swift surfaces teardown failures via `throws`; JS and Kotlin do not (Kotlin has no checked exceptions and the call is best-effort cleanup). Callers on all three already treat stop as fire-and-forget. |
| Network telemetry | `onNetworkMetrics` | `onNetworkMetrics` + published snapshot | `onNetworkMetrics` | **idiom** | All expose quality, RTT, and receive jitter from LiveKit-native stats on an approximately one-second cadence. Swift also publishes the latest value for SwiftUI observation, following the SDK's existing observable-state idiom. |
| Request-driven image capture | `onImageCaptureRequested` with `AbortSignal` | `onImageCaptureRequested` with task cancellation | `onImageCaptureRequested` with coroutine cancellation | **idiom** | All are explicit client opt-ins, return encoded JPEG/PNG/WebP over a targeted byte stream, and immediately reject requests when the handler is absent or fails. The graphical samples install the handler only for their persisted **On-demand images** camera mode; **Off** and **Live video** leave it unset. A mode change gates new requests but does not retroactively cancel one already accepted. |
