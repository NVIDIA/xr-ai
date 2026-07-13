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
| `send()` topic param | `send(data, { reliable, topic })` — supports a per-message `topic` | `send(_ data:, reliable:topic:)` — topic-aware | `send(data, reliable)` — no topic | **idiom** | iOS/visionOS threads an optional `topic` into `DataPublishOptions`; the visionOS XR client publishes `xr.session.started` on it for render-mcp. |
| `AudioConfig` fields | `mode`, `echoCancellation` | `mode`, `highpassFilter`, `typingNoiseDetection` | `mode` only | **idiom** | Each platform exposes the DSP knobs its capture stack actually supports (WebRTC `echoCancellation`; AVAudioSession `highpassFilter`/`typingNoiseDetection`; Android relies on the `MicrophoneMode` preset). `mode` is the shared cross-platform contract. |
| `CameraConfig.default` facing | `default` → front-facing | `default` → front-facing | `DEFAULT` → back-facing | **idiom** | Mobile/desktop default to the selfie camera; Android's sample is built around the rear/primary camera (and a synthetic provider), so its default differs deliberately. The sample apps select a camera explicitly, so the differing default does not change observed behavior. |
| `stopAudio()` error contract | `async stopAudio()` — does not throw | `func stopAudio() async throws` | `suspend fun stopAudio()` — does not throw | **idiom** | Swift surfaces teardown failures via `throws`; JS and Kotlin do not (Kotlin has no checked exceptions and the call is best-effort cleanup). Callers on all three already treat stop as fire-and-forget. |
