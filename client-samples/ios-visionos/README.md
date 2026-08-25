<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# iOS and visionOS StreamKit sample

Checked-in SwiftUI application and local StreamKit package for LiveKit audio,
camera, data, participant status, network metrics, and native CloudXR on Apple
Vision Pro.

Open `StreamKitSample.xcworkspace` in Xcode, select an iOS or visionOS
destination and signing team, then build and run.

<!-- Compatibility anchors for headings consolidated into the documentation. -->
<a id="ai-sdk-sample"></a><a id="repository-layout"></a>
<a id="creating-the-xcode-project"></a><a id="1-new-project"></a>
<a id="2-add-destinations"></a><a id="3-add-the-streamkit-package"></a>
<a id="4-replace-the-generated-source-files"></a>
<a id="5-infoplist-entries"></a>
<a id="6-visionos-passthrough-camera--device-only"></a>
<a id="bundling-enterpriselicense"></a><a id="7-build-and-run"></a>
<a id="simulator-camera-feed"></a>
<a id="trusting-the-hubs-self-signed-cert-one-time-per-device"></a>
<a id="enable-full-trust-toggle-does-not-appear"></a>
<a id="connection-fails-with-errsslbadcert---1202-after-the-cert-is-trusted"></a>
<a id="tls-succeeds-but-the-room-rejects-the-token-with-401"></a>
<a id="microphone-fails-to-start-with-a-timed-out-error"></a>
<a id="orange-mic-indicator-stays-lit-after-stopping-audio"></a>
<a id="mic--camera-go-dead-while-the-ui-still-says-on"></a>
<a id="launching-xr-cloudxr"></a><a id="two-parallel-transports"></a>
<a id="server-prerequisite-change-nv_device_profile-to-auto-native"></a>
<a id="cloudxrkit-spm-dependency"></a><a id="apple-developer-program"></a>
<a id="on-device-flow"></a><a id="cert--trust-notes"></a>
<a id="render-target"></a><a id="quick-start-usage"></a>
<a id="adding-a-custom-backend"></a><a id="token-server-livekit"></a>

Refer to [Connecting clients](https://nvidia.github.io/xr-ai/latest/getting_started/clients.html#ios-visionos)
for platform requirements, enterprise camera access, certificate trust,
connection, and CloudXR instructions.
