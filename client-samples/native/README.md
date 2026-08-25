<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Native C++ StreamKit sample

C++20 StreamKit implementation backed by the LiveKit C++ SDK. It supports
participant data, status, telemetry, and host-injected audio and video frames.

```bash
cmake -S . -B build -DLIVEKIT_SDK_ROOT=/path/to/livekit-cpp-sdk
cmake --build build
./build/bin/streamkit_sample --host 192.168.1.100 --port 8080 \
  --secure --token <jwt>
```

Omit `LIVEKIT_SDK_ROOT` for stub mode. Build unit tests with
`-DSTREAMKIT_BUILD_TESTS=ON` and run `ctest --test-dir build`.

Install the hub certificate in the native host's trust store before connecting.
Refer to the
[Linux trust procedure](https://nvidia.github.io/xr-ai/latest/getting_started/networking.html#linux-native-certificate-trust).
The sample also defaults to insecure port 7880 for direct LiveKit SDK debugging,
but that plaintext port must remain restricted to the XR AI host or a trusted
development network.

<!-- Compatibility anchors for headings consolidated into the documentation. -->
<a id="streamkit-for-native-c--livekit-backed-client"></a>
<a id="running-the-tests"></a><a id="constraints-in-the-current-native-backend"></a>
<a id="what-streamkit-is-and-isnt"></a>
<a id="what-streamkit-adds-on-top-of-livekit"></a>
<a id="1-a-single-entry-point-with-decoupled-media"></a>
<a id="2-a-typed-connectionstate-enum"></a><a id="3-typed-errors"></a>
<a id="4-the-agent-status-channel"></a>
<a id="5-audioconfig-and-microphonemode"></a><a id="6-token-acquisition"></a>
<a id="7-frame-injection-optional-for-external-video-sources"></a>
<a id="the-streamingbackend-interface-you-need-to-implement"></a>
<a id="implementing-livekitbackend-in-c"></a>
<a id="what-you-get-for-free-once-the-backend-is-done"></a>

Refer to [Connecting clients](https://nvidia.github.io/xr-ai/latest/getting_started/clients.html#native-c)
for the current backend contract and limitations.
