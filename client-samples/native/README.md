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
but that internal port must remain restricted to the XR AI host or a trusted
development network.

Refer to [Connecting clients](https://nvidia.github.io/xr-ai/latest/getting_started/clients.html#native-c)
for the current backend contract and limitations.
