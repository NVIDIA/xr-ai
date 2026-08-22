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
./build/bin/streamkit_sample --host 192.168.1.100 --token <jwt>
```

Omit `LIVEKIT_SDK_ROOT` for stub mode. Build unit tests with
`-DSTREAMKIT_BUILD_TESTS=ON` and run `ctest --test-dir build`.

Refer to [Connecting clients](../../docs/source/getting_started/clients.md#native-c)
for the current backend contract and limitations.
