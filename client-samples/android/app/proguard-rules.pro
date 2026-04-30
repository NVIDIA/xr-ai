# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# LiveKit — keep all classes used by reflection
-keep class io.livekit.** { *; }
-dontwarn io.livekit.**

# WebRTC native symbols
-keep class org.webrtc.** { *; }
-dontwarn org.webrtc.**

# OkHttp / okio used transitively by LiveKit
-dontwarn okhttp3.**
-dontwarn okio.**
