// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

namespace streamkit {

/// Configures camera capture passed to StreamSession::StartCamera().
///
/// Resolution and frame-rate are intentionally not exposed: the LiveKit
/// backend negotiates the best supported format with the hardware
/// automatically (matching the iOS and Android behaviour).
///
/// ## Platform contract for `facing` and `device_id`
///
/// These fields are only honoured by backends that open a camera themselves
/// (iOS, Android, Web — all platforms with a portable camera-open API). The
/// built-in C++ `LiveKitBackend` has no portable way to open a camera, so it
/// **ignores both fields** and expects the host to capture externally and
/// push frames via `FrameSink::InjectVideoFrame`. The host's own camera-open
/// code chooses front vs back. The fields stay on the struct so the
/// cross-platform `CameraConfig` shape is identical everywhere — silently
/// inert on backends that can't act on them.
///
/// Mirror of Swift `CameraConfig` and Kotlin `CameraConfig`.
struct CameraConfig {

    enum class Facing {
        kFront,
        kBack,
    };

    /// Which camera to use. Ignored when device_id is set.
    Facing facing = Facing::kFront;

    /// Pin to a specific device by its platform identifier.
    /// When set, this takes precedence over `facing`.
    std::optional<std::string> device_id;

    // ── Presets ────────────────────────────────────────────────────────────

    static CameraConfig Default() { return {}; }
    static CameraConfig Rear()    { return {Facing::kBack, std::nullopt}; }
};

} // namespace streamkit
