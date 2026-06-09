// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace streamkit {

/// Optional publish-side encoding controls for externally captured video.
///
/// Backends that open their own platform camera may ignore this. The C++
/// LiveKitBackend consumes it because callers push frames through FrameSink
/// and need to specify publish options before the first frame creates the
/// LocalVideoTrack.
struct CameraEncodingConfig {
    std::uint64_t max_bitrate_bps = 0;
    double max_framerate = 0.0;
    std::optional<bool> simulcast;
};

/// Configures camera capture passed to StreamSession::StartCamera().
///
/// Capture resolution is intentionally not exposed: backends that open a
/// camera negotiate the best supported format with the hardware automatically
/// (matching the iOS and Android behaviour). `encoding` only controls
/// publish-side media options after capture.
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

    /// Optional publish-side encoding controls for externally captured video.
    std::optional<CameraEncodingConfig> encoding;

    // ── Presets ────────────────────────────────────────────────────────────

    static CameraConfig Default() { return {}; }
    static CameraConfig Rear()    { return {Facing::kBack, std::nullopt, std::nullopt}; }
};

} // namespace streamkit
