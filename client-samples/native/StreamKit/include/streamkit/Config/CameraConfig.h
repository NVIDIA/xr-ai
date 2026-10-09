// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace streamkit {

/// Encoder tradeoff under resource or bandwidth pressure; not a quality guarantee.
enum class VideoQualityPreference { kBalanced, kDetail, kMotion };

/// Optional publish-side encoding controls. Capture format is unchanged.
///
/// Limits apply to each encoded stream, not aggregate network traffic. Backends
/// apply supported settings. The C++ LiveKitBackend consumes it through FrameSink
/// to specify publish options before the first frame creates the
/// LocalVideoTrack. Restart the camera to apply changes.
struct CameraEncodingConfig {
    /// Ceiling in bits per second; zero preserves the SDK default.
    std::uint64_t max_bitrate_bps = 0;
    /// Ceiling in frames per second; zero preserves the SDK default.
    double max_framerate = 0.0;
    /// Publish multiple resolutions when supported. Unset preserves backend defaults.
    std::optional<bool> simulcast;
    /// Unset preserves the backend's adaptation preference.
    std::optional<VideoQualityPreference> quality_preference;

    /// Favors resolution and disables simulcast; cannot guarantee minimum resolution.
    static CameraEncodingConfig Detail() {
        return {3'000'000, 30.0, false, VideoQualityPreference::kDetail};
    }
    static CameraEncodingConfig Motion() {
        return {3'000'000, 30.0, std::nullopt, VideoQualityPreference::kMotion};
    }
    static CameraEncodingConfig Balanced() {
        return {3'000'000, 30.0, std::nullopt, VideoQualityPreference::kBalanced};
    }
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
