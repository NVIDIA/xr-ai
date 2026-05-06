// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace streamkit {

/// Configures microphone capture passed to StreamSession::StartAudio().
///
/// Mirror of Swift `AudioConfig` and Kotlin `AudioConfig`.
struct AudioConfig {

    /// Controls which audio processing pipeline is applied before the PCM
    /// audio is handed to WebRTC.
    enum class MicrophoneMode {
        /// Platform hardware voice processing (e.g. platform AEC, AGC, NR).
        /// Disable WebRTC's own DSP to avoid double-processing.
        /// Best choice for voice calls on most platforms.
        kVoiceProcessing,

        /// WebRTC software DSP: echo cancellation, AGC, noise suppression.
        /// Useful when hardware voice processing is unavailable.
        kSoftwareProcessing,

        /// Raw PCM — no processing. Choose this when the server-side agent
        /// handles DSP, or for non-voice audio.
        kRaw,

        /// Microphone is not captured or published.
        kDisabled,
    };

    MicrophoneMode mode = MicrophoneMode::kVoiceProcessing;

    /// High-pass filter to cut sub-200 Hz rumble.
    /// Only effective with kSoftwareProcessing.
    bool highpass_filter = false;

    /// Keyboard / typing noise suppression.
    /// Only effective with kSoftwareProcessing.
    bool typing_noise_detection = false;

    // ── Presets ────────────────────────────────────────────────────────────

    static AudioConfig Default()            { return {}; }
    static AudioConfig SoftwareProcessing() { return {MicrophoneMode::kSoftwareProcessing}; }
    static AudioConfig Raw()                { return {MicrophoneMode::kRaw}; }
    static AudioConfig Disabled()           { return {MicrophoneMode::kDisabled}; }
};

} // namespace streamkit
