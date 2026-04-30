// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Configures microphone capture for a ``StreamSession``.
///
/// ## Presets
/// ```swift
/// .default          // Apple AUVoiceIO — best for voice calls
/// .softwareProcessing // WebRTC DSP stack
/// .raw              // no processing — ideal when the server handles DSP
/// .disabled         // microphone off
/// ```
public struct AudioConfig: Sendable, Equatable {

    // MARK: - Mode

    public enum MicrophoneMode: Sendable, Equatable {
        /// Apple's native Voice-Processing I/O (AUVoiceIO): echo cancel, AGC, noise suppression.
        /// Default on physical devices.
        case voiceProcessing

        /// WebRTC software DSP: echo cancellation, AGC, noise suppression via the WebRTC stack.
        /// Useful in simulator or when bypassing Apple's voice processing is needed.
        case softwareProcessing

        /// Raw PCM — no processing. Choose this for non-voice audio or when your server does DSP.
        case raw

        /// Microphone is not captured or published.
        case disabled
    }

    // MARK: - Properties

    public var mode: MicrophoneMode

    /// High-pass filter to cut sub-200 Hz rumble.
    /// Only effective with ``MicrophoneMode/softwareProcessing``.
    public var highpassFilter: Bool

    /// Keyboard / typing noise suppression.
    /// Only effective with ``MicrophoneMode/softwareProcessing``.
    public var typingNoiseDetection: Bool

    // MARK: - Presets

    public static let `default`          = AudioConfig()
    public static let softwareProcessing = AudioConfig(mode: .softwareProcessing)
    public static let raw                = AudioConfig(mode: .raw)
    public static let disabled           = AudioConfig(mode: .disabled)

    // MARK: - Init

    public init(
        mode: MicrophoneMode = .voiceProcessing,
        highpassFilter: Bool = false,
        typingNoiseDetection: Bool = false
    ) {
        self.mode = mode
        self.highpassFilter = highpassFilter
        self.typingNoiseDetection = typingNoiseDetection
    }
}
