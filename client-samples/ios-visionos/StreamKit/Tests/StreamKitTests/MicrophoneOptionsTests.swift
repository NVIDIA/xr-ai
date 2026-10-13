// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import LiveKit
import Testing
@testable import StreamKit

@Suite("Microphone processing presets")
struct MicrophoneOptionsTests {
    @Test func softwareActuallySelectsSoftware() {
        let options = AudioCaptureOptions(from: .softwareProcessing)
        #expect(options.echoCancellation && options.autoGainControl && options.noiseSuppression)
        #expect(options.echoCancellationMode == .software)
        #expect(options.autoGainControlMode == .software)
        #expect(options.noiseSuppressionMode == .software)
        #expect(options.highpassFilterMode == .software)
        #expect(!options.highpassFilter && !options.typingNoiseDetection)
    }

    @Test func softwarePreservesOptionalFilters() {
        let options = AudioCaptureOptions(from: AudioConfig(
            mode: .softwareProcessing, highpassFilter: true, typingNoiseDetection: true
        ))
        #expect(options.highpassFilter && options.typingNoiseDetection)
        #expect(options.highpassFilterMode == .software)
    }

    @Test func voiceProcessingIsEnabled() {
        let options = AudioCaptureOptions(from: .default)
        #expect(options.echoCancellation && options.autoGainControl && options.noiseSuppression)
        #if targetEnvironment(simulator)
        #expect(options.echoCancellationMode == .software)
        #expect(options.autoGainControlMode == .software)
        #expect(options.noiseSuppressionMode == .software)
        #else
        #expect(options.echoCancellationMode == .platform)
        #expect(options.autoGainControlMode == .platform)
        #expect(options.noiseSuppressionMode == .platform)
        #endif
    }

    @Test(arguments: [AudioConfig.MicrophoneMode.raw, .disabled])
    func unprocessedPresetsDisableAllEffects(_ mode: AudioConfig.MicrophoneMode) {
        let options = AudioCaptureOptions(from: AudioConfig(mode: mode, highpassFilter: true, typingNoiseDetection: true))
        #expect(!options.echoCancellation && !options.autoGainControl && !options.noiseSuppression)
        #expect(!options.highpassFilter && !options.typingNoiseDetection)
    }

    @Test(arguments: [AudioConfig.default, .softwareProcessing, .raw, .disabled,
                      AudioConfig(mode: .softwareProcessing, highpassFilter: true, typingNoiseDetection: true)])
    func prewarmMatchesTrack(_ config: AudioConfig) {
        let options = AudioCaptureOptions(from: config)
        let recording = options.recordingProcessingOptions
        #expect(recording.echoCancellation == options.echoCancellation)
        #expect(recording.autoGainControl == options.autoGainControl)
        #expect(recording.noiseSuppression == options.noiseSuppression)
        #expect(recording.highpassFilter == options.highpassFilter)
        #expect(recording.echoCancellationMode == options.echoCancellationMode)
        #expect(recording.autoGainControlMode == options.autoGainControlMode)
        #expect(recording.noiseSuppressionMode == options.noiseSuppressionMode)
        #expect(recording.highpassFilterMode == options.highpassFilterMode)
    }
}
