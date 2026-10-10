// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import LiveKit

/// Validate when a publication consumes the policy, not on every injected frame.
/// Omitted fields inherit the connection's SDK options; unrelated fields survive.
func cameraPublishOptions(_ encoding: CameraEncodingConfig?,
                          defaults: VideoPublishOptions) throws -> VideoPublishOptions? {
    guard let encoding else { return nil }
    guard encoding.maxBitrateBps > 0, encoding.maxFramerate > 0 else {
        throw StreamError.invalidCameraEncoding("Bitrate and frame rate must be positive.")
    }
    let preference: DegradationPreference
    switch encoding.qualityPreference {
    case .detail: preference = .maintainResolution
    case .motion: preference = .maintainFramerate
    case .balanced: preference = .balanced
    case nil: preference = defaults.degradationPreference
    }
    return VideoPublishOptions(
        name: defaults.name,
        encoding: VideoEncoding(maxBitrate: encoding.maxBitrateBps, maxFps: encoding.maxFramerate),
        screenShareEncoding: defaults.screenShareEncoding,
        simulcast: encoding.simulcast ?? defaults.simulcast,
        simulcastLayers: defaults.simulcastLayers,
        screenShareSimulcastLayers: defaults.screenShareSimulcastLayers,
        preferredCodec: defaults.preferredCodec,
        preferredBackupCodec: defaults.preferredBackupCodec,
        degradationPreference: preference,
        streamName: defaults.streamName
    )
}
