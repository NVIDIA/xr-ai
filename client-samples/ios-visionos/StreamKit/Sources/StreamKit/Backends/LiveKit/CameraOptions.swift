// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import LiveKit

/// Validate when a publication consumes the policy, not on every injected frame.
/// Omitted fields inherit the connection's SDK options; unrelated fields survive.
func cameraPublishOptions(_ encoding: CameraEncodingConfig?,
                          defaults: VideoPublishOptions,
                          dimensions: Dimensions? = nil) throws -> VideoPublishOptions? {
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
    var layers = defaults.simulcastLayers
    if encoding.simulcast ?? defaults.simulcast {
        if layers.isEmpty, let dimensions {
            let aspect = Double(max(dimensions.width, dimensions.height)) / Double(min(dimensions.width, dimensions.height))
            layers = abs(aspect - 16.0 / 9.0) < abs(aspect - 4.0 / 3.0)
                ? [.presetH180_169, .presetH360_169] : [.presetH180_43, .presetH360_43]
        }
        let budgets = layers.map(\.encoding.maxBitrate)
        let largest = budgets.max() ?? 0
        let reduced = largest > encoding.maxBitrateBps
        let scaled = try budgets.map { budget in
            guard reduced else { return budget }
            guard budget > 0 else {
                throw StreamError.invalidCameraEncoding("Simulcast bitrate limits must be positive integers.")
            }
            // Full-width integer arithmetic avoids overflow and floating-point
            // collapse when scaling caller-supplied SDK budgets.
            let product = UInt(budget).multipliedFullWidth(by: UInt(encoding.maxBitrateBps))
            return Int(UInt(largest).dividingFullWidth(product).quotient)
        }
        guard !reduced || (scaled.allSatisfy { $0 > 0 } && budgets.indices.allSatisfy { i in
            budgets.indices.allSatisfy { j in budgets[i] == budgets[j] || scaled[i] != scaled[j] }
        }) else {
            throw StreamError.invalidCameraEncoding("Primary bitrate cannot preserve the SDK lower-layer budget ordering.")
        }
        layers = layers.enumerated().map { i, layer in
            VideoParameters(dimensions: layer.dimensions, encoding: VideoEncoding(
                maxBitrate: scaled[i],
                maxFps: layer.encoding.maxFps,
                bitratePriority: layer.encoding.bitratePriority,
                networkPriority: layer.encoding.networkPriority
            ))
        }
    }
    return VideoPublishOptions(
        name: defaults.name,
        encoding: VideoEncoding(maxBitrate: encoding.maxBitrateBps, maxFps: encoding.maxFramerate),
        screenShareEncoding: defaults.screenShareEncoding,
        simulcast: encoding.simulcast ?? defaults.simulcast,
        simulcastLayers: layers,
        screenShareSimulcastLayers: defaults.screenShareSimulcastLayers,
        preferredCodec: defaults.preferredCodec,
        preferredBackupCodec: defaults.preferredBackupCodec,
        degradationPreference: preference,
        streamName: defaults.streamName
    )
}
