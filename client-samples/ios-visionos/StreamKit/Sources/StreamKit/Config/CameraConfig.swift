// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Encoder tradeoff under resource or bandwidth pressure; not a quality guarantee.
public enum VideoQualityPreference: Sendable, Equatable {
    case balanced
    /// Favor resolution over frame rate.
    case detail
    /// Favor frame rate over resolution.
    case motion
}

/// Optional publishing policy. Capture format is unchanged.
/// Limits apply to each encoded stream, not aggregate network traffic.
/// Backends apply supported settings. Restart the camera to apply changes.
public struct CameraEncodingConfig: Sendable, Equatable {
    /// Positive ceiling in bits per second. Defaults to 3 Mbps when opting in.
    public var maxBitrateBps: Int
    /// Positive integer ceiling in frames per second. Actual FPS may be lower.
    public var maxFramerate: Int
    /// Nil preserves the backend's adaptation preference.
    public var qualityPreference: VideoQualityPreference?
    /// Publish multiple resolutions when supported. Nil preserves backend defaults.
    public var simulcast: Bool?

    /// Favors resolution and disables simulcast; cannot guarantee minimum resolution.
    public static let detail = CameraEncodingConfig(qualityPreference: .detail, simulcast: false)
    public static let motion = CameraEncodingConfig(qualityPreference: .motion)
    public static let balanced = CameraEncodingConfig(qualityPreference: .balanced)

    public init(maxBitrateBps: Int = 3_000_000, maxFramerate: Int = 30,
                qualityPreference: VideoQualityPreference? = nil, simulcast: Bool? = nil) {
        self.maxBitrateBps = maxBitrateBps
        self.maxFramerate = maxFramerate
        self.qualityPreference = qualityPreference
        self.simulcast = simulcast
    }
}

/// Configures camera capture passed to ``StreamSession/startCamera(config:)``.
///
/// Capture resolution and frame-rate are intentionally not exposed: both iOS (AVFoundation)
/// and visionOS (ARKit `CameraFrameProvider`) negotiate the best supported format
/// with the hardware automatically.
///
/// On **visionOS** ``position`` is ignored; the SDK always uses the main
/// passthrough camera via ARKit's `CameraFrameProvider`, which requires an open
/// immersive space before ``StreamSession/startCamera(config:)`` is called.
public struct CameraConfig: Sendable, Equatable {

    // MARK: - Camera position (iOS only)

    public enum Position: Sendable, Equatable {
        case front
        case back
    }

    /// Which camera to use. Ignored on visionOS.
    public var position: Position

    /// Optional publish-side policy. Nil preserves backend defaults.
    public var encoding: CameraEncodingConfig?

    // MARK: - Presets

    public static let `default` = CameraConfig()
    public static let rear      = CameraConfig(position: .back)

    // MARK: - Init

    public init(position: Position = .front, encoding: CameraEncodingConfig? = nil) {
        self.position = position
        self.encoding = encoding
    }
}
