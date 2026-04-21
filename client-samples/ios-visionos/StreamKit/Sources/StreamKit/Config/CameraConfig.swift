import Foundation

/// Configures camera capture passed to ``StreamSession/startCamera(config:)``.
///
/// Resolution and frame-rate are intentionally not exposed: both iOS (AVFoundation)
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

    // MARK: - Presets

    public static let `default` = CameraConfig()
    public static let rear      = CameraConfig(position: .back)

    // MARK: - Init

    public init(position: Position = .front) {
        self.position = position
    }
}
