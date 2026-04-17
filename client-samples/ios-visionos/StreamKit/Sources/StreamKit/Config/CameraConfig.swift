import Foundation

/// Configures camera capture for a ``StreamSession``.
///
/// Resolution and frame-rate are intentionally not exposed: both iOS (AVFoundation)
/// and visionOS (ARKit `CameraFrameProvider`) negotiate the best supported format
/// with the hardware automatically. Specifying explicit values rarely produces the
/// expected result and adds unnecessary API surface.
///
/// On **visionOS** the ``position`` field is ignored; the SDK always uses the
/// main passthrough camera via ARKit's `CameraFrameProvider`.
///
/// Camera capture on visionOS requires an **open immersive space** before calling
/// ``StreamSession/startCamera()``. Your app manages the immersive space lifecycle;
/// the SDK only manages the ARKit session internally.
public struct CameraConfig: Sendable, Equatable {

    // MARK: - Camera position (iOS only)

    public enum Position: Sendable, Equatable {
        case front
        case back
    }

    // MARK: - Properties

    /// Whether the camera should be streamed at all.
    public var enabled: Bool

    /// Which camera to use. Ignored on visionOS.
    public var position: Position

    // MARK: - Presets

    public static let `default` = CameraConfig()
    public static let disabled  = CameraConfig(enabled: false)
    public static let rear      = CameraConfig(position: .back)

    // MARK: - Init

    public init(enabled: Bool = true, position: Position = .front) {
        self.enabled  = enabled
        self.position = position
    }
}
