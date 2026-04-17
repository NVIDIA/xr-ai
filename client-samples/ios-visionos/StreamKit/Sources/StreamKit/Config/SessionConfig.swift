import Foundation

/// Generic session configuration passed to ``StreamSession/connect(config:)``.
///
/// Network endpoint details (host, port, token) are backend-specific and live in
/// ``LiveKitConfig`` (or your custom backend's own config type). `SessionConfig`
/// captures only the cross-backend concerns: participant identity and media settings.
public struct SessionConfig: Sendable {

    /// Microphone capture settings.
    public var audio: AudioConfig

    /// Camera capture settings.
    public var camera: CameraConfig

    /// A unique identity for this participant.
    public var identity: String

    public static let `default` = SessionConfig()

    public init(
        audio: AudioConfig = .default,
        camera: CameraConfig = .default,
        identity: String = "participant-\(UInt32.random(in: 100_000...999_999))"
    ) {
        self.audio = audio
        self.camera = camera
        self.identity = identity
    }
}
