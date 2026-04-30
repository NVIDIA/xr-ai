/*
 * StreamKit — StreamingBackend protocol
 *
 * This is the single seam between StreamSession and any networking technology.
 * The SDK ships with a LiveKit implementation; to use a different transport
 * (proprietary streaming SDK, custom WebRTC, etc.) just conform to this protocol
 * and pass your instance to StreamSession(backend:).
 */

import Foundation

// MARK: - StreamingBackend

/// The contract that every networking backend must satisfy.
///
/// ``StreamSession`` delegates all network operations to an object conforming to this
/// protocol. The call-site never depends on a specific transport technology.
///
/// ## Lifecycle
///
/// ```
/// connect()            → WebRTC peer connection + data channel only
/// startAudio(config:)  → microphone capture + publish  (independent, throws on failure)
/// startCamera(config:) → camera capture + publish      (independent, throws on failure)
/// send(_:reliable:)    → data channel message
/// stopAudio()          → stop microphone
/// stopCamera()         → stop camera
/// disconnect()         → tear down everything
/// ```
///
/// Audio and camera failures never affect the connection itself.
///
/// ## Implementing a custom backend
///
/// ```swift
/// final class MyBackend: StreamingBackend {
///
///     var onConnectionStateChanged: (@Sendable (ConnectionState) -> Void)?
///     var onDataReceived: (@Sendable (_ topic: String, _ data: Data) -> Void)?
///     var onAgentStatus: (@Sendable (String) -> Void)?
///
///     func connect(config: SessionConfig) async throws {
///         onConnectionStateChanged?(.connected)
///     }
///     func disconnect() async { … }
///     func startAudio(config: AudioConfig) async throws { … }
///     func stopAudio() async throws { … }
///     func startCamera(config: CameraConfig) async throws { … }
///     func stopCamera() async throws { … }
///     func send(_ data: Data, reliable: Bool) async throws { … }
/// }
///
/// let session = StreamSession(backend: MyBackend())
/// ```
public protocol StreamingBackend: AnyObject, Sendable {

    // MARK: - Event hooks
    //
    // StreamSession sets these before calling connect().
    // Invoke them from any thread; StreamSession re-dispatches to @MainActor.

    /// Fired when the connection state changes.
    var onConnectionStateChanged: (@Sendable (ConnectionState) -> Void)? { get set }

    /// Fired when binary data arrives from the remote end.
    var onDataReceived: (@Sendable (_ topic: String, _ data: Data) -> Void)? { get set }

    /// Fired when an agent publishes a status update.
    /// Common values: `"idle"`, `"processing"`.
    /// These messages are delivered on the reserved `_agent.status` topic and are
    /// **not** forwarded to ``onDataReceived``.
    var onAgentStatus: (@Sendable (String) -> Void)? { get set }

    // MARK: - Connection

    /// Establish a WebRTC peer connection and data channel.
    /// Does **not** start audio or camera capture.
    func connect(config: SessionConfig) async throws

    /// Cleanly disconnect and release all resources.
    func disconnect() async

    // MARK: - Audio

    /// Begin microphone capture and publish an audio track.
    ///
    /// Throws if the audio device is unavailable. Does not affect the connection.
    func startAudio(config: AudioConfig) async throws

    /// Stop microphone capture.
    func stopAudio() async throws

    // MARK: - Camera

    /// Begin camera capture and publish a video track.
    ///
    /// On **visionOS** an immersive space must already be open.
    /// Throws if the camera is unavailable. Does not affect the connection.
    func startCamera(config: CameraConfig) async throws

    /// Stop camera capture.
    func stopCamera() async throws

    // MARK: - Data channel

    /// Send binary data to remote participants.
    ///
    /// - Parameters:
    ///   - data: Payload bytes.
    ///   - reliable: `true` for ordered, guaranteed delivery (default).
    func send(_ data: Data, reliable: Bool) async throws
}
