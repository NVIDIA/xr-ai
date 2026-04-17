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
/// protocol, so the call-site never depends on a specific transport technology.
///
/// ## Implementing a custom backend
///
/// ```swift
/// final class MyBackend: StreamingBackend {
///
///     var onConnectionStateChanged: (@Sendable (ConnectionState) -> Void)?
///     var onDataReceived: (@Sendable (Data) -> Void)?
///
///     func connect(config: SessionConfig) async throws {
///         // establish connection using config.roomName / config.identity …
///         onConnectionStateChanged?(.connected)
///     }
///
///     func disconnect() async { … }
///     func startCamera() async throws { … }
///     func stopCamera() async throws { … }
///     func send(_ data: Data, reliable: Bool) async throws { … }
/// }
///
/// // Then:
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
    var onDataReceived: (@Sendable (Data) -> Void)? { get set }

    // MARK: - Lifecycle

    /// Establish a connection using the provided session configuration.
    ///
    /// - Parameter config: Room/participant metadata and media settings.
    ///   Network endpoint details are supplied at backend-construction time
    ///   (see ``BackendConfiguration``).
    func connect(config: SessionConfig) async throws

    /// Cleanly disconnect and release all resources.
    func disconnect() async

    // MARK: - Media

    /// Begin capturing the local camera and streaming it to remote participants.
    ///
    /// - Note: On **visionOS** this requires an immersive space to already be open
    ///   in your app. The backend manages the ARKit session internally.
    func startCamera() async throws

    /// Stop camera capture and streaming.
    func stopCamera() async throws

    // MARK: - Data channel

    /// Send binary data to remote participants.
    ///
    /// - Parameters:
    ///   - data: Payload bytes. Keep individual messages under the transport's MTU
    ///     (15 KB for LiveKit's WebRTC data channel).
    ///   - reliable: `true` for ordered, guaranteed delivery (default).
    ///     `false` for low-latency best-effort delivery.
    func send(_ data: Data, reliable: Bool) async throws
}
