/*
 * StreamKit — StreamSession
 *
 * The single public entry-point of the SDK.
 * Runs on @MainActor so it is safe to bind directly to SwiftUI.
 */

import Foundation

// MARK: - StreamSession

/// A transport-agnostic streaming session.
///
/// `StreamSession` wraps any ``StreamingBackend`` with a clean, SwiftUI-friendly API.
///
/// ## Lifecycle
///
/// ```swift
/// // 1. Connect — WebRTC peer connection + data channel only
/// try await session.connect(config: SessionConfig(identity: "ipad-1"))
///
/// // 2. Start media independently — each throws its own error, never drops the connection
/// try await session.startAudio()
/// try await session.startCamera()
///
/// // 3. Send / receive data
/// session.onDataReceived = { data in … }
/// try await session.send(Data("hello".utf8))
///
/// // 4. Stop media / disconnect
/// try await session.stopAudio()
/// try await session.stopCamera()
/// await session.disconnect()
/// ```
@MainActor
public final class StreamSession: ObservableObject {

    // MARK: - Published state

    /// Current connection state. Safe to observe from SwiftUI.
    @Published public private(set) var connectionState: ConnectionState = .disconnected

    // MARK: - Callbacks

    /// Called on the main actor when the connection state changes.
    public var onConnectionStateChanged: ((ConnectionState) -> Void)?

    /// Called on the main actor when binary data is received.
    public var onDataReceived: ((Data) -> Void)?

    // MARK: - Private

    private var backend: any StreamingBackend

    // MARK: - Init

    /// Creates a session backed by one of the built-in transports.
    public init(_ backendConfig: BackendConfiguration) {
        backend = backendConfig.makeBackend()
        wireCallbacks()
    }

    /// Creates a session backed by a custom ``StreamingBackend`` implementation.
    public init(backend: any StreamingBackend) {
        self.backend = backend
        wireCallbacks()
    }

    // MARK: - Connection

    /// Establishes a WebRTC peer connection and data channel.
    /// Does **not** start audio or camera — call ``startAudio(config:)`` and
    /// ``startCamera(config:)`` explicitly once connected.
    public func connect(config: SessionConfig = .default) async throws {
        try await backend.connect(config: config)
    }

    /// Disconnects and releases all resources.
    public func disconnect() async {
        await backend.disconnect()
    }

    // MARK: - Audio

    /// Starts microphone capture and publishes an audio track.
    ///
    /// Throws if the audio device is unavailable. Never drops the connection.
    public func startAudio(config: AudioConfig = .default) async throws {
        try await backend.startAudio(config: config)
    }

    /// Stops microphone capture.
    public func stopAudio() async throws {
        try await backend.stopAudio()
    }

    // MARK: - Camera

    /// Starts camera capture and publishes a video track.
    ///
    /// On **visionOS** an immersive space must already be open.
    /// Throws if the camera is unavailable. Never drops the connection.
    public func startCamera(config: CameraConfig = .default) async throws {
        try await backend.startCamera(config: config)
    }

    /// Stops camera capture.
    public func stopCamera() async throws {
        try await backend.stopCamera()
    }

    // MARK: - Data channel

    /// Sends binary data to all participants.
    ///
    /// - Parameters:
    ///   - data: Payload. Keep individual messages ≤ 15 KB on most transports.
    ///   - reliable: Ordered + guaranteed delivery when `true` (default).
    public func send(_ data: Data, reliable: Bool = true) async throws {
        try await backend.send(data, reliable: reliable)
    }

    // MARK: - Private

    private func wireCallbacks() {
        backend.onConnectionStateChanged = { [weak self] state in
            Task { @MainActor [weak self] in
                guard let self else { return }
                connectionState = state
                onConnectionStateChanged?(state)
            }
        }
        backend.onDataReceived = { [weak self] data in
            Task { @MainActor [weak self] in
                self?.onDataReceived?(data)
            }
        }
    }
}
