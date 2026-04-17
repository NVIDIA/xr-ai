/*
 * StreamKit — StreamSession
 *
 * The single public entry-point of the SDK.
 * Runs on @MainActor so it is safe to bind directly to SwiftUI Published properties.
 */

import Foundation

// MARK: - StreamSession

/// A transport-agnostic streaming session.
///
/// `StreamSession` wraps any ``StreamingBackend`` with a clean, SwiftUI-friendly API.
/// The choice of backend (e.g. LiveKit) is made at init time and is completely hidden
/// from the call-site.
///
/// ## Using a built-in backend
///
/// ```swift
/// let session = StreamSession(.liveKit(LiveKitConfig(host: "192.168.1.100", token: jwt)))
/// try await session.connect()
/// try await session.startCamera()
/// try await session.send(Data("hello".utf8))
/// ```
///
/// ## Using a custom backend
///
/// Conform to ``StreamingBackend`` and pass your instance to ``init(backend:)``:
///
/// ```swift
/// let session = StreamSession(backend: MyCustomBackend())
/// ```
///
/// ## visionOS
///
/// On visionOS, ``startCamera()`` requires an immersive space to be open in your app
/// **before** it is called. The SDK manages the ARKit session internally; your app only
/// needs to keep the immersive space active while streaming.
@MainActor
public final class StreamSession: ObservableObject {

    // MARK: - Published state

    /// Current connection state. Safe to observe from SwiftUI.
    @Published public private(set) var connectionState: ConnectionState = .disconnected

    // MARK: - Callbacks (alternative to Combine / observation)

    /// Called on the main actor when binary data is received from the server.
    public var onDataReceived: ((Data) -> Void)?

    /// Called on the main actor when the connection state changes.
    public var onConnectionStateChanged: ((ConnectionState) -> Void)?

    // MARK: - Private

    private var backend: any StreamingBackend

    // MARK: - Init

    /// Creates a session backed by one of the built-in transports.
    public init(_ backendConfig: BackendConfiguration) {
        backend = backendConfig.makeBackend()
        wireCallbacks()
    }

    /// Creates a session backed by a custom ``StreamingBackend`` implementation.
    ///
    /// ```swift
    /// let session = StreamSession(backend: MyCustomBackend())
    /// ```
    public init(backend: any StreamingBackend) {
        self.backend = backend
        wireCallbacks()
    }

    // MARK: - Connect / disconnect

    /// Connects using the current backend.
    ///
    /// - Parameter config: Room / participant metadata and media settings.
    public func connect(config: SessionConfig = .default) async throws {
        try await backend.connect(config: config)
    }

    /// Disconnects and releases all resources.
    public func disconnect() async {
        await backend.disconnect()
    }

    // MARK: - Camera

    /// Starts camera capture and publishes a video track.
    ///
    /// On **visionOS** an immersive space must already be open.
    public func startCamera() async throws {
        try await backend.startCamera()
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
