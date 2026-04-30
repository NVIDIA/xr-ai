// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * StreamKit — StreamSession
 *
 * The single public entry-point of the SDK.
 * Runs on @MainActor so it is safe to bind directly to SwiftUI.
 */

import CoreMedia
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

    /// Latest agent status. `nil` when disconnected or no status has been received yet.
    /// Common values: `"idle"`, `"processing"`.
    @Published public private(set) var agentStatus: String?

    // MARK: - Callbacks

    /// Called on the main actor when the connection state changes.
    public var onConnectionStateChanged: ((ConnectionState) -> Void)?

    /// Called on the main actor when data is received.
    /// `topic` identifies the logical channel; `data` is the raw payload.
    public var onDataReceived: ((_ topic: String, _ data: Data) -> Void)?

    /// Called on the main actor when the agent publishes a status update.
    /// Common values: `"idle"`, `"processing"`.
    public var onAgentStatus: ((String) -> Void)?

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
        agentStatus = nil
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

    // MARK: - Frame injection

    /// Pushes a ``CMSampleBuffer`` from an external camera source into the video stream.
    ///
    /// Use this to stream video from the **Meta wearables SDK** or any other source that
    /// delivers `CMSampleBuffer` frames. A LiveKit video track is created and published
    /// automatically on the first call; subsequent calls deliver frames to the
    /// already-published track.
    ///
    /// On the **simulator**, ``startCamera()`` calls this method internally with synthetic
    /// test frames, so you can develop and test without wearable hardware.
    ///
    /// - Parameter sampleBuffer: A `CMSampleBuffer` containing a `CVPixelBuffer`.
    /// - Throws: ``StreamError/notConnected`` if not connected.
    public func injectVideoFrame(_ sampleBuffer: sending CMSampleBuffer) async throws {
        guard let injectable = backend as? FrameInjectable else { return }
        try await injectable.injectVideoFrame(sampleBuffer)
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
        backend.onDataReceived = { [weak self] topic, data in
            Task { @MainActor [weak self] in
                self?.onDataReceived?(topic, data)
            }
        }
        backend.onAgentStatus = { [weak self] status in
            Task { @MainActor [weak self] in
                guard let self else { return }
                agentStatus = status
                onAgentStatus?(status)
            }
        }
    }
}
