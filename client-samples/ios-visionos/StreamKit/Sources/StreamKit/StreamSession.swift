// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * StreamKit — StreamSession
 *
 * The single public entry-point of the SDK.
 * Runs on @MainActor so it is safe to bind directly to SwiftUI.
 */

import CoreImage
import CoreMedia
import Foundation
import ImageIO
import LiveKit

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

    /// Latest once-per-second network telemetry snapshot.
    @Published public private(set) var networkMetrics: NetworkMetrics?

    // MARK: - Callbacks

    /// Called on the main actor when the connection state changes.
    public var onConnectionStateChanged: ((ConnectionState) -> Void)?

    /// Called on the main actor when data is received.
    /// `topic` identifies the logical channel; `data` is the raw payload.
    public var onDataReceived: ((_ topic: String, _ data: Data) -> Void)?

    /// Called on the main actor when the agent publishes a status update.
    /// Common values: `"idle"`, `"processing"`.
    public var onAgentStatus: ((String) -> Void)?

    /// Called on the main actor when a new network telemetry snapshot is ready.
    public var onNetworkMetrics: ((NetworkMetrics) -> Void)?

    /// Opt-in handler invoked when the remote agent asks this client for a still image.
    public var onImageCaptureRequested:
        (@MainActor (ImageCaptureRequest) async throws -> CapturedImage)?

    // MARK: - Private

    private var backend: any StreamingBackend
    private var captureTasks: [String: Task<Void, Never>] = [:]

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
        captureTasks.values.forEach { $0.cancel() }
        captureTasks.removeAll()
        await backend.disconnect()
        agentStatus = nil
        networkMetrics = nil
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

    /// The currently active local camera track, if any. Used by
    /// ``CameraPreviewView`` to render the outgoing video locally; app code
    /// typically does not access this directly.
    ///
    /// Returns `nil` when the active backend is not LiveKit-backed or while
    /// the camera is stopped.
    public var localCameraTrack: LocalVideoTrack? {
        (backend as? LiveKitBackend)?.localCameraTrack
    }

    /// Capture the next frame from the active local camera as a JPEG still.
    public func captureCurrentCameraImage() async throws -> CapturedImage {
        guard let track = localCameraTrack else {
            throw StreamError.imageCaptureUnavailable("Camera is not active.")
        }
        return try await StillImageCapture.capture(track: track)
    }

    /// Capture one local still without publishing video when the camera is off.
    public func captureImage(config: CameraConfig = .default) async throws -> CapturedImage {
        try await backend.captureImage(config: config)
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

    /// Sends binary data to all participants, optionally on a named topic.
    ///
    /// Topics are how the web client and the server-side worker route
    /// out-of-band signals (e.g. `xr.session.started`). Pass `nil` for the
    /// transport's default topic.
    ///
    /// - Parameters:
    ///   - data: Payload. Keep individual messages ≤ 15 KB on most transports.
    ///   - reliable: Ordered + guaranteed delivery when `true` (default).
    ///   - topic: Optional topic name. Backends that don't support topics
    ///     ignore this.
    public func send(_ data: Data, reliable: Bool = true, topic: String? = nil) async throws {
        try await backend.send(data, reliable: reliable, topic: topic)
    }

    // MARK: - Private

    private func wireCallbacks() {
        backend.onConnectionStateChanged = { [weak self] state in
            Task { @MainActor [weak self] in
                guard let self else { return }
                connectionState = state
                if state != .connected {
                    networkMetrics = nil
                }
                onConnectionStateChanged?(state)
            }
        }
        backend.onDataReceived = { [weak self] topic, data in
            Task { @MainActor [weak self] in
                guard let self else { return }
                switch topic {
                case "camera.capture.request": handleCaptureRequest(data)
                case "camera.capture.cancel": handleCaptureCancel(data)
                default: onDataReceived?(topic, data)
                }
            }
        }
        backend.onAgentStatus = { [weak self] status in
            Task { @MainActor [weak self] in
                guard let self else { return }
                agentStatus = status
                onAgentStatus?(status)
            }
        }
        backend.onNetworkMetrics = { [weak self] metrics in
            Task { @MainActor [weak self] in
                guard let self else { return }
                guard connectionState == .connected else { return }
                networkMetrics = metrics
                onNetworkMetrics?(metrics)
            }
        }
    }

    private func handleCaptureRequest(_ data: Data) {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              object["version"] as? Int == 1,
              let requestID = object["request_id"] as? String,
              !requestID.isEmpty,
              let handler = onImageCaptureRequested else { return }
        captureTasks[requestID]?.cancel()
        captureTasks[requestID] = Task { @MainActor [weak self] in
            defer { self?.captureTasks.removeValue(forKey: requestID) }
            do {
                let image = try await handler(ImageCaptureRequest(
                    requestID: requestID,
                    timeoutMilliseconds: object["timeout_ms"] as? Int ?? 0
                ))
                try Task.checkCancellation()
                guard !image.data.isEmpty,
                      ["image/jpeg", "image/png", "image/webp"].contains(image.mimeType) else {
                    return
                }
                try await self?.backend.sendImage(
                    image.data,
                    requestID: requestID,
                    mimeType: image.mimeType,
                    name: image.name
                )
            } catch is CancellationError {
                return
            } catch {
                return
            }
        }
    }

    private func handleCaptureCancel(_ data: Data) {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let requestID = object["request_id"] as? String else { return }
        captureTasks.removeValue(forKey: requestID)?.cancel()
    }
}

final class StillImageCapture: NSObject, VideoRenderer, @unchecked Sendable {
    @MainActor var isAdaptiveStreamEnabled: Bool { false }
    @MainActor var adaptiveStreamSize: CGSize { .zero }

    private let track: LocalVideoTrack
    private let continuation: CheckedContinuation<CapturedImage, Error>
    private let lock = NSLock()
    private var completed = false
    private var retainedSelf: StillImageCapture?

    private init(
        track: LocalVideoTrack,
        continuation: CheckedContinuation<CapturedImage, Error>
    ) {
        self.track = track
        self.continuation = continuation
    }

    static func capture(track: LocalVideoTrack) async throws -> CapturedImage {
        let holder = StillImageCaptureHolder()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let renderer = StillImageCapture(track: track, continuation: continuation)
                renderer.retainedSelf = renderer
                track.add(videoRenderer: renderer)
                holder.install(renderer)
            }
        } onCancel: {
            holder.cancel()
        }
    }

    nonisolated func render(frame: VideoFrame) {
        guard claimCompletion() else { return }
        track.remove(videoRenderer: self)
        guard let pixelBuffer = frame.toCVPixelBuffer(),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            continuation.resume(throwing: StreamError.imageCaptureUnavailable("JPEG encoding failed."))
            return
        }
        let orientation: CGImagePropertyOrientation = switch frame.rotation {
        case ._0: .up
        case ._90: .right
        case ._180: .down
        case ._270: .left
        }
        guard let jpeg = CIContext().jpegRepresentation(
            of: CIImage(cvPixelBuffer: pixelBuffer).oriented(orientation),
            colorSpace: colorSpace,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.9]
        ) else {
            continuation.resume(throwing: StreamError.imageCaptureUnavailable("JPEG encoding failed."))
            return
        }
        continuation.resume(returning: CapturedImage(data: jpeg))
    }

    nonisolated func cancel() {
        guard claimCompletion() else { return }
        track.remove(videoRenderer: self)
        continuation.resume(throwing: CancellationError())
    }

    nonisolated private func claimCompletion() -> Bool {
        lock.lock()
        guard !completed else { lock.unlock(); return false }
        completed = true
        retainedSelf = nil
        lock.unlock()
        return true
    }
}

private final class StillImageCaptureHolder: @unchecked Sendable {
    private let lock = NSLock()
    private var renderer: StillImageCapture?
    private var cancelled = false

    func install(_ renderer: StillImageCapture) {
        lock.lock()
        if cancelled {
            lock.unlock()
            renderer.cancel()
            return
        }
        self.renderer = renderer
        lock.unlock()
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let renderer = renderer
        self.renderer = nil
        lock.unlock()
        renderer?.cancel()
    }
}
