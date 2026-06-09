// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * StreamKit — LiveKitBackend
 *
 * Implements StreamingBackend using the LiveKit WebRTC SDK.
 * This file is the only place in StreamKit that imports LiveKit directly.
 */

import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation
import LiveKit

// MARK: - LiveKitBackend

/// ``StreamingBackend`` implementation that uses LiveKit WebRTC for transport.
///
/// Not intended to be used directly — create it via
/// `StreamSession(.liveKit(LiveKitConfig(…)))`.
public final class LiveKitBackend: NSObject, StreamingBackend, FrameInjectable, @unchecked Sendable {

    // MARK: StreamingBackend hooks

    public var onConnectionStateChanged: (@Sendable (ConnectionState) -> Void)?
    public var onDataReceived: (@Sendable (_ topic: String, _ data: Data) -> Void)?
    public var onAgentStatus: (@Sendable (String) -> Void)?

    // MARK: Private constants

    /// Reserved LiveKit topic for internal agent status messages.
    /// Matches the web client's `LiveKitBackend.#STATUS_TOPIC`.
    private static let agentStatusTopic = "_agent.status"

    // MARK: Private state

    private let config: LiveKitConfig
    private var room: Room?
    private var sessionConfig: SessionConfig = .default

    /// Publication for the device camera track (iOS) or ARKit track (visionOS).
    /// Nil on simulator — all video goes through the buffer track path.
    private var cameraPublication: LocalTrackPublication?

    /// Buffer-capturer-backed track for frame injection.
    /// Created lazily on the first injectVideoFrame(_:) call (or by startCamera on simulator).
    private var bufferTrack: LocalVideoTrack?

    /// Publication for the buffer-capturer track. Published after the first injected frame.
    private var bufferPublication: LocalTrackPublication?

    /// Currently active local camera track (device camera, ARKit, or simulator
    /// buffer capturer). Used by ``CameraPreviewView`` to render the outgoing
    /// stream locally. Nil while the camera is stopped.
    public internal(set) var localCameraTrack: LocalVideoTrack?

    #if targetEnvironment(simulator)
    /// Background task that feeds synthetic test frames into the buffer track.
    /// Runs while the simulated camera is "active".
    private var simulatorFrameTask: Task<Void, Never>?
    #endif

    // MARK: - Init

    init(config: LiveKitConfig) {
        self.config = config
    }

    // MARK: - StreamingBackend: connect / disconnect

    /// Establishes the WebRTC peer connection and data channel only.
    /// Audio and camera are not started — call ``startAudio(config:)`` and
    /// ``startCamera(config:)`` explicitly after connecting.
    public func connect(config sessionConfig: SessionConfig) async throws {
        self.sessionConfig = sessionConfig

        await tearDown()

        guard !config.host.isEmpty else {
            throw StreamError.invalidHost(config.host)
        }

        let url = "wss://\(config.host):\(config.port)"

        let token: String
        if let t = config.token, !t.isEmpty {
            token = t
        } else if let tokenURL = config.tokenURL {
            token = try await Self.fetchToken(from: tokenURL, identity: sessionConfig.identity)
        } else {
            throw StreamError.missingToken
        }

        #if targetEnvironment(simulator)
        // Disable Voice-Processing I/O (AUVoiceIO) before the room connects.
        // AUVoiceIO doesn't exist in the simulator and causes error 4010 when the
        // mic is started. We must call this *before* the audio engine starts, not
        // immediately before startMicrophone — the required engine restart needs
        // time to complete, and the connection handshake provides that gap.
        try? AudioManager.shared.setVoiceProcessingEnabled(false)
        #endif

        let room = Room()
        self.room = room
        room.delegates.add(delegate: self)

        let roomOptions = RoomOptions(stopLocalTrackOnUnpublish: true)
        // Audio isolation: when hubIdentity is set, disable auto-subscribe and
        // subscribe only to the hub participant's tracks (post-connect below +
        // the didPublishTrack delegate), so a client never receives another
        // participant's microphone.
        let connectOptions = ConnectOptions(
            autoSubscribe: config.hubIdentity == nil,
            socketConnectTimeoutInterval: 5,   // fail fast if the host is unreachable
            primaryTransportConnectTimeout: 5  // fail fast if ICE/DTLS stalls
        )
        try await room.connect(url: url, token: token, connectOptions: connectOptions, roomOptions: roomOptions)

        // Subscribe to any hub tracks already published before we connected.
        if let hub = config.hubIdentity {
            let hubID = Participant.Identity(from: hub)
            for participant in room.remoteParticipants.values where participant.identity == hubID {
                for publication in participant.trackPublications.values {
                    if let remotePub = publication as? RemoteTrackPublication {
                        try? await remotePub.set(subscribed: true)
                    }
                }
            }
        }
    }

    public func disconnect() async {
        await tearDown()
    }

    // MARK: - StreamingBackend: audio

    public func startAudio(config: AudioConfig) async throws {
        guard let room, room.connectionState == .connected else {
            throw StreamError.notConnected
        }

        // Pre-warm the recording engine before publishing, or the publish's
        // frame watcher never sees a buffer and times out. Reset availability
        // first because stopAudio pins input down, and prepared mode can't
        // start a disabled engine.
        do {
            try AudioManager.shared.setEngineAvailability(.default)
        } catch {
            #if DEBUG
            print("startAudio: setEngineAvailability(.default) failed: \(error)")
            #endif
        }
        do {
            try await AudioManager.shared.setRecordingAlwaysPreparedMode(true)
        } catch {
            #if DEBUG
            print("startAudio: setRecordingAlwaysPreparedMode(true) failed: \(error)")
            #endif
        }

        do {
            let captureOptions = AudioCaptureOptions(from: config)
            try await room.localParticipant.setMicrophone(
                enabled: true,
                captureOptions: captureOptions
            )
        } catch {
            // A failed publish must not leave the engine hot, or the mic
            // indicator stays lit with no way to clear it but disconnecting.
            await releaseRecordingEngine()
            throw error
        }
    }

    public func stopAudio() async throws {
        guard let room else { return }
        let micError: Error?
        do {
            try await room.localParticipant.setMicrophone(enabled: false)
            micError = nil
        } catch {
            micError = error
        }
        await releaseRecordingEngine()
        if let micError { throw micError }
    }

    // MARK: - StreamingBackend: camera

    public func startCamera(config: CameraConfig) async throws {
        guard let room, room.connectionState == .connected else {
            throw StreamError.notConnected
        }

        // Stop any currently active camera before starting a new one.
        try await stopCamera()

        #if targetEnvironment(simulator)
        // No physical camera in any simulator (iOS or visionOS). Set up the buffer track, capture one synthetic
        // frame to resolve stream dimensions, publish synchronously, then start the frame loop.
        // Capturing before publish is a LiveKit requirement — publish times out otherwise.
        // The frame loop feeds frames directly to the capturer (not through injectVideoFrame)
        // so publish is only ever called once, here, not concurrently from the loop.
        let simTrack = LocalVideoTrack.createBufferTrack(name: "meta-camera", source: .camera)
        bufferTrack = simTrack
        // Seed one frame before publish so LiveKit can resolve stream dimensions.
        // Prefer the first GIF frame so the seed matches the loop content.
        let seedBuffer: CMSampleBuffer? = {
            let pts = CMClockGetTime(CMClockGetHostTimeClock())
            if let frames = Self.loadGIFFrames() {
                return Self.sampleBuffer(from: frames[0].image, pts: pts)
            }
            return Self.makeSyntheticSampleBuffer(frameIndex: 0)
        }()
        if let seed = seedBuffer, let capturer = simTrack.capturer as? BufferCapturer {
            capturer.capture(seed)
        }
        bufferPublication = try await room.localParticipant.publish(videoTrack: simTrack)
        localCameraTrack = simTrack
        simulatorFrameTask = Task { [weak self] in
            await self?.runSimulatorFrameLoop(startingAt: 1)
        }

        #elseif os(visionOS)
        // ARKit passthrough camera — device only, requires an open ImmersiveSpace and
        // the com.apple.developer.arkit.main-camera-access.allow enterprise entitlement.
        let track = makeVisionOSTrack()
        cameraPublication = try await room.localParticipant.publish(videoTrack: track)
        localCameraTrack = track

        #else
        // Physical iOS/iPadOS camera.
        let track = makeIOSTrack(config: config)
        cameraPublication = try await room.localParticipant.publish(videoTrack: track)
        localCameraTrack = track
        #endif
    }

    public func stopCamera() async throws {
        #if targetEnvironment(simulator)
        // Stop the synthetic frame generator first so no more frames are injected.
        simulatorFrameTask?.cancel()
        simulatorFrameTask = nil
        #endif

        // Unpublish device / ARKit camera track.
        if let pub = cameraPublication {
            try await room?.localParticipant.unpublish(publication: pub)
            cameraPublication = nil
        }

        // Unpublish and teardown the buffer-capturer track.
        if let pub = bufferPublication {
            try await room?.localParticipant.unpublish(publication: pub)
            bufferPublication = nil
        }
        bufferTrack = nil
        localCameraTrack = nil
    }

    // MARK: - FrameInjectable

    /// Push a ``CMSampleBuffer`` from an external camera source into the LiveKit video stream.
    ///
    /// Frames from any external camera — such as the **Meta wearables SDK** — can be
    /// streamed to remote participants by calling this method for every buffer the SDK
    /// delivers. A `BufferCapturer`-backed LiveKit track is created on the first call
    /// and published to the room automatically once dimensions are known.
    ///
    /// On the **simulator** this is called internally by `startCamera()` with synthetic
    /// test frames, so you can exercise the full pipeline without hardware.
    ///
    /// - Parameter sampleBuffer: A `CMSampleBuffer` containing a `CVPixelBuffer`.
    ///   The pixel format must be one supported by LiveKit
    ///   (`kCVPixelFormatType_420YpCbCr8BiPlanarFullRange`, `kCVPixelFormatType_32BGRA`, etc.).
    /// - Throws: ``StreamError/notConnected`` if not connected.
    public func injectVideoFrame(_ sampleBuffer: sending CMSampleBuffer) async throws {
        guard let room, room.connectionState == .connected else {
            throw StreamError.notConnected
        }

        // Create the buffer track lazily — allows calling injectVideoFrame without
        // calling startCamera first (useful for the Meta wearables use case on device).
        if bufferTrack == nil {
            let t = LocalVideoTrack.createBufferTrack(name: "meta-camera", source: .camera)
            bufferTrack = t
            localCameraTrack = t
        }

        guard let track = bufferTrack,
              let capturer = track.capturer as? BufferCapturer else { return }

        // Deliver the frame (LiveKit resolves stream dimensions on the first call).
        capturer.capture(sampleBuffer)

        // Publish the track lazily — dimensions are now known from the frame above.
        if bufferPublication == nil {
            bufferPublication = try await room.localParticipant.publish(videoTrack: track)
        }
    }

    // MARK: - StreamingBackend: data

    public func send(_ data: Data, reliable: Bool) async throws {
        guard let room, room.connectionState == .connected else {
            throw StreamError.notConnected
        }
        // Address outbound data to the hub participant only so it is not
        // broadcast to — and surfaced by — other participants sharing the room.
        // Clients only ever talk to the hub. nil hubIdentity → whole-room broadcast.
        let destinations = config.hubIdentity.map { [Participant.Identity(from: $0)] } ?? []
        let options = DataPublishOptions(destinationIdentities: destinations, reliable: reliable)
        try await room.localParticipant.publish(data: data, options: options)
    }

    // MARK: - Private helpers

    /// Roll the recording engine back to its idle state: drop prepared mode and
    /// take the mic input side down so the OS mic indicator clears, leaving output
    /// up for agent playback.
    private func releaseRecordingEngine() async {
        try? await AudioManager.shared.setRecordingAlwaysPreparedMode(false)
        try? AudioManager.shared.setEngineAvailability(
            AudioEngineAvailability(isInputAvailable: false, isOutputAvailable: true)
        )
    }

    private func tearDown() async {
        #if targetEnvironment(simulator)
        simulatorFrameTask?.cancel()
        simulatorFrameTask = nil
        #endif

        if let room {
            room.delegates.remove(delegate: self)
            await room.disconnect()
        }
        room = nil
        cameraPublication = nil
        bufferPublication = nil
        bufferTrack = nil
        localCameraTrack = nil
    }

    // MARK: - Track factories

    #if os(visionOS) && !targetEnvironment(simulator)
    private func makeVisionOSTrack() -> LocalVideoTrack {
        // ARCameraFrameProvider streams at the hardware's native resolution;
        // dimension/fps hints are ignored by the system.
        // Requires com.apple.developer.arkit.main-camera-access.allow entitlement.
        return LocalVideoTrack.createARCameraTrack(options: ARCameraCaptureOptions())
    }
    #elseif !os(visionOS) && !targetEnvironment(simulator)
    private func makeIOSTrack(config: CameraConfig) -> LocalVideoTrack {
        let options = CameraCaptureOptions(position: config.avCapturePosition)
        return LocalVideoTrack.createCameraTrack(options: options)
    }
    #endif

    // MARK: - Simulator frame loop

    #if targetEnvironment(simulator)

    // ── GIF frame type ───────────────────────────────────────────────────────────

    private struct GIFFrame {
        let image: CGImage
        /// Display duration in nanoseconds.
        let durationNanos: UInt64
    }

    // ── GIF loader ───────────────────────────────────────────────────────────────

    /// Loads ``SimulatorFeed.gif`` from the StreamKit bundle.
    ///
    /// To use a custom feed, replace `StreamKit/Sources/StreamKit/Resources/SimulatorFeed.gif`
    /// with any animated GIF before building. The file is declared as a package resource so
    /// Swift Package Manager copies it into `Bundle.module` automatically.
    ///
    /// Returns `nil` if the file is missing or unreadable, in which case the frame loop
    /// falls back to a synthetic colour-cycling pattern.
    private static func loadGIFFrames() -> [GIFFrame]? {
        guard let url  = Bundle.module.url(forResource: "SimulatorFeed", withExtension: "gif"),
              let data = try? Data(contentsOf: url),
              let src  = CGImageSourceCreateWithData(data as CFData, nil)
        else { return nil }

        let count = CGImageSourceGetCount(src)
        guard count > 0 else { return nil }

        var frames: [GIFFrame] = []
        for i in 0 ..< count {
            guard let img = CGImageSourceCreateImageAtIndex(src, i, nil) else { continue }

            // Prefer unclampedDelayTime (more accurate for variable-rate GIFs).
            let props    = CGImageSourceCopyPropertiesAtIndex(src, i, nil) as? [CFString: Any]
            let gifProps = props?[kCGImagePropertyGIFDictionary] as? [CFString: Any]
            let secs = (gifProps?[kCGImagePropertyGIFUnclampedDelayTime] as? Double)
                    ?? (gifProps?[kCGImagePropertyGIFDelayTime]          as? Double)
                    ?? (1.0 / 12.0)

            frames.append(GIFFrame(image: img, durationNanos: UInt64(max(secs, 0.01) * 1_000_000_000)))
        }
        return frames.isEmpty ? nil : frames
    }

    // ── CGImage → CMSampleBuffer conversion ─────────────────────────────────────

    private static func sampleBuffer(from image: CGImage, pts: CMTime) -> CMSampleBuffer? {
        let w = image.width, h = image.height

        // Allocate a BGRA pixel buffer (IOSurface-backed so LiveKit can consume it).
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                                  kCVPixelFormatType_32BGRA, attrs, &pb) == kCVReturnSuccess,
              let pixelBuffer = pb else { return nil }

        // Draw the CGImage into the pixel buffer using CoreGraphics.
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        let ctx = CGContext(
            data:            CVPixelBufferGetBaseAddress(pixelBuffer),
            width:           w,
            height:          h,
            bitsPerComponent: 8,
            bytesPerRow:     CVPixelBufferGetBytesPerRow(pixelBuffer),
            space:           CGColorSpaceCreateDeviceRGB(),
            bitmapInfo:      CGBitmapInfo.byteOrder32Little.rawValue
                           | CGImageAlphaInfo.premultipliedFirst.rawValue
        )
        ctx?.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])

        var fd: CMFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
                                                    imageBuffer: pixelBuffer,
                                                    formatDescriptionOut: &fd)
        guard let formatDesc = fd else { return nil }

        var timing = CMSampleTimingInfo(duration: .invalid,
                                        presentationTimeStamp: pts,
                                        decodeTimeStamp: .invalid)
        var sb: CMSampleBuffer?
        CMSampleBufferCreateForImageBuffer(allocator: kCFAllocatorDefault,
                                           imageBuffer: pixelBuffer,
                                           dataReady: true,
                                           makeDataReadyCallback: nil,
                                           refcon: nil,
                                           formatDescription: formatDesc,
                                           sampleTiming: &timing,
                                           sampleBufferOut: &sb)
        return sb
    }

    // ── Frame loop ───────────────────────────────────────────────────────────────

    /// Feeds the simulator's virtual camera with frames from ``SimulatorFeed.gif``,
    /// falling back to a synthetic colour-cycling pattern when the GIF is unavailable.
    ///
    /// - Parameter startingAt: Frame index to begin from. Pass `1` when frame 0 was
    ///   already captured to seed the track before publish.
    private func runSimulatorFrameLoop(startingAt startIndex: Int = 0) async {
        if let gifFrames = Self.loadGIFFrames() {
            // ── GIF playback ─────────────────────────────────────────────────────
            var idx = startIndex % gifFrames.count
            var pts = CMClockGetTime(CMClockGetHostTimeClock())

            while !Task.isCancelled {
                let frame = gifFrames[idx]
                if let buf = Self.sampleBuffer(from: frame.image, pts: pts) {
                    try? await injectVideoFrame(buf)
                }
                idx  = (idx + 1) % gifFrames.count
                pts  = CMTimeAdd(pts, CMTimeMake(value: Int64(frame.durationNanos), timescale: 1_000_000_000))
                try? await Task.sleep(nanoseconds: frame.durationNanos)
            }
        } else {
            // ── Synthetic colour-cycle fallback ──────────────────────────────────
            var frameIndex = startIndex
            while !Task.isCancelled {
                if let buf = Self.makeSyntheticSampleBuffer(frameIndex: frameIndex) {
                    try? await injectVideoFrame(buf)
                }
                frameIndex += 1
                try? await Task.sleep(nanoseconds: 33_333_333) // ~30 fps
            }
        }
    }

    // ── Synthetic fallback generator ─────────────────────────────────────────────

    /// Returns a 640×480 BGRA ``CMSampleBuffer`` whose colour cycles slowly over time.
    /// Used only when ``SimulatorFeed.gif`` cannot be loaded from the bundle.
    private static func makeSyntheticSampleBuffer(frameIndex: Int) -> CMSampleBuffer? {
        let width = 640, height = 480
        let t = Double(frameIndex) / 30.0

        let r = UInt8(max(0, min(255, Int(sin(t * 0.3)         * 100 + 155))))
        let g = UInt8(max(0, min(255, Int(sin(t * 0.5 + 2.094) * 100 + 155))))
        let b = UInt8(max(0, min(255, Int(sin(t * 0.7 + 4.189) * 100 + 155))))

        var pixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                  kCVPixelFormatType_32BGRA, attrs,
                                  &pixelBuffer) == kCVReturnSuccess,
              let pb = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(pb, [])
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)
        if let base = CVPixelBufferGetBaseAddress(pb) {
            let buf = base.bindMemory(to: UInt8.self, capacity: height * bytesPerRow)
            for row in 0 ..< height {
                for col in 0 ..< width {
                    let i = row * bytesPerRow + col * 4
                    buf[i] = b; buf[i+1] = g; buf[i+2] = r; buf[i+3] = 255
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, [])

        var formatDesc: CMFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
                                                    imageBuffer: pb,
                                                    formatDescriptionOut: &formatDesc)
        guard let fd = formatDesc else { return nil }

        var timing = CMSampleTimingInfo(duration: CMTimeMake(value: 1, timescale: 30),
                                        presentationTimeStamp: CMClockGetTime(CMClockGetHostTimeClock()),
                                        decodeTimeStamp: .invalid)
        var sampleBuffer: CMSampleBuffer?
        CMSampleBufferCreateForImageBuffer(allocator: kCFAllocatorDefault,
                                           imageBuffer: pb, dataReady: true,
                                           makeDataReadyCallback: nil, refcon: nil,
                                           formatDescription: fd,
                                           sampleTiming: &timing,
                                           sampleBufferOut: &sampleBuffer)
        return sampleBuffer
    }
    #endif

    // MARK: - Token fetch

    private static func fetchToken(from base: URL, identity: String) async throws -> String {
        var components = URLComponents(url: base, resolvingAgainstBaseURL: true)
        var items = components?.queryItems ?? []
        items.append(URLQueryItem(name: "identity", value: identity))
        components?.queryItems = items

        guard let url = components?.url else { throw StreamError.tokenFetchFailed(base) }

        // Use a session that accepts self-signed / untrusted certificates.
        // StreamKit is a developer tool that connects to operator-controlled LAN servers;
        // NSAllowsArbitraryLoads in Info.plist covers the LiveKit WSS connection, while
        // this delegate covers the token endpoint (belt-and-suspenders).
        let session = URLSession(
            configuration: .ephemeral,
            delegate: TrustingSessionDelegate(),
            delegateQueue: nil
        )
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse,
              (200 ..< 300).contains(http.statusCode)
        else { throw StreamError.tokenFetchFailed(url) }

        struct Envelope: Decodable { let token: String }
        if let e = try? JSONDecoder().decode(Envelope.self, from: data) { return e.token }
        if let plain = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines), !plain.isEmpty { return plain }
        throw StreamError.tokenFetchFailed(url)
    }
}

// MARK: - RoomDelegate

extension LiveKitBackend: RoomDelegate {

    public func room(
        _ room: Room,
        didUpdateConnectionState connectionState: LiveKit.ConnectionState,
        from _: LiveKit.ConnectionState
    ) {
        onConnectionStateChanged?(connectionState.toStreamKitState())
    }

    public func room(
        _ room: Room,
        participant: RemoteParticipant?,
        didReceiveData data: Data,
        forTopic topic: String,
        encryptionType _: EncryptionType
    ) {
        // Intercept the reserved agent-status topic — never forward to onDataReceived.
        if topic == Self.agentStatusTopic {
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let status = json["status"] as? String, !status.isEmpty {
                onAgentStatus?(status)
            }
            return
        }
        // Isolation: only surface data from the hub/agent, never from peer
        // participants. nil hubIdentity keeps the legacy accept-from-anyone behaviour.
        if let hub = config.hubIdentity, participant?.identity != Participant.Identity(from: hub) {
            return
        }
        onDataReceived?(topic, data)
    }

    public func room(
        _ room: Room,
        participant: RemoteParticipant,
        didPublishTrack publication: RemoteTrackPublication
    ) {
        // Audio isolation: subscribe only to the hub participant's tracks.
        guard let hub = config.hubIdentity,
              participant.identity == Participant.Identity(from: hub) else { return }
        Task { try? await publication.set(subscribed: true) }
    }
}

// MARK: - Helpers

private extension LiveKit.ConnectionState {
    func toStreamKitState() -> ConnectionState {
        switch self {
        case .disconnected, .disconnecting: return .disconnected
        case .connecting:                   return .connecting
        case .connected:                    return .connected
        case .reconnecting:                 return .reconnecting
        }
    }
}

private extension AudioCaptureOptions {
    convenience init(from config: AudioConfig) {
        // AUVoiceIO (Apple's hardware Voice-Processing I/O unit) is unavailable in the
        // simulator, so we silently promote voiceProcessing → softwareProcessing there.
        #if targetEnvironment(simulator)
        let mode: AudioConfig.MicrophoneMode =
            config.mode == .voiceProcessing ? .softwareProcessing : config.mode
        #else
        let mode = config.mode
        #endif

        switch mode {
        case .voiceProcessing:
            // AUVoiceIO owns echo cancellation / AGC / NR at the OS level;
            // tell LiveKit's WebRTC stack to leave them off.
            self.init(echoCancellation: false, autoGainControl: false, noiseSuppression: false,
                      highpassFilter: config.highpassFilter, typingNoiseDetection: config.typingNoiseDetection)
        case .softwareProcessing:
            self.init(echoCancellation: true,  autoGainControl: true,  noiseSuppression: true,
                      highpassFilter: config.highpassFilter, typingNoiseDetection: config.typingNoiseDetection)
        case .raw, .disabled:
            self.init(echoCancellation: false, autoGainControl: false, noiseSuppression: false,
                      highpassFilter: false, typingNoiseDetection: false)
        }
    }
}

private extension CameraConfig {
    #if !os(visionOS)
    var avCapturePosition: AVCaptureDevice.Position {
        position == .front ? .front : .back
    }
    #endif
}

// MARK: - TrustingSessionDelegate

/// URLSession delegate that accepts any server certificate. Used only for
/// the /token fetch — the LiveKit Swift SDK owns its own URLSession, so
/// the wss handshake to a self-signed cert still requires a trusted
/// profile on the device.
private final class TrustingSessionDelegate: NSObject, URLSessionDelegate, @unchecked Sendable {
    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        if challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
           let trust = challenge.protectionSpace.serverTrust {
            completionHandler(.useCredential, URLCredential(trust: trust))
            return
        }
        completionHandler(.performDefaultHandling, nil)
    }
}
