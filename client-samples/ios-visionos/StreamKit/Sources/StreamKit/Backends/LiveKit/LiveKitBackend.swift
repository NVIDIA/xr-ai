/*
 * StreamKit — LiveKitBackend
 *
 * Implements StreamingBackend using the LiveKit WebRTC SDK.
 * This file is the only place in StreamKit that imports LiveKit directly.
 */

import AVFoundation
import Foundation
import LiveKit

// MARK: - LiveKitBackend

/// ``StreamingBackend`` implementation that uses LiveKit WebRTC for transport.
///
/// Not intended to be used directly — create it via
/// `StreamSession(.liveKit(LiveKitConfig(…)))`.
public final class LiveKitBackend: NSObject, StreamingBackend, @unchecked Sendable {

    // MARK: StreamingBackend hooks

    public var onConnectionStateChanged: (@Sendable (ConnectionState) -> Void)?
    public var onDataReceived: (@Sendable (Data) -> Void)?

    // MARK: Private state

    private let config: LiveKitConfig
    private var room: Room?
    private var sessionConfig: SessionConfig = .default
    private var videoPublication: LocalTrackPublication?

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

        let scheme = config.secure ? "wss" : "ws"
        let url = "\(scheme)://\(config.host):\(config.port)"

        let token: String
        if let t = config.token, !t.isEmpty {
            token = t
        } else if let tokenURL = config.tokenURL {
            token = try await Self.fetchToken(from: tokenURL, identity: sessionConfig.identity)
        } else {
            throw StreamError.missingToken
        }

        let room = Room()
        self.room = room
        room.delegates.add(delegate: self)

        let roomOptions = RoomOptions(stopLocalTrackOnUnpublish: true)
        try await room.connect(url: url, token: token, roomOptions: roomOptions)
    }

    public func disconnect() async {
        await tearDown()
    }

    // MARK: - StreamingBackend: audio

    public func startAudio(config: AudioConfig) async throws {
        guard let room, room.connectionState == .connected else {
            throw StreamError.notConnected
        }
        let captureOptions = AudioCaptureOptions(from: config)
        try await room.localParticipant.setMicrophone(
            enabled: true,
            captureOptions: captureOptions
        )
    }

    public func stopAudio() async throws {
        guard let room else { return }
        try await room.localParticipant.setMicrophone(enabled: false)
    }

    // MARK: - StreamingBackend: camera

    public func startCamera(config: CameraConfig) async throws {
        guard let room, room.connectionState == .connected else {
            throw StreamError.notConnected
        }
        if let existing = videoPublication {
            try await room.localParticipant.unpublish(publication: existing)
            videoPublication = nil
        }

        #if os(visionOS)
        let track = makeVisionOSTrack()
        #else
        let track = makeIOSTrack(config: config)
        #endif

        videoPublication = try await room.localParticipant.publish(videoTrack: track)
    }

    public func stopCamera() async throws {
        guard let pub = videoPublication else { return }
        try await room?.localParticipant.unpublish(publication: pub)
        videoPublication = nil
    }

    // MARK: - StreamingBackend: data

    public func send(_ data: Data, reliable: Bool) async throws {
        guard let room, room.connectionState == .connected else {
            throw StreamError.notConnected
        }
        let options = DataPublishOptions(reliable: reliable)
        try await room.localParticipant.publish(data: data, options: options)
    }

    // MARK: - Private helpers

    private func tearDown() async {
        if let room {
            room.delegates.remove(delegate: self)
            await room.disconnect()
        }
        room = nil
        videoPublication = nil
    }

    // MARK: - Track factories

    #if os(visionOS)
    private func makeVisionOSTrack() -> LocalVideoTrack {
        // ARCameraFrameProvider streams at the hardware's native resolution;
        // dimension/fps hints are ignored by the system.
        return LocalVideoTrack.createARCameraTrack(options: ARCameraCaptureOptions())
    }
    #else
    private func makeIOSTrack(config: CameraConfig) -> LocalVideoTrack {
        let options = CameraCaptureOptions(position: config.avCapturePosition)
        return LocalVideoTrack.createCameraTrack(options: options)
    }
    #endif

    // MARK: - Token fetch

    private static func fetchToken(from base: URL, identity: String) async throws -> String {
        var components = URLComponents(url: base, resolvingAgainstBaseURL: true)
        var items = components?.queryItems ?? []
        items.append(URLQueryItem(name: "identity", value: identity))
        components?.queryItems = items

        guard let url = components?.url else { throw StreamError.tokenFetchFailed(base) }

        let (data, response) = try await URLSession.shared.data(from: url)
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
        participant _: RemoteParticipant?,
        didReceiveData data: Data,
        forTopic _: String,
        encryptionType _: EncryptionType
    ) {
        onDataReceived?(data)
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
        switch config.mode {
        case .voiceProcessing:
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
