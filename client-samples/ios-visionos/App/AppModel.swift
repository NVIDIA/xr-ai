import Foundation
import Observation
import StreamKit

// MARK: - AppModel

/// Observable state shared across the sample app.
@MainActor
@Observable
final class AppModel {

    // MARK: - ImmersiveSpace (visionOS)

    #if os(visionOS)
    static let immersiveSpaceID = "StreamKitSpace"
    /// Set to `true` by the ImmersiveSpace scene's `.onAppear`.
    var immersiveSpaceIsOpen = false
    #endif

    // MARK: - Connection settings

    var host: String = "192.168.1.100"
    var port: String = "7880"
    var token: String = ""
    var tokenServerURL: String = ""
    var identity: String = "ios-client"

    // MARK: - Media settings

    var audioMode: AudioConfig.MicrophoneMode = .voiceProcessing

    // MARK: - Live state

    var session: StreamSession?
    var connectionState: ConnectionState = .disconnected
    var isCameraActive = false
    var receivedMessages: [ReceivedMessage] = []
    var lastError: String?

    // MARK: - Connect / disconnect

    func connect() async {
        lastError = nil
        receivedMessages.removeAll()

        let portNumber = Int(port) ?? 7880

        // Mirror web client behaviour: if neither token nor tokenServerURL is
        // provided, fall back to http://<host>:8080/token (the server's built-in
        // token endpoint served alongside the web client).
        let effectiveTokenURL: URL?
        if !tokenServerURL.isEmpty {
            effectiveTokenURL = URL(string: tokenServerURL)
        } else if token.isEmpty {
            effectiveTokenURL = URL(string: "http://\(host):8080/token")
        } else {
            effectiveTokenURL = nil
        }

        let lkConfig = LiveKitConfig(
            host: host,
            port: portNumber,
            token: token.isEmpty ? nil : token,
            tokenURL: effectiveTokenURL
        )

        let newSession = StreamSession(.liveKit(lkConfig))

        // Wire callbacks before connecting.
        newSession.onConnectionStateChanged = { [weak self] state in
            self?.connectionState = state
            // Camera is no longer active after disconnect.
            if state == .disconnected { self?.isCameraActive = false }
        }
        newSession.onDataReceived = { [weak self] data in
            let text = String(data: data, encoding: .utf8) ?? "[\(data.count) bytes binary]"
            self?.receivedMessages.insert(ReceivedMessage(text: text), at: 0)
        }

        session = newSession

        let config = SessionConfig(
            audio: AudioConfig(mode: audioMode),
            identity: identity
        )

        do {
            try await newSession.connect(config: config)
        } catch {
            lastError = error.localizedDescription
            session = nil
        }
    }

    func disconnect() async {
        await session?.disconnect()
        session = nil
        connectionState = .disconnected
        isCameraActive = false
    }

    // MARK: - Camera

    func startCamera() async {
        do {
            try await session?.startCamera()
            isCameraActive = true
        } catch {
            lastError = error.localizedDescription
        }
    }

    func stopCamera() async {
        do {
            try await session?.stopCamera()
        } catch {
            lastError = error.localizedDescription
        }
        isCameraActive = false
    }

    // MARK: - Data

    func sendPing() async {
        let payload = Data("ping:\(Date().timeIntervalSince1970)".utf8)
        do {
            try await session?.send(payload)
        } catch {
            lastError = error.localizedDescription
        }
    }

    func sendCustom(text: String) async {
        guard !text.isEmpty, let data = text.data(using: .utf8) else { return }
        do {
            try await session?.send(data)
        } catch {
            lastError = error.localizedDescription
        }
    }
}

// MARK: - ReceivedMessage

struct ReceivedMessage: Identifiable {
    let id = UUID()
    let text: String
    let timestamp = Date()
}
