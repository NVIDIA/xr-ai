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

    // MARK: - Audio settings

    var audioMode: AudioConfig.MicrophoneMode = .voiceProcessing

    // MARK: - Live state

    var session: StreamSession?
    var connectionState: ConnectionState = .disconnected
    var isAudioActive = false
    var isCameraActive = false
    var receivedMessages: [ReceivedMessage] = []
    var lastError: String?

    // MARK: - Connect / disconnect

    func connect() async {
        lastError = nil
        receivedMessages.removeAll()

        let portNumber = Int(port) ?? 7880
        let trimmedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedTokenURL = tokenServerURL.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedTokenURL = trimmedTokenURL.isEmpty
            ? URL(string: "http://\(host):8080/token")
            : URL(string: trimmedTokenURL)
        let lkConfig = LiveKitConfig(
            host: host,
            port: portNumber,
            token: trimmedToken.isEmpty ? nil : trimmedToken,
            tokenURL: resolvedTokenURL
        )

        let newSession = StreamSession(.liveKit(lkConfig))

        newSession.onConnectionStateChanged = { [weak self] state in
            self?.connectionState = state
            if state == .disconnected {
                self?.isAudioActive = false
                self?.isCameraActive = false
            }
        }
        newSession.onDataReceived = { [weak self] data in
            let text = String(data: data, encoding: .utf8) ?? "[\(data.count) bytes binary]"
            self?.receivedMessages.insert(ReceivedMessage(text: text), at: 0)
        }

        session = newSession

        do {
            try await newSession.connect(config: SessionConfig(identity: identity))
        } catch {
            lastError = error.localizedDescription
            await newSession.disconnect()
            session = nil
        }
    }

    func disconnect() async {
        await session?.disconnect()
        session = nil
        connectionState = .disconnected
        isAudioActive = false
        isCameraActive = false
    }

    // MARK: - Audio

    func startAudio() async {
        do {
            try await session?.startAudio(config: AudioConfig(mode: audioMode))
            isAudioActive = true
        } catch {
            lastError = error.localizedDescription
        }
    }

    func stopAudio() async {
        do {
            try await session?.stopAudio()
        } catch {
            lastError = error.localizedDescription
        }
        isAudioActive = false
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
