// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Observation
import StreamKit

#if os(visionOS)
import ARKit
#endif

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

    // MARK: - Connection settings (persisted across launches via UserDefaults)

    var host: String = AppModel.defaults.string(forKey: Keys.host) ?? "192.168.1.100" {
        didSet { AppModel.defaults.set(host, forKey: Keys.host) }
    }
    var port: String = AppModel.defaults.string(forKey: Keys.port) ?? "8080" {
        didSet { AppModel.defaults.set(port, forKey: Keys.port) }
    }
    /// Bearer token bypass — intentionally not persisted.
    var token: String = ""
    var tokenServerURL: String = AppModel.defaults.string(forKey: Keys.tokenServerURL) ?? "" {
        didSet { AppModel.defaults.set(tokenServerURL, forKey: Keys.tokenServerURL) }
    }
    var identity: String = AppModel.defaults.string(forKey: Keys.identity) ?? "ios-client" {
        didSet { AppModel.defaults.set(identity, forKey: Keys.identity) }
    }

    // MARK: - Audio settings (persisted)

    var audioMode: AudioConfig.MicrophoneMode = AppModel.loadAudioMode() {
        didSet { AppModel.defaults.set(AppModel.encode(audioMode), forKey: Keys.audioMode) }
    }

    // MARK: - Camera settings (persisted)

    var cameraPosition: CameraConfig.Position = AppModel.loadCameraPosition() {
        didSet { AppModel.defaults.set(AppModel.encode(cameraPosition), forKey: Keys.cameraPosition) }
    }

    // MARK: - Topic routing

    /// Topics carrying the agent's final text reply (mirrors web client).
    /// Routed to `agentResponse`; never appended to `receivedMessages`.
    static let agentReplyTopics: Set<String> = ["agent.response", "vlm.response"]

    // MARK: - Persistence helpers

    private static let defaults = UserDefaults.standard

    private enum Keys {
        static let host           = "settings.host"
        static let port           = "settings.port"
        static let tokenServerURL = "settings.tokenServerURL"
        static let identity       = "settings.identity"
        static let audioMode      = "settings.audioMode"
        static let cameraPosition = "settings.cameraPosition"
    }

    private static func encode(_ mode: AudioConfig.MicrophoneMode) -> String {
        switch mode {
        case .voiceProcessing:    return "voiceProcessing"
        case .softwareProcessing: return "softwareProcessing"
        case .raw:                return "raw"
        case .disabled:           return "disabled"
        }
    }
    private static func loadAudioMode() -> AudioConfig.MicrophoneMode {
        switch defaults.string(forKey: Keys.audioMode) {
        case "softwareProcessing": return .softwareProcessing
        case "raw":                return .raw
        case "disabled":           return .disabled
        default:                   return .voiceProcessing
        }
    }
    private static func encode(_ pos: CameraConfig.Position) -> String {
        switch pos {
        case .front: return "front"
        case .back:  return "back"
        }
    }
    private static func loadCameraPosition() -> CameraConfig.Position {
        // Default to the back camera; honour an explicitly saved "front".
        defaults.string(forKey: Keys.cameraPosition) == "front" ? .front : .back
    }

    // MARK: - Live state

    var session: StreamSession?
    var connectionState: ConnectionState = .disconnected
    var agentStatus: String?
    /// Latest final-reply text received on `agent.response` or `vlm.response`.
    /// Mirrors the web client's Agent panel; nil shows the "Waiting for agent..." placeholder.
    var agentResponse: String?
    var isAudioActive = false
    var isCameraActive = false
    private var isCameraStarting = false
    var isConnecting = false
    var receivedMessages: [ReceivedMessage] = []
    var lastError: String?

    // MARK: - Connect / disconnect

    func connect() async {
        guard !isConnecting, connectionState == .disconnected else { return }
        isConnecting = true
        defer { isConnecting = false }

        lastError = nil
        receivedMessages.removeAll()

        let portNumber = Int(port) ?? 8080
        let trimmedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedTokenURL = tokenServerURL.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedTokenURL = trimmedTokenURL.isEmpty
            ? URL(string: "https://\(host):\(portNumber)/token")
            : URL(string: trimmedTokenURL)
        let lkConfig = LiveKitConfig(
            host: host,
            port: portNumber,
            token: trimmedToken.isEmpty ? nil : trimmedToken,
            tokenURL: resolvedTokenURL
        )

        let newSession = StreamSession(.liveKit(lkConfig))

        // Capture newSession weakly in every callback so that stale Tasks dispatched
        // to @MainActor (e.g. a ".connecting" update that arrives after the catch block
        // resets state) are silently dropped once `session` no longer points to this object.
        newSession.onConnectionStateChanged = { [weak self, weak newSession] state in
            guard let self, self.session === newSession else { return }
            self.connectionState = state
            if state == .disconnected {
                self.isAudioActive = false
                self.isCameraActive = false
                self.agentStatus = nil
                self.agentResponse = nil
            }
        }
        newSession.onAgentStatus = { [weak self, weak newSession] status in
            guard let self, self.session === newSession else { return }
            self.agentStatus = status
        }
        newSession.onDataReceived = { [weak self, weak newSession] topic, data in
            guard let self, self.session === newSession else { return }

            // Final agent reply text: route to the Agent panel and never list.
            // Topic set mirrors web/App/app.js AGENT_REPLY_TOPICS.
            if AppModel.agentReplyTopics.contains(topic) {
                self.agentResponse = String(data: data, encoding: .utf8) ?? ""
                return
            }

            // Always-on streaming: clientControl signals from the agent are
            // silently dropped and never surfaced in received messages.
            if topic == "clientControl" {
                return
            }

            let body = String(data: data, encoding: .utf8) ?? "[\(data.count) bytes binary]"
            let text = topic.isEmpty ? body : "[\(topic)] \(body)"
            self.receivedMessages.insert(ReceivedMessage(text: text), at: 0)
        }

        session = newSession

        do {
            try await newSession.connect(config: SessionConfig(identity: identity))
        } catch {
            lastError = error.localizedDescription
            // Tear down synchronously — don't rely on the delegate callback firing
            // when the connection never fully established.
            await newSession.disconnect()
            session = nil
            connectionState = .disconnected
        }
    }

    func disconnect() async {
        await session?.disconnect()
        session = nil
        connectionState = .disconnected
        agentStatus = nil
        agentResponse = nil
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
        guard !isCameraStarting, !isCameraActive else { return }
        isCameraStarting = true
        defer { isCameraStarting = false }

        #if os(visionOS)
        // Surface a friendly message when main-camera access is permanently
        // denied. Without this probe the user sees `LiveKitError.deviceAccessDenied`
        // from StreamKit's internal ARCameraCapturer.
        let result = await ARKitSession().requestAuthorization(for: [.cameraAccess])
        guard result[.cameraAccess] == .allowed else {
            lastError = "Main camera access was not granted. Enable it in Settings → Apps → NVIDIA XR-AI Sample."
            return
        }
        #endif

        do {
            try await session?.startCamera(config: CameraConfig(position: cameraPosition))
            isCameraActive = true
        } catch {
            #if DEBUG
            // `error.localizedDescription` strips domain/code/underlying cause.
            let ns = error as NSError
            print("startCamera failed: \(type(of: error)) \(ns.domain) #\(ns.code) — \(error)")
            print("  userInfo: \(ns.userInfo)")
            if let underlying = ns.userInfo[NSUnderlyingErrorKey] as? NSError {
                print("  underlying: \(underlying.domain) #\(underlying.code) — \(underlying.userInfo)")
            }
            #endif
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

    func switchCamera(to position: CameraConfig.Position) async {
        cameraPosition = position
        guard isCameraActive else { return }
        do {
            try await session?.startCamera(config: CameraConfig(position: cameraPosition))
        } catch {
            lastError = error.localizedDescription
            // The backend's startCamera() stops the previous track before
            // publishing the new one, so on a failed publish nothing is
            // streaming — reflect that instead of leaving the UI on "Streaming".
            isCameraActive = false
        }
    }

    // MARK: - Data

    func sendPing() async {
        do {
            try await session?.send(Data("ping".utf8))
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
