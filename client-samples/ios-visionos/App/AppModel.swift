// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
    var secure: Bool = false
    var token: String = ""
    var tokenServerURL: String = ""
    var identity: String = "ios-client"

    // MARK: - Audio settings

    var audioMode: AudioConfig.MicrophoneMode = .voiceProcessing

    // MARK: - Camera settings

    var cameraPosition: CameraConfig.Position = .front
    /// When `true`, ``clientControl`` startCamera/stopCamera messages from
    /// the agent are honoured.  When `false` (default — always-on), they are
    /// ignored and the camera button is the sole control.
    var cameraOnDemand: Bool = false

    // MARK: - Live state

    var session: StreamSession?
    var connectionState: ConnectionState = .disconnected
    var agentStatus: String?
    var isAudioActive = false
    var isCameraActive = false
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

        let portNumber = Int(port) ?? 7880
        let trimmedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedTokenURL = tokenServerURL.trimmingCharacters(in: .whitespacesAndNewlines)
        let tokenScheme = secure ? "https" : "http"
        let resolvedTokenURL = trimmedTokenURL.isEmpty
            ? URL(string: "\(tokenScheme)://\(host):8080/token")
            : URL(string: trimmedTokenURL)
        // LiveKit always runs on plain ws:// in this deployment — TLS is terminated
        // at the web-server layer (port 8080), not at the LiveKit signaling port (7880).
        // `secure` only affects the token-endpoint URL scheme (http vs https).
        let lkConfig = LiveKitConfig(
            host: host,
            port: portNumber,
            secure: false,
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
            }
        }
        newSession.onAgentStatus = { [weak self, weak newSession] status in
            guard let self, self.session === newSession else { return }
            self.agentStatus = status
        }
        newSession.onDataReceived = { [weak self, weak newSession] topic, data in
            guard let self, self.session === newSession else { return }

            // Camera on demand: intercept clientControl signals from the agent.
            // In always-on mode (cameraOnDemand = false) they are silently ignored.
            if topic == "clientControl" {
                if self.cameraOnDemand,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let action = json["action"] as? String {
                    if action == "startCamera" && !self.isCameraActive {
                        Task { await self.startCamera() }
                    } else if action == "stopCamera" && self.isCameraActive {
                        Task { await self.stopCamera() }
                    }
                }
                return  // never surface in received messages
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
            try await session?.startCamera(config: CameraConfig(position: cameraPosition))
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

    func switchCamera(to position: CameraConfig.Position) async {
        cameraPosition = position
        guard isCameraActive else { return }
        do {
            try await session?.startCamera(config: CameraConfig(position: cameraPosition))
        } catch {
            lastError = error.localizedDescription
        }
    }

    // MARK: - Data

    func sendPing() async {
        // In on-demand mode, start the camera now so it warms up in parallel
        // with the ping's round-trip and agent processing.
        if cameraOnDemand && !isCameraActive {
            Task { await startCamera() }
        }
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
