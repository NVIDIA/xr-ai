// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import AVFoundation
import Foundation
import Observation

import StreamKit

#if os(visionOS)
import ARKit
import CloudXRKit
#endif

// MARK: - XR state

/// Lifecycle of the CloudXR streaming side of the session.
/// Independent of ``ConnectionState``. They have their own transports.
enum XRState: Equatable {
    case idle
    case connecting
    case streaming
    case stopping
    case error(String)

    var isLive: Bool {
        switch self {
        case .connecting, .streaming: return true
        case .idle, .stopping, .error: return false
        }
    }
}

enum CameraMode: String, CaseIterable {
    case off
    case onDemand
    case live

    var displayName: String {
        switch self {
        case .off: "Off"
        case .onDemand: "On-demand images"
        case .live: "Live video"
        }
    }
}

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
    var cameraMode: CameraMode = AppModel.loadCameraMode() {
        didSet { AppModel.defaults.set(cameraMode.rawValue, forKey: Keys.cameraMode) }
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
        static let cameraMode     = "settings.cameraMode"
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
    private static func loadCameraMode() -> CameraMode {
        defaults.string(forKey: Keys.cameraMode).flatMap(CameraMode.init(rawValue:)) ?? .off
    }

    // MARK: - Live state

    var session: StreamSession?
    var connectionState: ConnectionState = .disconnected
    var agentStatus: String?
    var networkMetrics: NetworkMetrics?
    /// Latest final-reply text received on `agent.response` or `vlm.response`.
    /// Mirrors the web client's Agent panel; nil shows the "Waiting for agent..." placeholder.
    var agentResponse: String?
    var isAudioActive = false
    private var micEnabledByUser: Bool { microphone?.isEnabled == true }
    @ObservationIgnored private var microphone: MicrophoneReconciler?
    private(set) var isAudioStarting = false
    var isCameraActive = false
    private var isCameraStarting = false
    private var imageCaptureHandler:
        (@MainActor (ImageCaptureRequest) async throws -> CapturedImage)?
    private var isTearingDown = false
    var isConnecting = false
    var receivedMessages: [ReceivedMessage] = []
    var lastError: String?

    // MARK: - CloudXR live state (visionOS only)

    #if os(visionOS)
    var cloudxrSession: CloudXRKit.Session?

    var xrState: XRState {
        guard let state = cloudxrSession?.state else { return .idle }
        switch state {
        case .initialized, .paused, .disconnected(.success):
            return .idle
        case .connecting, .authenticating, .authenticated, .resuming:
            return .connecting
        case .connected:
            return .streaming
        case .disconnecting, .pausing:
            return .stopping
        case .disconnected(.failure(let err)):
            return .error(err.localizedDescription)
        @unknown default:
            return .idle
        }
    }
    #endif

    // MARK: - Lifecycle

    init() {
        audioInterruptionToken = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main
        ) { [weak self] note in
            MainActor.assumeIsolated { self?.handleAudioInterruption(note) }
        }
        // A media-services reset destroys the audio engine, leaving LiveKit's mic
        // track bound to nothing, so re-run mic recovery.
        mediaServicesResetToken = NotificationCenter.default.addObserver(
            forName: AVAudioSession.mediaServicesWereResetNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.handleMediaServicesReset() }
        }
    }

    deinit {
        if let audioInterruptionToken {
            NotificationCenter.default.removeObserver(audioInterruptionToken)
        }
        if let mediaServicesResetToken {
            NotificationCenter.default.removeObserver(mediaServicesResetToken)
        }
    }

    // MARK: - Connect / disconnect

    func connect() async {
        guard !isConnecting, !isTearingDown, connectionState == .disconnected else { return }
        isConnecting = true
        defer { isConnecting = false }

        // AudioManager is process-wide: drain the old session before replacing it.
        micRecoveryPending = false
        micRecoveryTask?.cancel()
        await microphone?.close().value
        await micRecoveryTask?.value
        guard !isTearingDown else { return }

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
            self.handleMicrophoneConnectionState(state)
            switch state {
            case .disconnected:
                self.isAudioActive = false
                self.isAudioStarting = false
                self.isCameraActive = false
                self.agentStatus = nil
                self.networkMetrics = nil
                self.agentResponse = nil
                #if os(visionOS)
                // An unexpected LiveKit drop would orphan the CloudXR session; tear
                // XR down so the two transports can't diverge (isTearingDown skips
                // this on the intentional path, where disconnect() already stopped XR).
                if !self.isTearingDown, self.xrState.isLive {
                    Task { await self.stopXR() }
                }
                #endif
            case .reconnecting:
                self.networkMetrics = nil
                if self.isCameraActive {
                    Task { await self.stopCamera() }
                }
            case .connected:
                if self.cameraMode == .live {
                    Task { await self.startCamera() }
                }
                #if os(visionOS)
                // Re-arm the started-signal publish: a LiveKit reconnect that
                // overlapped XR connect would have aborted the publish loop, and
                // CloudXR won't re-emit `.connected` to retrigger it.
                if self.xrState == .streaming {
                    self.publishXRStartedWhenConnected()
                }
                #endif
            case .connecting:
                break
            }
        }
        newSession.onAgentStatus = { [weak self, weak newSession] status in
            guard let self, self.session === newSession else { return }
            self.agentStatus = status
        }
        newSession.onNetworkMetrics = { [weak self, weak newSession] metrics in
            guard let self, self.session === newSession else { return }
            self.networkMetrics = metrics
        }
        imageCaptureHandler = { [weak self, weak newSession] _ in
            guard let self, let newSession, self.session === newSession else {
                throw CancellationError()
            }
            guard self.cameraMode == .onDemand else {
                throw StreamError.imageCaptureUnavailable("On-demand image capture is not enabled.")
            }
            return try await newSession.captureImage(
                config: CameraConfig(position: self.cameraPosition)
            )
        }
        newSession.onImageCaptureRequested = cameraMode == .onDemand
            ? imageCaptureHandler
            : nil
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
        microphone = makeMicrophone(for: newSession)

        do {
            try await newSession.connect(config: SessionConfig(identity: identity))
        } catch {
            lastError = error.localizedDescription
            // Tear down synchronously — don't rely on the delegate callback firing
            // when the connection never fully established.
            await newSession.disconnect()
            await microphone?.close().value
            microphone = nil
            session = nil
            imageCaptureHandler = nil
            connectionState = .disconnected
        }
    }

    private func makeMicrophone(for newSession: StreamSession) -> MicrophoneReconciler {
        MicrophoneReconciler(
            start: { [weak self, weak newSession] in
                guard let self, let newSession, self.session === newSession,
                      !self.isTearingDown, self.connectionState == .connected else {
                    throw StreamError.notConnected
                }
                try await newSession.startAudio(config: AudioConfig(mode: self.audioMode))
            },
            stop: { [weak newSession] in try await newSession?.stopAudio() },
            changed: { [weak self, weak newSession] active, starting in
                guard let self, let newSession, self.session === newSession else { return }
                self.isAudioActive = active
                self.isAudioStarting = starting
            },
            failed: { [weak self, weak newSession] error, operation in
                guard let self, let newSession, self.session === newSession else { return }
                guard !self.isTearingDown else { return }
                if operation == .start, error is CancellationError || self.isRecoveringMic { return }
                self.lastError = operation == .start
                    ? "Microphone couldn’t start. Please try again."
                    : error.localizedDescription
            }
        )
    }

    func disconnect() async {
        isTearingDown = true
        micRecoveryPending = false
        // Cancel any pending post-XR mic restore so it can't re-flag the mic live
        // against the session we're about to nil.
        micRecoveryTask?.cancel()
        await microphone?.close().value
        await micRecoveryTask?.value
        #if os(visionOS)
        // Tear down CloudXR first so the streaming view dismisses before the agent
        // channel closes (stopXR() awaits `.disconnected`, so this ordering holds).
        if xrState != .idle {
            await stopXR()
        }
        #endif
        await session?.disconnect()
        session = nil
        microphone = nil
        imageCaptureHandler = nil
        connectionState = .disconnected
        agentStatus = nil
        networkMetrics = nil
        agentResponse = nil
        isAudioActive = false
        isAudioStarting = false
        isCameraActive = false
        isTearingDown = false
    }

    // MARK: - Audio

    func enableMic() async {
        guard !isTearingDown, connectionState == .connected else { return }
        await microphone?.setEnabled(true)
    }

    func disableMic() async {
        micRecoveryPending = false
        micRecoveryTask?.cancel()
        await microphone?.setEnabled(false)
        await micRecoveryTask?.value
    }

    // MARK: - Camera

    func applyCameraMode(_ mode: CameraMode) async {
        cameraMode = mode
        session?.onImageCaptureRequested = mode == .onDemand
            ? imageCaptureHandler
            : nil
        if mode == .live, connectionState == .connected {
            await startCamera()
        } else if isCameraActive || isCameraStarting {
            await stopCamera()
        }
    }

    func startCamera() async {
        guard cameraMode == .live, !isCameraStarting, !isCameraActive else { return }
        isCameraStarting = true
        defer { isCameraStarting = false }

        #if os(visionOS)
        guard immersiveSpaceIsOpen else {
            lastError = "Open the immersive space before enabling live video."
            fallBackToCameraOff()
            return
        }
        // Surface a friendly message when main-camera access is permanently
        // denied. Without this probe the user sees `LiveKitError.deviceAccessDenied`
        // from StreamKit's internal ARCameraCapturer.
        let result = await ARKitSession().requestAuthorization(for: [.cameraAccess])
        guard result[.cameraAccess] == .allowed else {
            lastError = "Main camera access was not granted. Enable it in Settings → Apps → NVIDIA XR-AI Sample."
            fallBackToCameraOff()
            return
        }
        #endif

        guard cameraMode == .live, connectionState == .connected else { return }

        do {
            try await session?.startCamera(config: CameraConfig(position: cameraPosition))
            guard cameraMode == .live, connectionState == .connected else {
                try? await session?.stopCamera()
                return
            }
            isCameraActive = true
        } catch {
            #if DEBUG
            // `error.localizedDescription` strips domain/code/underlying cause.
            let ns = error as NSError
            print("startCamera failed: \(type(of: error)) \(ns.domain) #\(ns.code): \(error)")
            print("  userInfo: \(ns.userInfo)")
            if let underlying = ns.userInfo[NSUnderlyingErrorKey] as? NSError {
                print("  underlying: \(underlying.domain) #\(underlying.code): \(underlying.userInfo)")
            }
            #endif
            lastError = error.localizedDescription
            fallBackToCameraOff()
        }
    }

    private func fallBackToCameraOff() {
        guard cameraMode == .live, connectionState == .connected else { return }
        cameraMode = .off
        session?.onImageCaptureRequested = nil
    }

    func stopCamera() async {
        do {
            try await session?.stopCamera()
        } catch {
            lastError = error.localizedDescription
        }
        isCameraActive = false
        // A newer Live selection may have arrived while shutdown was
        // suspended. Reconcile the latest desired mode after it completes.
        if cameraMode == .live, connectionState == .connected {
            await startCamera()
        }
    }

    func switchCamera(to position: CameraConfig.Position) async {
        cameraPosition = position
        // Serialize against a concurrent start/switch via the same flag
        // startCamera() uses. switchCamera re-invokes the backend's
        // startCamera() (which tears down the active track before publishing
        // the new one), so an overlapping start must not interleave (#208).
        guard isCameraActive, !isCameraStarting else { return }
        isCameraStarting = true
        defer { isCameraStarting = false }
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

    // MARK: - Mic recovery
    //
    // XR exit posts `.began` with no `.ended`; recovery is a bounded
    // settle→stop→start→verify loop (see README).

    private var micRecoveryTask: Task<Void, Never>?
    private var isRecoveringMic = false
    private var micRecoveryPending = false

    private static let micRecoverySettleNanos: UInt64 = 500_000_000
    private static let micRecoveryVerifyNanos: UInt64 = 500_000_000
    private static let micRecoveryMaxAttempts = 4

    private var interruptionBeganGeneration = 0

    // Observer handles, removed in deinit; nonisolated(unsafe) because deinit reads them off the actor.
    private nonisolated(unsafe) var audioInterruptionToken: NSObjectProtocol?
    private nonisolated(unsafe) var mediaServicesResetToken: NSObjectProtocol?

    private func recoverMic() {
        guard micEnabledByUser, !isTearingDown, let microphone else { return }
        micRecoveryPending = true
        guard connectionState == .connected else { return }
        #if os(visionOS)
        // A failed XR disconnect is terminal too; it need not pass through idle.
        guard !xrState.isLive, xrState != .stopping else { return }
        #endif
        let previous = micRecoveryTask
        previous?.cancel()
        micRecoveryTask = Task {
            await previous?.value
            guard !Task.isCancelled else { return }
            isRecoveringMic = true
            defer { isRecoveringMic = false }
            for _ in 0 ..< Self.micRecoveryMaxAttempts {
                try? await Task.sleep(nanoseconds: Self.micRecoverySettleNanos)
                guard !Task.isCancelled, self.microphone === microphone, micEnabledByUser,
                      connectionState == .connected else { return }
                #if os(visionOS)
                guard !xrState.isLive, xrState != .stopping else { return }
                #endif

                // Snapshot before the attempt: a `.began` after this point means
                // NSK re-suspended the capture we are about to restart.
                let generation = interruptionBeganGeneration
                await microphone.restart().value
                guard !Task.isCancelled, self.microphone === microphone else { return }

                try? await Task.sleep(nanoseconds: Self.micRecoveryVerifyNanos)
                guard !Task.isCancelled, self.microphone === microphone, micEnabledByUser,
                      connectionState == .connected else { return }
                #if os(visionOS)
                guard !xrState.isLive, xrState != .stopping else { return }
                #endif
                if isAudioActive, interruptionBeganGeneration == generation {
                    micRecoveryPending = false
                    return
                }
            }
            guard !Task.isCancelled, self.microphone === microphone,
                  connectionState == .connected else { return }
            micRecoveryPending = false
            await microphone.setEnabled(false)
        }
    }

    private func handleMicrophoneConnectionState(_ state: ConnectionState) {
        switch state {
        case .disconnected:
            micRecoveryPending = false
            micRecoveryTask?.cancel()
            microphone?.close()
        case .reconnecting:
            micRecoveryTask?.cancel()
        case .connected:
            guard micEnabledByUser, !isTearingDown, let microphone else { return }
            #if os(visionOS)
            guard !xrState.isLive, xrState != .stopping else { return }
            #endif
            if micRecoveryPending {
                recoverMic()
            } else {
                // LiveKit owns healthy-track republishing on reconnect. Only
                // restart when an actual recovery request was deferred.
                let previous = micRecoveryTask
                previous?.cancel()
                micRecoveryTask = Task {
                    await previous?.value
                    guard !Task.isCancelled, self.microphone === microphone,
                          micEnabledByUser, !isTearingDown, connectionState == .connected else { return }
                    #if os(visionOS)
                    guard !xrState.isLive, xrState != .stopping else { return }
                    #endif
                    await microphone.setEnabled(true)
                }
            }
        default:
            break
        }
    }

    private func handleAudioInterruption(_ note: Notification) {
        guard let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: raw) else { return }
        switch type {
        case .began:
            interruptionBeganGeneration &+= 1
        case .ended:
            #if os(visionOS)
            guard micEnabledByUser, xrState == .idle else { return }
            #else
            guard micEnabledByUser else { return }
            #endif
            recoverMic()
        @unknown default:
            break
        }
    }

    private func handleMediaServicesReset() {
        #if os(visionOS)
        guard micEnabledByUser, xrState == .idle else { return }
        #else
        guard micEnabledByUser else { return }
        #endif
        recoverMic()
    }

    // MARK: - Data

    func sendCustom(text: String) async {
        guard !text.isEmpty, let data = text.data(using: .utf8) else { return }
        do {
            try await session?.send(data)
        } catch {
            lastError = error.localizedDescription
        }
    }

    // MARK: - CloudXR (visionOS only)

    #if os(visionOS)

    /// Topic the render worker watches to unlock LOVR.
    /// Must match `client-samples/web-xr/App/app.js`.
    private static let xrSessionStartedTopic = "xr.session.started"

    /// Set only after a successful `send`; reset on `.disconnected` so a second XR
    /// session republishes.
    private var hasPublishedXRStarted = false
    /// In-flight guard so overlapping observation ticks can't launch concurrent
    /// publish loops.
    private var isPublishingXRStarted = false
    /// Bounded publish retry: `.connected` won't re-fire while the session stays
    /// connected, so a transient send failure must not wedge the signal off until disconnect.
    private static let xrStartedMaxAttempts = 4
    private static let xrStartedRetryNanos: UInt64 = 500_000_000

    /// Single-flight guard: ``xrState`` stays `.idle` until CloudXRKit flips state,
    /// so a double tap would otherwise re-run `configure`/`connect` on the reused session.
    private var isStartingXR = false

    /// Resumed when CloudXR reaches a terminal `.disconnected` state so
    /// ``stopXR()`` can await teardown completion.
    private var xrDisconnectWaiters: [CheckedContinuation<Void, Never>] = []

    /// Build a CloudXR session and connect. The immersive space must already be
    /// open: the render target lives in `ImmersiveView`'s `RealityView` and must
    /// exist before frames arrive.
    func startXR() async {
        guard !isStartingXR else { return }
        // `.error` must stay retryable; reject only while connect/stream/teardown
        // is in flight.
        switch xrState {
        case .idle, .error: break
        case .connecting, .streaming, .stopping: return
        }
        isStartingXR = true
        defer { isStartingXR = false }

        var cxrConfig = CloudXRKit.Config()
        cxrConfig.connectionType = .local(ip: host)
        // Per-eye stream size; the framework default produces 4096² eyes that OOM the server compositor.
        cxrConfig.resolutionPreset = .standardPreset
        cxrConfig.handTrackingMode = .disabled
        cxrConfig.controllerTrackingMode = .disabled
        cxrConfig.controllerTrackingPredictionFactor = 1.0

        // Create the session once and reuse it across connect/disconnect cycles;
        // `configure` applies the real config per connect.
        if cloudxrSession == nil {
            cloudxrSession = CloudXRSession(config: CloudXRKit.Config())
            // Observe state in the model layer (not a view's `.onChange`) so
            // transitions are caught even on a reused session.
            beginObservingXRState()
        }
        guard let s = cloudxrSession else { return }

        do {
            s.configure(config: cxrConfig)
            try await s.connect()
        } catch {
            lastError = error.localizedDescription
        }
    }

    /// Disconnect CloudXR and await the `.disconnected` transition; the session is
    /// kept for reuse and LiveKit is unaffected. Awaiting lets ``disconnect()`` rely
    /// on the render target being released before it closes the agent channel.
    func stopXR() async {
        switch xrState {
        case .idle:
            return
        case .error:
            // Already disconnected(.failure): reusable, but no further transition
            // will arrive to await.
            cloudxrSession?.disconnect()
            return
        case .stopping:
            break  // teardown already in flight; await its completion
        case .connecting, .streaming:
            cloudxrSession?.disconnect()
        }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            switch xrState {
            case .idle, .error:
                continuation.resume()
            case .connecting, .streaming, .stopping:
                xrDisconnectWaiters.append(continuation)
            }
        }
    }

    private func handleCloudxrStateChange(_ newState: SessionState?) {
        guard let newState else { return }

        if case .disconnected(let result) = newState {
            // Unblock any stopXR() awaiting teardown completion.
            for waiter in xrDisconnectWaiters { waiter.resume() }
            xrDisconnectWaiters.removeAll()

            if case .failure(let err) = result {
                #if DEBUG
                print("CloudXR disconnected: \(err.kind): \(err.localizedDescription)")
                #endif
                lastError = err.localizedDescription
            }
            hasPublishedXRStarted = false
            // An intentional full disconnect tears both transports down; skip the
            // recovery so it can't republish against a session being nilled.
            guard !isTearingDown else { return }
            recoverMic()
            return
        }

        // Reconcile against live state so a coalesced observation that misses the
        // `.connected` edge still publishes once.
        if case .connected = newState {
            publishXRStartedWhenConnected()
        }
    }

    /// Re-arms `withObservationTracking` to drive edge-triggered side effects; UI
    /// reads computed `xrState`.
    private func beginObservingXRState() {
        guard let s = cloudxrSession else { return }
        withObservationTracking {
            _ = s.state
        } onChange: { [weak self] in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.handleCloudxrStateChange(self.cloudxrSession?.state)
                self.beginObservingXRState()
            }
        }
    }

    /// Publish `xr.session.started` once connected, retrying on a transient send
    /// failure while both transports stay up. A duplicate is harmless (treated
    /// idempotently); a missed signal is not.
    private func publishXRStartedWhenConnected() {
        guard !hasPublishedXRStarted, !isPublishingXRStarted else { return }
        isPublishingXRStarted = true
        Task { [weak self] in
            guard let self else { return }
            defer { self.isPublishingXRStarted = false }
            for _ in 0 ..< Self.xrStartedMaxAttempts {
                // Stop if either transport dropped: the signal is only meaningful
                // while XR is connected and the data channel is up.
                guard case .connected? = self.cloudxrSession?.state,
                      self.connectionState == .connected else { return }
                if await self.publishXRStarted() {
                    self.hasPublishedXRStarted = true
                    return
                }
                try? await Task.sleep(nanoseconds: Self.xrStartedRetryNanos)
            }
        }
    }

    /// Send the `xr.session.started` signal once; returns whether it succeeded.
    private func publishXRStarted() async -> Bool {
        guard let session else { return false }
        do {
            try await session.send(Data(), reliable: true, topic: Self.xrSessionStartedTopic)
            return true
        } catch {
            lastError = "xr.session.started publish failed: \(error.localizedDescription)"
            return false
        }
    }

    #endif
}

// MARK: - ReceivedMessage

struct ReceivedMessage: Identifiable {
    let id = UUID()
    let text: String
    let timestamp = Date()
}
