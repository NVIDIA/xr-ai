// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import AVFoundation
import Foundation
import StreamKit
import Testing

private actor AppGate {
    private var open = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        if open { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func release() {
        open = true
        let pending = waiters
        waiters.removeAll()
        pending.forEach { $0.resume() }
    }
}

private enum AppCaptureError: Error { case failed }
private actor AppCapture {
    let started = AppGate()
    let startMayFinish = AppGate()
    let stopping = AppGate()
    let stopMayFinish = AppGate()
    var starts = 0
    var stops = 0
    var closed = false
    var delayStart = false
    var delayStop = false
    var failStart = false
    var failStop = false
    var failDisconnect = false

    func configure(delayStart: Bool = false, delayStop: Bool = false,
                   failStart: Bool = false, failStop: Bool = false, failDisconnect: Bool = false) {
        self.delayStart = delayStart
        self.delayStop = delayStop
        self.failStart = failStart
        self.failStop = failStop
        self.failDisconnect = failDisconnect
    }
    func start() async throws {
        starts += 1
        await started.release()
        if delayStart { await startMayFinish.wait() }
        if failStart { throw AppCaptureError.failed }
    }
    func stop() async throws {
        stops += 1
        await stopping.release()
        if delayStop { await stopMayFinish.wait() }
        if failStop { throw AppCaptureError.failed }
    }
    func disconnect() throws {
        closed = true
        if failDisconnect { throw AppCaptureError.failed }
    }
}

private final class AppBackend: StreamingBackend, @unchecked Sendable {
    var onConnectionStateChanged: (@Sendable (ConnectionState) -> Void)?
    var onDataReceived: (@Sendable (String, Data) -> Void)?
    var onAgentStatus: (@Sendable (String) -> Void)?
    var onNetworkMetrics: (@Sendable (NetworkMetrics) -> Void)?
    let capture = AppCapture()
    func connect(config: SessionConfig) async throws {}
    func disconnect() async throws { try await capture.disconnect() }
    func startAudio(config: AudioConfig) async throws { try await capture.start() }
    func stopAudio() async throws { try await capture.stop() }
    func startCamera(config: CameraConfig) async throws {}
    func stopCamera() async throws {}
    func send(_ data: Data, reliable: Bool) async throws {}
}

// The Xcode target compiles the full production AppModel with a fake transport.
@Suite("App microphone requests", .serialized, .timeLimit(.minutes(1)))
@MainActor
struct AppModelMicrophoneTests {
    private func model(_ backend: AppBackend) -> AppModel {
        let model = AppModel()
        model.session = StreamSession(backend: backend)
        model.connectionState = .connected
        model.audioMode = .raw
        return model
    }

    @Test func startDuringStopWaitsThenActuallyStarts() async throws {
        let backend = AppBackend()
        let model = model(backend)
        await model.enableMic()
        await backend.capture.configure(delayStop: true)
        let stop = Task { await model.disableMic() }
        await backend.capture.stopping.wait()
        #expect(!model.isAudioActive)
        let start = Task { await model.enableMic() }
        while !model.micEnabledByUser { await Task.yield() }
        #expect(await backend.capture.starts == 1)
        await backend.capture.stopMayFinish.release()
        await stop.value
        await start.value
        #expect(model.isAudioActive)
        #expect(model.micEnabledByUser)
        #expect(!model.micCleanupRequired)
        #expect(await backend.capture.starts == 2)
    }

    @Test func stopAccountsForLateStartupBeforeApplyingNewStart() async throws {
        let backend = AppBackend()
        let model = model(backend)
        await backend.capture.configure(delayStart: true, delayStop: true)
        let first = Task { await model.enableMic() }
        await backend.capture.started.wait()
        let stop = Task { await model.disableMic() }
        await backend.capture.stopping.wait()
        let next = Task { await model.enableMic() }
        while !model.micEnabledByUser { await Task.yield() }
        await backend.capture.startMayFinish.release()
        await first.value
        await backend.capture.stopMayFinish.release()
        await stop.value
        await next.value
        #expect(await backend.capture.starts == 2)
        #expect(model.isAudioActive)
        #expect(!model.micCleanupRequired)
    }

    @Test func successfulStopClearsLateStartupAndIntent() async throws {
        let backend = AppBackend()
        let model = model(backend)
        await backend.capture.configure(delayStart: true, delayStop: true)
        let start = Task { await model.enableMic() }
        await backend.capture.started.wait()
        let stop = Task { await model.disableMic() }
        await backend.capture.stopping.wait()
        await backend.capture.startMayFinish.release()
        await start.value
        await backend.capture.stopMayFinish.release()
        await stop.value
        #expect(!model.isAudioActive)
        #expect(!model.micEnabledByUser)
        #expect(!model.micCleanupRequired)
    }

    @Test func disconnectFailureIsReportedAndNextSuccessfulCleanupClearsIt() async throws {
        let backend = AppBackend()
        let model = model(backend)
        await model.enableMic()
        await backend.capture.configure(failDisconnect: true)
        await model.disconnect()
        #expect(await backend.capture.closed)
        #expect(model.session == nil)
        #expect(!model.isAudioActive)
        #expect(!model.micEnabledByUser)
        #expect(model.micCleanupRequired)
        #expect(model.lastError?.contains("Reconnect") == true)
        let next = AppBackend()
        model.session = StreamSession(backend: next)
        model.connectionState = .connected
        await model.disconnect()
        #expect(!model.micCleanupRequired)
    }

    @Test func failedManualStartDoesNotLeaveRecoveryIntent() async throws {
        let backend = AppBackend()
        let model = model(backend)
        await backend.capture.configure(failStart: true)
        await model.enableMic()
        #expect(!model.isAudioActive)
        #expect(!model.micEnabledByUser)
        await backend.capture.configure()
        await model.startAudio()
        #expect(await backend.capture.starts == 1)
    }

    @Test func failedStopKeepsCleanupRetryAvailable() async throws {
        let backend = AppBackend()
        let model = model(backend)
        await model.enableMic()
        await backend.capture.configure(failStop: true)
        await model.disableMic()
        #expect(model.micCleanupRequired)
        #expect(model.isAudioActive)
        #expect(!model.micEnabledByUser)
        await backend.capture.configure()
        await model.disableMic()
        #expect(!model.micCleanupRequired)
        #expect(!model.isAudioActive)
    }

    @Test func offlineRecoveryWaitsForReconnectWithoutConsumingAttempts() async throws {
        let backend = AppBackend()
        let model = model(backend)
        await model.enableMic()
        model.connectionState = .reconnecting
        NotificationCenter.default.post(name: AVAudioSession.mediaServicesWereResetNotification, object: nil)
        #expect(model.micRecoveryTask == nil)
        #expect(await backend.capture.stops == 0)
        #expect(model.micEnabledByUser)
        model.isAudioActive = false
        model.connectionState = .connected
        model.resumeMicAfterReconnect()
        while await backend.capture.starts < 2 { await Task.yield() }
        // Join the app's in-flight startup before checking its observable state.
        await model.startAudio()
        #expect(model.isAudioActive)
        #expect(await backend.capture.starts == 2)
    }
}
