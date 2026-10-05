// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Testing
@testable import StreamKit

private actor Gate {
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
        for waiter in pending { waiter.resume() }
    }
}

private enum CaptureFailure: Error { case startup, cleanup }

private actor Capture {
    var events: [String] = []
    var cleanupFails = false
    func failCleanup() { cleanupFails = true }
    func allowCleanup() { cleanupFails = false }
    func record(_ event: String) { events.append(event) }
    func clean() throws {
        try Task.checkCancellation()
        events.append("cleanup")
        if cleanupFails { throw CaptureFailure.cleanup }
    }
}

/// Uses the production transaction queue without opening hardware or a room.
private final class TransactionBackend: StreamingBackend, @unchecked Sendable {
    var onConnectionStateChanged: (@Sendable (StreamKit.ConnectionState) -> Void)?
    var onDataReceived: (@Sendable (String, Data) -> Void)?
    var onAgentStatus: (@Sendable (String) -> Void)?
    var onNetworkMetrics: (@Sendable (NetworkMetrics) -> Void)?
    let operations = MicrophoneOperations()
    let capture = Capture()
    let entered = Gate()
    func connect(config: SessionConfig) async throws {}
    @MainActor func startAudio(config: AudioConfig) async throws {
        try await operations.start(config: config) { [self] in
            await entered.release()
            try await Task.sleep(for: .seconds(60))
            Issue.record("Disconnect did not cancel the suspended start")
        } cleanup: { [self] in try await capture.clean() }
    }
    @MainActor func stopAudio() async throws {
        try await operations.stop { [self] in try await capture.clean() }
    }
    @MainActor func disconnect() async throws {
        try await operations.disconnect { [self] in try await capture.clean() }
        close: { [self] in await capture.record("disconnect") }
    }
    func startCamera(config: CameraConfig) async throws {}
    func stopCamera() async throws {}
    func send(_ data: Data, reliable: Bool) async throws {}
}

@Suite("Microphone transactions", .timeLimit(.minutes(1)))
struct MicrophoneOperationsTests {
    @Test @MainActor func stopBeforePreparationCancelsRegisteredStart() async throws {
        let backend = TransactionBackend()
        let session = StreamSession(backend: backend)
        // Run both callers on the registration executor. Stop enters before the
        // queued prepare task gets a turn; it must still find and cancel Start.
        let start = Task { try await session.startAudio() }
        let stop = Task { try await session.stopAudio() }
        try await stop.value
        do { try await start.value; Issue.record("Stopped start succeeded") }
        catch { #expect(error is CancellationError) }
        #expect(backend.operations.state == .idle)
        #expect(await backend.capture.events == ["cleanup"])
    }

    @Test func rollbackFailureStaysPendingUntilSuccessfulStop() async throws {
        let operations = MicrophoneOperations()
        let capture = Capture()
        do {
            try await operations.start(config: .default) {
                await capture.failCleanup()
                throw CaptureFailure.startup
            } cleanup: { try await capture.clean() }
            Issue.record("Startup unexpectedly succeeded")
        } catch StreamError.microphoneCleanupFailed(let startup, let cleanup) {
            #expect(startup as? CaptureFailure == .startup)
            #expect(cleanup as? CaptureFailure == .cleanup)
        }
        #expect(await operations.state == .cleanupRequired)
        do {
            try await operations.stop { try await capture.clean() }
            Issue.record("Failed stop was swallowed")
        } catch { #expect(error as? CaptureFailure == .cleanup) }
        #expect(await operations.state == .cleanupRequired)
        await capture.allowCleanup()
        try await operations.stop { try await capture.clean() }
        #expect(await operations.state == .idle)
    }

    @Test func cancellationAfterPublicationPropagatesRollbackFailure() async throws {
        let operations = MicrophoneOperations()
        let capture = Capture()
        let published = Gate()
        let resume = Gate()
        let start = Task {
            try await operations.start(config: .default) {
                await capture.failCleanup()
                await published.release()
                await resume.wait()
            } cleanup: { try await capture.clean() }
        }
        await published.wait()
        start.cancel()
        await resume.release()
        do {
            try await start.value
            Issue.record("Cancelled startup succeeded")
        } catch StreamError.microphoneCleanupFailed(let startup, let cleanup) {
            #expect(startup is CancellationError)
            #expect(cleanup as? CaptureFailure == .cleanup)
        }
        #expect(await operations.state == .cleanupRequired)
    }

    @Test func rollbackFinishesBeforeNextStart() async throws {
        let operations = MicrophoneOperations()
        let capture = Capture()
        let cleaning = Gate()
        let resumeCleanup = Gate()
        let first = Task {
            try await operations.start(config: .default) {
                await capture.record("published")
                throw CaptureFailure.startup
            } cleanup: {
                try Task.checkCancellation()
                if await capture.events.contains("published") {
                    await cleaning.release()
                    await resumeCleanup.wait()
                }
                await capture.record("cleaned")
            }
        }
        await cleaning.wait()
        let next = Task {
            try await operations.start(config: .raw) { await capture.record("next start") }
            cleanup: { try await capture.clean() }
        }
        while await operations.pendingStartCount < 2 { await Task.yield() }
        first.cancel()
        await resumeCleanup.release()
        do { try await first.value; Issue.record("Startup failure was swallowed") }
        catch { #expect(error as? CaptureFailure == .startup) }
        try await next.value
        #expect(await capture.events == ["cleaned", "published", "cleaned", "cleanup", "next start"])
        #expect(await operations.state == .active)
    }

    @Test func cancelledStopStillCompletes() async throws {
        let operations = MicrophoneOperations()
        let entered = Gate()
        let resume = Gate()
        let stop = Task {
            try await operations.stop {
                await entered.release()
                await resume.wait()
                try Task.checkCancellation()
            }
        }
        await entered.wait()
        stop.cancel()
        await resume.release()
        try await stop.value
        #expect(await operations.state == .idle)
    }

    @Test func failedPrestartCleanupDoesNotPrepareCapture() async throws {
        let operations = MicrophoneOperations()
        let capture = Capture()
        await capture.failCleanup()
        do {
            try await operations.start(config: .default) { await capture.record("unexpected start") }
            cleanup: { try await capture.clean() }
            Issue.record("Failed cleanup was swallowed")
        } catch StreamError.microphoneCleanupFailed(let startup, let cleanup) {
            #expect(startup == nil)
            #expect(cleanup as? CaptureFailure == .cleanup)
        }
        #expect(await capture.events == ["cleanup"])
        #expect(await operations.state == .cleanupRequired)
        await capture.allowCleanup()
        try await operations.start(config: .default) { await capture.record("started") }
        cleanup: { try await capture.clean() }
        #expect(await operations.state == .active)
        #expect(await capture.events == ["cleanup", "cleanup", "started"])
    }

    @Test func stopCancelsBothActiveAndQueuedStarts() async throws {
        let operations = MicrophoneOperations()
        let entered = Gate()
        let capture = Capture()
        let first = Task {
            try await operations.start(config: .default) {
                await entered.release()
                try await Task.sleep(for: .seconds(60))
            } cleanup: { try await capture.clean() }
        }
        await entered.wait()
        let next = Task {
            try await operations.start(config: .raw) { await capture.record("unexpected start") }
            cleanup: { try await capture.clean() }
        }
        while await operations.pendingStartCount < 2 { await Task.yield() }
        try await operations.stop { try await capture.clean() }
        for task in [first, next] {
            do { try await task.value; Issue.record("Start was not cancelled") }
            catch { #expect(error is CancellationError) }
        }
        #expect(await operations.state == .idle)
        #expect(await capture.events == ["cleanup", "cleanup", "cleanup"])
    }

    @Test func disconnectClosesEvenWhenCleanupFailsAndRemainsPending() async throws {
        let operations = MicrophoneOperations()
        let capture = Capture()
        await capture.failCleanup()
        do {
            try await operations.disconnect { try await capture.clean() }
            close: { await capture.record("disconnect") }
            Issue.record("Cleanup failure was swallowed")
        } catch { #expect(error as? CaptureFailure == .cleanup) }
        #expect(await capture.events == ["cleanup", "disconnect"])
        #expect(await operations.state == .cleanupRequired)
    }

    @Test @MainActor func directSessionDisconnectCancelsSuspendedStart() async throws {
        let backend = TransactionBackend()
        let session = StreamSession(backend: backend)
        let start = Task { try await session.startAudio() }
        await backend.entered.wait()
        try await session.disconnect()
        do { try await start.value; Issue.record("Suspended start was not cancelled") }
        catch { #expect(error is CancellationError) }
        #expect(backend.operations.state == .idle)
        #expect(await backend.capture.events == ["cleanup", "cleanup", "cleanup", "disconnect"])
    }

    @Test @MainActor func publicDisabledPresetStopsWithoutStartingCapture() async throws {
        let backend = TransactionBackend()
        let session = StreamSession(backend: backend)
        try await session.startAudio(config: .disabled)
        #expect(backend.operations.state == .idle)
        #expect(await backend.capture.events == ["cleanup"])
    }

    @Test @MainActor func sessionDisconnectPropagatesCleanupFailureAndCanRetry() async throws {
        let backend = TransactionBackend()
        let session = StreamSession(backend: backend)
        await backend.capture.failCleanup()
        do { try await session.disconnect(); Issue.record("Cleanup failure was swallowed") }
        catch { #expect(error as? CaptureFailure == .cleanup) }
        #expect(backend.operations.state == .cleanupRequired)
        #expect(await backend.capture.events == ["cleanup", "disconnect"])
        await backend.capture.allowCleanup()
        try await session.disconnect()
        #expect(backend.operations.state == .idle)
    }
}
