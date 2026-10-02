// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Testing
@testable import MicrophoneLifecycle

actor Gate {
    private var open = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        if open { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func release() {
        open = true
        waiters.forEach { $0.resume() }
        waiters.removeAll()
    }
}

/// Uses the production backend queue, but replaces LiveKit/AVAudioEngine calls
/// with observable publication/input state and deterministic suspension points.
@MainActor
final class TestMicrophoneBackend {
    let operations = MicrophoneOperations()
    let enteredStart = Gate()
    var holdStart: Gate?
    let enteredStop = Gate()
    var holdStop: Gate?
    var published = false
    var inputRunning = false
    var failStart = false
    var failUnpublish = false
    var failRelease = false
    var waitForCancellation = false
    var onStarted: (() -> Void)?
    var rollbackFailures = 0
    var events: [String] = []
    enum Failure: Error { case start, unpublish, release }

    func start() async throws {
        try await operations.run { [self] in try await startPhysical() }
    }
    func stop() async throws {
        try await operations.run(cancelWithCaller: false) { [self] in try await stopPhysical() }
    }
    private func startPhysical() async throws {
        try await stopPhysical()
        try await MicrophoneOperations.withRollback {
            events.append("start")
            inputRunning = true
            await enteredStart.release()
            await holdStart?.wait()
            if waitForCancellation { try await Task.sleep(nanoseconds: 30_000_000_000) }
            try Task.checkCancellation()
            if failStart { throw Failure.start }
            published = true
            onStarted?()
        } cleanup: {
            try await stopPhysical()
        } cleanupFailed: { _ in
            rollbackFailures += 1
        }
    }
    private func stopPhysical() async throws {
        events.append("stop")
        if let holdStop {
            await enteredStop.release()
            await holdStop.wait()
        }
        var failure: Failure?
        if failUnpublish { failure = .unpublish }
        else { published = false }
        if failRelease { if failure == nil { failure = .release } }
        else { inputRunning = false }
        if let failure { throw failure }
    }
}

@MainActor
private final class Fixture {
    let backend = TestMicrophoneBackend()
    var active = false
    var starting = false
    var errors = 0
    lazy var mic = MicrophoneReconciler(
        start: { [backend] in try await backend.start() },
        stop: { [backend] in try await backend.stop() },
        changed: { [weak self] active, starting in
            self?.active = active
            self?.starting = starting
        },
        failed: { [weak self] _, _ in self?.errors += 1 }
    )

    func expectCapture(_ expected: Bool) {
        #expect(active == expected)
        #expect(mic.isActive == expected)
        #expect(backend.published == expected)
        #expect(backend.inputRunning == expected)
        #expect(!starting)
    }
}

@Suite("Composed microphone lifecycle", .timeLimit(.minutes(1)))
@MainActor
struct MicrophoneLifecycleTests {
    @Test func duplicateStart() async {
        let f = Fixture()
        let gate = Gate()
        f.backend.holdStart = gate
        let first = Task { await f.mic.setEnabled(true) }
        await f.backend.enteredStart.wait()
        let second = Task { await f.mic.setEnabled(true) }
        await gate.release()
        await first.value
        await second.value
        f.expectCapture(true)
        #expect(f.backend.events == ["stop", "start"])
    }

    @Test func stopDuringSuspendedStart() async {
        let f = Fixture()
        let gate = Gate()
        f.backend.holdStart = gate
        let start = Task { await f.mic.setEnabled(true) }
        await f.backend.enteredStart.wait()
        let stop = Task { await f.mic.setEnabled(false) }
        while f.mic.isEnabled { await Task.yield() }
        await gate.release()
        await start.value
        await stop.value
        f.expectCapture(false)
        #expect(f.backend.events.last == "stop")
        #expect(f.errors == 0)
    }

    @Test func startStopStartDuringSuspendedStart() async {
        let f = Fixture()
        let gate = Gate()
        f.backend.holdStart = gate
        let first = Task { await f.mic.setEnabled(true) }
        await f.backend.enteredStart.wait()
        let stop = Task { await f.mic.setEnabled(false) }
        while f.mic.isEnabled { await Task.yield() }
        let last = Task { await f.mic.setEnabled(true) }
        while !f.mic.isEnabled { await Task.yield() }
        await gate.release()
        await first.value
        await stop.value
        await last.value
        f.expectCapture(true)
        #expect(f.backend.events.filter { $0 == "start" }.count == 2)
        #expect(f.backend.events.last == "start")
        #expect(f.errors == 0)
    }

    @Test func startDuringSuspendedStop() async {
        let f = Fixture()
        await f.mic.setEnabled(true)
        let gate = Gate()
        f.backend.holdStop = gate
        let stop = Task { await f.mic.setEnabled(false) }
        await f.backend.enteredStop.wait()
        let start = Task { await f.mic.setEnabled(true) }
        while !f.mic.isEnabled { await Task.yield() }
        await gate.release()
        await stop.value
        await start.value
        f.expectCapture(true)
        #expect(f.backend.events == ["stop", "start", "stop", "stop", "start"])
    }

    @Test func stopDuringRecovery() async {
        let f = Fixture()
        await f.mic.setEnabled(true)
        let gate = Gate()
        f.backend.holdStop = gate
        let recovery = Task { await f.mic.restart() }
        await f.backend.enteredStop.wait()
        let stop = Task { await f.mic.setEnabled(false) }
        while f.mic.isEnabled { await Task.yield() }
        recovery.cancel()
        await gate.release()
        await recovery.value
        await stop.value
        f.expectCapture(false)
        #expect(f.backend.events.filter { $0 == "start" }.count == 1)
    }

    @Test func disconnectDuringSuspendedStart() async {
        let f = Fixture()
        let gate = Gate()
        f.backend.holdStart = gate
        let start = Task { await f.mic.setEnabled(true) }
        await f.backend.enteredStart.wait()
        let close = f.mic.close()
        await f.mic.setEnabled(true)
        start.cancel()
        await gate.release()
        await close.value
        await start.value
        f.expectCapture(false)
        #expect(!f.mic.isEnabled)
        #expect(f.backend.events.last == "stop")
        #expect(f.errors == 0)
    }

    @Test func failureCanRetryWithoutSpinning() async {
        let f = Fixture()
        f.backend.failStart = true
        await f.mic.setEnabled(true)
        f.expectCapture(false)
        #expect(f.errors == 1)
        #expect(f.backend.events == ["stop", "start", "stop"])
        f.backend.failStart = false
        await f.mic.setEnabled(true)
        f.expectCapture(true)
        #expect(f.backend.events == ["stop", "start", "stop", "stop", "start"])
    }

    @Test func partialStopFailureDoesNotClaimIdle() async {
        let f = Fixture()
        await f.mic.setEnabled(true)
        f.backend.failRelease = true
        await f.mic.setEnabled(false)
        #expect(f.active && f.mic.isActive)
        #expect(!f.backend.published && f.backend.inputRunning)
        #expect(f.errors == 1)
        f.backend.failRelease = false
        await f.mic.setEnabled(false)
        f.expectCapture(false)
    }

    @Test func unpublishFailureStillReleasesInput() async {
        let f = Fixture()
        await f.mic.setEnabled(true)
        f.backend.failUnpublish = true
        await f.mic.setEnabled(false)
        #expect(f.backend.published && !f.backend.inputRunning)
        #expect(f.active && f.errors == 1)
        f.backend.failUnpublish = false
        await f.mic.setEnabled(false)
        f.expectCapture(false)
    }

    @Test func closeCancelsPendingStartWithoutWaitingForFrames() async {
        let f = Fixture()
        f.backend.waitForCancellation = true
        let start = Task { await f.mic.setEnabled(true) }
        await f.backend.enteredStart.wait()
        await f.mic.close().value
        await start.value
        f.expectCapture(false)
        #expect(f.errors == 0)
    }

    @Test func repeatedCloseDoesNotRepeatSuccessfulStop() async {
        let f = Fixture()
        await f.mic.setEnabled(true)
        await f.mic.close().value
        let events = f.backend.events
        await f.mic.close().value
        await f.mic.close().value
        #expect(f.backend.events == events)
        f.expectCapture(false)
    }

    @Test func closeRetriesFailedStop() async {
        let f = Fixture()
        await f.mic.setEnabled(true)
        f.backend.failRelease = true
        await f.mic.close().value
        #expect(f.backend.inputRunning)
        f.backend.failRelease = false
        await f.mic.close().value
        f.expectCapture(false)
    }

    @Test func restartDoesNotCancelPendingStart() async {
        let f = Fixture()
        let gate = Gate()
        f.backend.holdStart = gate
        let first = Task { await f.mic.setEnabled(true) }
        await f.backend.enteredStart.wait()
        let recovery = Task { await f.mic.restart() }
        await gate.release()
        await first.value
        await recovery.value
        f.expectCapture(true)
        #expect(f.errors == 0)
    }
}
