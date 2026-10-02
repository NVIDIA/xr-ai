// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Testing
@testable import MicrophoneLifecycle

private actor Gate {
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
private final class Backend {
    let operations = MicrophoneOperations()
    let enteredStart = Gate()
    var holdStart: Gate?
    let enteredStop = Gate()
    var holdStop: Gate?
    var published = false
    var inputRunning = false
    var failStart = false
    var failStop = false
    var events: [String] = []
    enum Failure: Error { case start, stop }

    func start() async throws {
        try await operations.run { [self] in try await startPhysical() }
    }
    func stop() async throws {
        try await operations.run(cancelWithCaller: false) { [self] in try await stopPhysical() }
    }
    private func startPhysical() async throws {
        events.append("start")
        inputRunning = true
        await enteredStart.release()
        await holdStart?.wait()
        if failStart {
            inputRunning = false
            throw Failure.start
        }
        published = true
    }
    private func stopPhysical() async throws {
        events.append("stop")
        await enteredStop.release()
        await holdStop?.wait()
        if failStop { throw Failure.stop }
        published = false
        inputRunning = false
    }
}

@MainActor
private final class Fixture {
    let backend = Backend()
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
        failed: { [weak self] _ in self?.errors += 1 }
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
        #expect(f.backend.events == ["start"])
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
        #expect(f.backend.events == ["start", "stop"])
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
        #expect(f.backend.events == ["start", "stop", "start"])
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
        #expect(f.backend.events == ["start", "stop", "start"])
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
        #expect(!f.backend.events.dropFirst().contains("start"))
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
        #expect(f.backend.events == ["start", "stop"])
    }

    @Test func failureCanRetryWithoutSpinning() async {
        let f = Fixture()
        f.backend.failStart = true
        await f.mic.setEnabled(true)
        f.expectCapture(false)
        #expect(f.errors == 1)
        #expect(f.backend.events == ["start"])
        f.backend.failStart = false
        await f.mic.setEnabled(true)
        f.expectCapture(true)
        #expect(f.backend.events == ["start", "start"])
    }

    @Test func stopFailureDoesNotClaimIdle() async {
        let f = Fixture()
        await f.mic.setEnabled(true)
        f.backend.failStop = true
        await f.mic.setEnabled(false)
        f.expectCapture(true)
        #expect(f.errors == 1)
        f.backend.failStop = false
        await f.mic.setEnabled(false)
        f.expectCapture(false)
    }
}
