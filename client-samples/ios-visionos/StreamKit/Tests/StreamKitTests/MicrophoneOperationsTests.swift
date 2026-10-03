// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Testing
@testable import StreamKit

private actor MicrophoneTestGate {
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

private actor MicrophoneTestEvents {
    var values: [String] = []
    func append(_ value: String) { values.append(value) }
}

@Suite("Microphone transition ordering", .timeLimit(.minutes(1)))
struct MicrophoneOperationsTests {
    @Test func rollbackIgnoresCallerCancellationAndPreservesOriginalError() async throws {
        enum OriginalFailure: Error { case startup }
        let entered = MicrophoneTestGate()
        let resume = MicrophoneTestGate()
        let events = MicrophoneTestEvents()
        let task = Task {
            try await MicrophoneOperations.withRollback {
                await entered.release()
                await resume.wait()
                throw OriginalFailure.startup
            } cleanup: {
                try Task.checkCancellation()
                await events.append("cleanup completed")
            } cleanupFailed: { _ in
                Issue.record("Rollback inherited startup cancellation")
            }
        }
        await entered.wait()
        task.cancel()
        await resume.release()
        do {
            try await task.value
            Issue.record("Original startup failure was swallowed")
        } catch { #expect(error is OriginalFailure) }
        #expect(await events.values == ["cleanup completed"])
    }

    @Test func cancelledStartFinishesCleanupBeforeNextStart() async throws {
        let operations = MicrophoneOperations()
        let started = MicrophoneTestGate()
        let resumeStart = MicrophoneTestGate()
        let cleaning = MicrophoneTestGate()
        let resumeCleanup = MicrophoneTestGate()
        let events = MicrophoneTestEvents()
        let first = Task {
            try await operations.run {
                try await MicrophoneOperations.withRollback {
                    await started.release()
                    await resumeStart.wait()
                    try Task.checkCancellation()
                } cleanup: {
                    try Task.checkCancellation()
                    await events.append("cleanup started")
                    await cleaning.release()
                    await resumeCleanup.wait()
                    await events.append("cleanup finished")
                } cleanupFailed: { _ in
                    Issue.record("Rollback failed")
                }
            }
        }
        await started.wait()
        first.cancel()
        await resumeStart.release()
        await cleaning.wait()
        let next = await operations.enqueue { await events.append("next start") }
        await resumeCleanup.release()
        do {
            try await first.value
            Issue.record("Cancelled start unexpectedly succeeded")
        } catch { #expect(error is CancellationError) }
        try await next.value
        #expect(await events.values == ["cleanup started", "cleanup finished", "next start"])
    }

    @Test func cancelledQueuedStartNeverTouchesEngine() async throws {
        let operations = MicrophoneOperations()
        let entered = MicrophoneTestGate()
        let resume = MicrophoneTestGate()
        let events = MicrophoneTestEvents()
        let first = Task {
            try await operations.run {
                await entered.release()
                await resume.wait()
            }
        }
        await entered.wait()
        let cancelled = await operations.enqueue { await events.append("unexpected start") }
        cancelled.cancel()
        await resume.release()
        try await first.value
        do {
            try await cancelled.value
            Issue.record("Cancelled queued start unexpectedly succeeded")
        } catch { #expect(error is CancellationError) }
        #expect(await events.values.isEmpty)
    }

    @Test func stopCompletesEvenWhenCallerIsCancelled() async throws {
        let operations = MicrophoneOperations()
        let entered = MicrophoneTestGate()
        let resume = MicrophoneTestGate()
        let events = MicrophoneTestEvents()
        let stop = Task {
            try await operations.run(cancelWithCaller: false) {
                await entered.release()
                await resume.wait()
                try Task.checkCancellation()
                await events.append("stopped")
            }
        }
        await entered.wait()
        stop.cancel()
        await resume.release()
        try await stop.value
        #expect(await events.values == ["stopped"])
    }

    @Test func engineFailurePropagatesWithoutPoisoningNextOperation() async throws {
        enum EngineFailure: Error { case unavailable }
        let operations = MicrophoneOperations()
        do {
            try await operations.run { throw EngineFailure.unavailable }
            Issue.record("Engine failure was swallowed")
        } catch { #expect(error is EngineFailure) }
        let events = MicrophoneTestEvents()
        try await operations.run(cancelWithCaller: false) { await events.append("cleanup") }
        #expect(await events.values == ["cleanup"])
    }
}
