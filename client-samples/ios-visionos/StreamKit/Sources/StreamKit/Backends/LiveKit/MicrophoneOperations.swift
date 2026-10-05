// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Serializes complete capture transactions, not just the synchronous actor work.
actor MicrophoneOperations {
    enum State { case idle, active, cleanupRequired }
    private(set) var state: State = .idle
    private var tail: Task<Void, Error>?
    private var starts: [UUID: Task<Void, Error>] = [:]
    var pendingStartCount: Int { starts.count }

    func start(
        config: AudioConfig,
        prepare: @escaping @Sendable () async throws -> Void,
        cleanup: @escaping @Sendable () async throws -> Void
    ) async throws {
        if config.mode == .disabled {
            try await stop(cleanup: cleanup)
            return
        }
        let id = UUID()
        let task = enqueue { operations in
            // Also remove publications retained or republished by the SDK.
            do { try await operations.clean(cleanup) }
            catch { throw StreamError.microphoneCleanupFailed(startup: nil, cleanup: error) }
            operations.state = .cleanupRequired
            do {
                try Task.checkCancellation()
                try await prepare()
                try Task.checkCancellation()
                operations.state = .active
            } catch {
                // LiveKit may wrap cancellation in its own error type.
                let startupError: any Error = Task.isCancelled ? CancellationError() : error
                // A cancelled start must not cancel its own rollback.
                do { try await operations.clean(cleanup) }
                catch {
                    throw StreamError.microphoneCleanupFailed(startup: startupError, cleanup: error)
                }
                throw startupError
            }
        }
        starts[id] = task
        defer { starts[id] = nil }
        try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    func stop(cleanup: @escaping @Sendable () async throws -> Void) async throws {
        cancelStarts()
        let task = enqueue { operations in try await operations.clean(cleanup) }
        // Neither a stop nor disconnect inherits caller cancellation.
        try await task.value
    }

    func disconnect(
        cleanup: @escaping @Sendable () async throws -> Void,
        close: @escaping @Sendable () async -> Void
    ) async throws {
        cancelStarts()
        let task = enqueue { operations in
            let result: Result<Void, Error>
            do {
                try await operations.clean(cleanup)
                result = .success(())
            } catch { result = .failure(error) }
            // Close the transport even if capture teardown fails, within the queue.
            await close()
            try result.get()
        }
        try await task.value
    }

    private func clean(_ cleanup: @escaping @Sendable () async throws -> Void) async throws {
        state = .cleanupRequired
        try await Task { try await cleanup() }.value
        state = .idle
    }

    private func cancelStarts() {
        for task in starts.values { task.cancel() }
    }

    private func enqueue(
        _ operation: @escaping @Sendable (isolated MicrophoneOperations) async throws -> Void
    ) -> Task<Void, Error> {
        let previous = tail
        let task = Task {
            _ = await previous?.result
            try Task.checkCancellation()
            try await operation(self)
        }
        tail = task
        return task
    }
}
