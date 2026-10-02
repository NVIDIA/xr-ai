// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Serializes the entire microphone transition, including suspended engine cleanup.
/// Actor isolation alone would allow a subsequent start to interleave with a stop.
actor MicrophoneOperations {
    private var tail: Task<Void, Error>?

    /// Returns only after the transition is registered behind its predecessor.
    func enqueue(
        _ operation: @escaping @Sendable () async throws -> Void
    ) -> Task<Void, Error> {
        let previous = tail
        let task = Task {
            // A failed or cancelled transition must finish cleanup before the next one.
            _ = await previous?.result
            try Task.checkCancellation()
            try await operation()
        }
        tail = task
        return task
    }

    func run(
        cancelWithCaller: Bool = true,
        _ operation: @escaping @Sendable () async throws -> Void
    ) async throws {
        let task = enqueue(operation)
        try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            // Stop/disconnect cleanup must run even when its caller was cancelled.
            if cancelWithCaller { task.cancel() }
        }
    }

    /// Capture failures must finish rollback before the next queued transition runs.
    static func withRollback(
        isolation: isolated (any Actor)? = #isolation,
        _ operation: () async throws -> Void,
        cleanup: () async throws -> Void,
        cleanupFailed: (Error) -> Void
    ) async throws {
        do { try await operation() }
        catch {
            do { try await cleanup() }
            catch { cleanupFailed(error) }
            throw error
        }
    }
}
