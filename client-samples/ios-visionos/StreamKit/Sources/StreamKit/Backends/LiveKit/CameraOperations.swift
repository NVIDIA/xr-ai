// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// One ordered owner for complete camera transitions, including failed rollback.
/// Admission is synchronous on the session executor; native cleanup is not
/// cancelled when a caller stops waiting for it.
@MainActor
final class CameraOperations {
    nonisolated init() {}
    private var tail: Task<Void, any Error>?
    private var starts: [UUID: Task<Void, any Error>] = [:]
    private var stops = 0
    var isStopping: Bool { stops != 0 }
    var hasPendingStart: Bool { !starts.isEmpty }

    func start(_ operation: @escaping @MainActor @Sendable () async throws -> Void) async throws {
        let id = UUID()
        let task = enqueue(operation)
        starts[id] = task
        defer { starts[id] = nil }
        try await withTaskCancellationHandler { try await task.value } onCancel: { task.cancel() }
    }

    func stop(_ cleanup: @escaping @MainActor @Sendable () async throws -> Void) async throws {
        // This applies on EVERY admission, even when another stop is pending:
        // an intervening queued start must not outlive the later stop.
        for task in starts.values { task.cancel() }
        stops += 1
        let task = enqueue(cleanup)
        defer { stops -= 1 }
        try await task.value
    }

    private func enqueue(_ operation: @escaping @MainActor @Sendable () async throws -> Void) -> Task<Void, any Error> {
        let previous = tail
        let task = Task {
            _ = await previous?.result
            try Task.checkCancellation()
            try await operation()
        }
        tail = task
        return task
    }
}
