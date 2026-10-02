// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Serializes the entire microphone transition, including suspended engine cleanup.
/// Actor isolation alone would allow a subsequent start to interleave with a stop.
actor MicrophoneOperations {
    private var tail: Task<Void, Error>?

    func run(
        cancelWithCaller: Bool = true,
        _ operation: @escaping @Sendable () async throws -> Void
    ) async throws {
        let previous = tail
        let task = Task {
            // A failed or cancelled transition must finish cleanup before the next one.
            _ = await previous?.result
            try Task.checkCancellation()
            try await operation()
        }
        tail = task
        try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            // Stop/disconnect cleanup must run even when its caller was cancelled.
            if cancelWithCaller { task.cancel() }
        }
    }
}
