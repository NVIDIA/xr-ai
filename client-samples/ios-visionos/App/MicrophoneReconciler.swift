// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// App-private intent owner. Every completed physical transition is accounted for
/// before applying the latest request; callers never cancel the drain task.
@MainActor
final class MicrophoneReconciler {
    enum Operation { case start, stop }
    private(set) var isEnabled = false
    private(set) var isActive = false
    private var closed = false
    private var needsStop = false
    private var restartRequested = false
    private var revision: UInt64 = 0
    private var drain: Task<Void, Never>?
    private var pendingStart: Task<Void, Error>?
    private let start: () async throws -> Void
    private let stop: () async throws -> Void
    private let changed: (Bool, Bool) -> Void
    private let failed: (Error, Operation) -> Void

    init(start: @escaping () async throws -> Void,
         stop: @escaping () async throws -> Void,
         changed: @escaping (Bool, Bool) -> Void,
         failed: @escaping (Error, Operation) -> Void) {
        self.start = start
        self.stop = stop
        self.changed = changed
        self.failed = failed
    }

    func setEnabled(_ enabled: Bool) async {
        guard !closed else { return }
        if isEnabled != enabled {
            isEnabled = enabled
            revision &+= 1
            // Preserve a requested stop even if Start arrives before it completes.
            if !enabled {
                needsStop = true
                restartRequested = false
            }
        }
        if !enabled { pendingStart?.cancel() }
        await reconcile().value
    }

    /// Registers recovery before returning its completion task.
    func restart() -> Task<Void, Never> {
        guard isEnabled, !closed else { return Task {} }
        needsStop = true
        restartRequested = true
        revision &+= 1
        let completion = reconcile()
        return Task { await completion.value }
    }

    /// Closes intent synchronously, including from a disconnected notification.
    @discardableResult
    func close() -> Task<Void, Never> {
        if !closed {
            closed = true
            isEnabled = false
            needsStop = true
            revision &+= 1
        }
        pendingStart?.cancel()
        return reconcile()
    }

    private func reconcile() -> Task<Void, Never> {
        if let drain { return drain }
        let task = Task {
            defer { drain = nil }
            while true {
                let attempt = revision
                if needsStop || (!isEnabled && isActive) {
                    needsStop = false
                    do {
                        try await stop()
                        isActive = false
                        changed(false, false)
                    } catch {
                        // Do not report idle when physical teardown failed.
                        needsStop = true
                        failed(error, .stop)
                        if attempt == revision { return }
                    }
                } else if isEnabled && !isActive {
                    let isRestart = restartRequested
                    restartRequested = false
                    changed(false, true)
                    let startup = Task { try await start() }
                    pendingStart = startup
                    do {
                        try await startup.value
                        pendingStart = nil
                        isActive = true
                        changed(true, false)
                    } catch {
                        pendingStart = nil
                        changed(false, false)
                        // The backend may wrap cancellation in its own error type.
                        // Suppress only the startup explicitly cancelled by an off request.
                        if !startup.isCancelled { failed(error, .start) }
                        // Explicit enable/recovery may retry, but a stable failing
                        // request must not spin forever in the background.
                        if attempt == revision {
                            // A failed manual Start must not re-arm on a later
                            // OS event. Recovery retains intent for bounded retries.
                            if !isRestart { isEnabled = false }
                            return
                        }
                    }
                } else {
                    return
                }
            }
        }
        drain = task
        return task
    }
}
