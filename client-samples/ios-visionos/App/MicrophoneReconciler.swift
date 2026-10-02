// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// App-private intent owner. Every completed physical transition is accounted for
/// before applying the latest request; callers never cancel the drain task.
@MainActor
final class MicrophoneReconciler {
    private(set) var isEnabled = false
    private(set) var isActive = false
    private var closed = false
    private var needsStop = false
    private var revision: UInt64 = 0
    private var drain: Task<Void, Never>?
    private let start: () async throws -> Void
    private let stop: () async throws -> Void
    private let changed: (Bool, Bool) -> Void
    private let failed: (Error) -> Void

    init(start: @escaping () async throws -> Void,
         stop: @escaping () async throws -> Void,
         changed: @escaping (Bool, Bool) -> Void,
         failed: @escaping (Error) -> Void) {
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
            if !enabled { needsStop = true }
        }
        await reconcile().value
    }

    func restart() async {
        guard isEnabled, !closed else { return }
        needsStop = true
        revision &+= 1
        await reconcile().value
    }

    /// Closes intent synchronously, including from a disconnected notification.
    @discardableResult
    func close() -> Task<Void, Never> {
        closed = true
        isEnabled = false
        needsStop = true
        revision &+= 1
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
                        failed(error)
                        if attempt == revision { return }
                    }
                } else if isEnabled && !isActive {
                    changed(false, true)
                    do {
                        try await start()
                        isActive = true
                        changed(true, false)
                    } catch {
                        changed(false, false)
                        failed(error)
                        // Explicit enable/recovery may retry, but a stable failing
                        // request must not spin forever in the background.
                        if attempt == revision { return }
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
