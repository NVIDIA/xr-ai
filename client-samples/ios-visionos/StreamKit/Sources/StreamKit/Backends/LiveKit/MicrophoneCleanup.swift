// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Called only within the serialized microphone transaction.
@MainActor
final class MicrophoneCleanup {
    nonisolated init() {}
    private var pending: [(id: ObjectIdentifier, unpublish: @Sendable () async throws -> Void)] = []

    /// Retain exact resources until teardown succeeds, even if the SDK removes
    /// their publications from its registry before throwing.
    func stop<Publication: AnyObject & Sendable>(
        publications: [Publication],
        mute: @escaping @Sendable (Publication) async throws -> Void,
        unpublish: @escaping @Sendable (Publication) async throws -> Void,
        release: @Sendable () async throws -> Void
    ) async throws {
        for publication in publications {
            let id = ObjectIdentifier(publication)
            guard !pending.contains(where: { $0.id == id }) else { continue }
            pending.append((id, {
                try? await mute(publication)
                try await unpublish(publication)
            }))
        }
        var failure: (any Error)?
        for obligation in pending {
            do {
                try await obligation.unpublish()
                pending.removeAll { $0.id == obligation.id }
            } catch { if failure == nil { failure = error } }
        }
        do { try await release() }
        catch { if failure == nil { failure = error } }
        if let failure { throw failure }
    }

    nonisolated static func releaseEngine(
        releasePrepared: @Sendable () async throws -> Void,
        releaseInput: @Sendable () async throws -> Void
    ) async throws {
        var failure: (any Error)?
        do { try await releasePrepared() }
        catch { failure = error }
        do { try await releaseInput() }
        catch { if failure == nil { failure = error } }
        if let failure { throw failure }
    }
}
