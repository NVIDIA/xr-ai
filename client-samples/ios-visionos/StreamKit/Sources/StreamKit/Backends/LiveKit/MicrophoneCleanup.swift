// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

enum MicrophoneCleanup {
    /// Every publication and engine release is attempted, even after a failure.
    static func stop<Publication: Sendable>(
        publications: [Publication],
        mute: @Sendable (Publication) async throws -> Void,
        unpublish: @Sendable (Publication) async throws -> Void,
        release: @Sendable () async throws -> Void
    ) async throws {
        var failure: (any Error)?
        for publication in publications {
            try? await mute(publication)
            do { try await unpublish(publication) }
            catch { if failure == nil { failure = error } }
        }
        do { try await release() }
        catch { if failure == nil { failure = error } }
        if let failure { throw failure }
    }

    static func releaseEngine(
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
