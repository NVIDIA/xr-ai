// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Testing
@testable import StreamKit

private final class Publication: Sendable {
    let id: Int
    init(_ id: Int) { self.id = id }
}

private enum CleanupFailure: Error { case mute, firstPublication, secondPublication, prepared, input }
private actor CleanupEvents {
    var values: [String] = []
    func record(_ value: String) { values.append(value) }
}

@Suite("Microphone cleanup aggregation")
struct MicrophoneCleanupTests {
    @Test @MainActor func removedPublicationRemainsAnObligationAcrossDisconnectAndStart() async throws {
        let cleanup = MicrophoneCleanup()
        let operations = MicrophoneOperations()
        let owner = PublicationOwner()
        let stop: @Sendable () async throws -> Void = {
            try await cleanup.stop(publications: await owner.publications) { _ in }
            unpublish: { try await owner.unpublish($0) }
            release: { await owner.recordRelease() }
        }
        do {
            try await operations.disconnect(cleanup: stop, close: { await owner.close() })
            Issue.record("Cleanup failure was swallowed")
        } catch { #expect(error as? CleanupFailure == .firstPublication) }
        #expect(await owner.publications.isEmpty)
        #expect(await owner.captureRunning)
        #expect(operations.state == .cleanupRequired)

        // Fresh enumeration is empty after SDK removal and transport close.
        do {
            try await operations.start(config: .default) {
                Issue.record("Start ran with an outstanding publication")
            } cleanup: { try await stop() }
            Issue.record("Start ignored pending cleanup")
        } catch StreamError.microphoneCleanupFailed { }
        #expect(await owner.unpublishAttempts == [1, 1])
        #expect(operations.state == .cleanupRequired)

        await owner.allowCleanup()
        try await operations.stop(cleanup: stop)
        #expect(await owner.unpublishAttempts == [1, 1, 1])
        #expect(await owner.releaseAttempts == 3)
        #expect(await owner.closed)
        #expect(await !owner.captureRunning)
        #expect(operations.state == .idle)
        try await operations.stop(cleanup: stop)
        #expect(await owner.unpublishAttempts == [1, 1, 1])
    }

    @Test func firstUnpublishFailureDoesNotSkipOtherPublicationsOrEngineRelease() async throws {
        let events = CleanupEvents()
        do {
            try await MicrophoneCleanup().stop(publications: [Publication(1), Publication(2)]) { id in
                await events.record("mute \(id.id)")
            } unpublish: { id in
                await events.record("unpublish \(id.id)")
                throw id.id == 1 ? CleanupFailure.firstPublication : CleanupFailure.secondPublication
            } release: {
                await events.record("release")
                throw CleanupFailure.input
            }
            Issue.record("Cleanup failure was swallowed")
        } catch { #expect(error as? CleanupFailure == .firstPublication) }
        #expect(await events.values == ["mute 1", "unpublish 1", "mute 2", "unpublish 2", "release"])
    }

    @Test func successfulUnpublishPropagatesReleaseFailure() async throws {
        let events = CleanupEvents()
        do {
            try await MicrophoneCleanup().stop(publications: [Publication(1)]) { _ in }
            unpublish: { _ in await events.record("unpublish") }
            release: { await events.record("release"); throw CleanupFailure.input }
            Issue.record("Release failure was swallowed")
        } catch { #expect(error as? CleanupFailure == .input) }
        #expect(await events.values == ["unpublish", "release"])
    }

    @Test func muteFailureDoesNotPreventSuccessfulCleanup() async throws {
        let events = CleanupEvents()
        try await MicrophoneCleanup().stop(publications: [Publication(1)]) { _ in
            await events.record("mute")
            throw CleanupFailure.mute
        } unpublish: { _ in await events.record("unpublish") }
        release: { await events.record("release") }
        #expect(await events.values == ["mute", "unpublish", "release"])
    }

    @Test func preparedFailureStillReleasesInputAndWinsOverLaterError() async throws {
        let events = CleanupEvents()
        do {
            try await MicrophoneCleanup.releaseEngine {
                await events.record("prepared")
                throw CleanupFailure.prepared
            } releaseInput: {
                await events.record("input")
                throw CleanupFailure.input
            }
            Issue.record("Engine failure was swallowed")
        } catch { #expect(error as? CleanupFailure == .prepared) }
        #expect(await events.values == ["prepared", "input"])
    }

    @Test func noPublicationsStillReleasesEngine() async throws {
        let events = CleanupEvents()
        try await MicrophoneCleanup().stop(publications: [Publication]()) { _ in Issue.record("Unexpected mute") }
        unpublish: { _ in Issue.record("Unexpected unpublish") }
        release: { await events.record("release") }
        #expect(await events.values == ["release"])
    }
}

/// Mirrors LiveKit's registry mutation before a fallible track stop.
private actor PublicationOwner {
    var publications = [Publication(1)]
    var captureRunning = true
    var closed = false
    var unpublishAttempts: [Int] = []
    var releaseAttempts = 0
    private var fails = true
    func allowCleanup() { fails = false }
    func close() { closed = true }
    func recordRelease() { releaseAttempts += 1 }
    func unpublish(_ publication: Publication) throws {
        unpublishAttempts.append(publication.id)
        publications.removeAll { $0 === publication }
        if fails { throw CleanupFailure.firstPublication }
        captureRunning = false
    }
}
