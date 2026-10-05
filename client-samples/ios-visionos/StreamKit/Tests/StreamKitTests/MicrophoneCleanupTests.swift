// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Testing
@testable import StreamKit

private enum CleanupFailure: Error { case mute, firstPublication, secondPublication, prepared, input }
private actor CleanupEvents {
    var values: [String] = []
    func record(_ value: String) { values.append(value) }
}

@Suite("Microphone cleanup aggregation")
struct MicrophoneCleanupTests {
    @Test func firstUnpublishFailureDoesNotSkipOtherPublicationsOrEngineRelease() async throws {
        let events = CleanupEvents()
        do {
            try await MicrophoneCleanup.stop(publications: [1, 2]) { id in
                await events.record("mute \(id)")
            } unpublish: { id in
                await events.record("unpublish \(id)")
                throw id == 1 ? CleanupFailure.firstPublication : CleanupFailure.secondPublication
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
            try await MicrophoneCleanup.stop(publications: [1]) { _ in }
            unpublish: { _ in await events.record("unpublish") }
            release: { await events.record("release"); throw CleanupFailure.input }
            Issue.record("Release failure was swallowed")
        } catch { #expect(error as? CleanupFailure == .input) }
        #expect(await events.values == ["unpublish", "release"])
    }

    @Test func muteFailureDoesNotPreventSuccessfulCleanup() async throws {
        let events = CleanupEvents()
        try await MicrophoneCleanup.stop(publications: [1]) { _ in
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
        try await MicrophoneCleanup.stop(publications: [Int]()) { _ in Issue.record("Unexpected mute") }
        unpublish: { _ in Issue.record("Unexpected unpublish") }
        release: { await events.record("release") }
        #expect(await events.values == ["release"])
    }
}
