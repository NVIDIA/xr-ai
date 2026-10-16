// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import CoreMedia
import CoreVideo
import Foundation
@testable import LiveKit
import Testing
@testable import StreamKit

@MainActor
private final class CameraBoundaryGate {
    private var entered = false
    private var observer: CheckedContinuation<Void, Never>?
    private var pending: CheckedContinuation<Void, Never>?

    func hold() async {
        entered = true
        observer?.resume()
        observer = nil
        await withCheckedContinuation { pending = $0 }
    }

    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { observer = $0 }
    }

    func release() {
        pending?.resume()
        pending = nil
    }
}

private enum CameraBoundaryFailure: Error { case stop, unpublish, frameAllocation }

/// Runs public backend methods; only the SDK/network failure boundary is controlled.
@MainActor
private final class CameraLifecycleFixture {
    var backend: LiveKitBackend!
    var rooms: [Room] = []
    var tracks: [LocalVideoTrack] = []
    var starts: [LocalVideoTrack] = []
    var connectGate: CameraBoundaryGate?
    var unpublishGate: CameraBoundaryGate?
    var publishGate: CameraBoundaryGate?
    var stopFailures = 0
    var failPartialStart = false
    var startGate: CameraBoundaryGate?
    var failUnpublish = false
    var unpublishCount = 0
    var stopCount = 0
    var disconnectCount = 0
    var states: [StreamKit.ConnectionState] = []

    init() {
        var operations = LiveKitBackend.SDKOperations()
        operations.start = { [weak self] track in
            guard let self else { return }
            self.starts.append(track)
            await self.startGate?.hold()
            if self.failPartialStart {
                _ = try await track.capturer.startCapture()
                throw CameraBoundaryFailure.stop
            }
            try await track.start()
        }
        operations.connect = { [weak self] room, _, _, _, _ in
            guard let self else { return }
            self.rooms.append(room)
            await self.connectGate?.hold()
            room._state.mutate { $0.connectionState = .connected }
        }
        operations.publish = { [weak self] room, track, _ in
            guard let self else { throw CancellationError() }
            try await track.start()
            self.tracks.append(track)
            var info = Livekit_TrackInfo()
            info.sid = UUID().uuidString
            info.type = .video
            info.source = .camera
            let publication = LocalTrackPublication(info: info, participant: room.localParticipant)
            _ = await publication.set(track: track)
            room.localParticipant.add(publication: publication)
            await self.publishGate?.hold()
            return publication
        }
        operations.unpublish = { [weak self] room, publication in
            guard let self else { return }
            self.unpublishCount += 1
            // Match the SDK: its registry entry is removed BEFORE track.stop.
            room.localParticipant._state.mutate { $0.trackPublications.removeValue(forKey: publication.sid) }
            await self.unpublishGate?.hold()
            if self.failUnpublish { throw CameraBoundaryFailure.unpublish }
            if let track = publication.track as? LocalVideoTrack { try await track.stop() }
        }
        operations.disconnect = { [weak self] room in
            guard let self else { return }
            self.disconnectCount += 1
            // SDK disconnect cannot reach a publication removed by failed unpublish.
            for publication in room.localParticipant.trackPublications.values {
                try? await publication.track?.stop()
            }
            room.localParticipant._state.mutate { $0.trackPublications.removeAll() }
            room._state.mutate { $0.connectionState = .disconnected }
        }
        operations.stop = { [weak self] track in
            guard let self else { return }
            self.stopCount += 1
            if self.stopFailures > 0 {
                self.stopFailures -= 1
                throw CameraBoundaryFailure.stop
            }
            try await track.stop()
        }
        backend = LiveKitBackend(config: LiveKitConfig(host: "test", token: "test", hubIdentity: nil), sdk: operations)
        backend.onConnectionStateChanged = { [weak self] state in
            Task { @MainActor in self?.states.append(state) }
        }
    }

    nonisolated func frame() throws -> sending CMSampleBuffer {
        var pixel: CVPixelBuffer?
        let pixelStatus = CVPixelBufferCreate(kCFAllocatorDefault, 640, 480, kCVPixelFormatType_32BGRA, nil, &pixel)
        guard pixelStatus == kCVReturnSuccess, let image = pixel else { throw CameraBoundaryFailure.frameAllocation }
        var format: CMVideoFormatDescription?
        let formatStatus = CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
                                                                       imageBuffer: image, formatDescriptionOut: &format)
        guard formatStatus == noErr, let format else { throw CameraBoundaryFailure.frameAllocation }
        var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: .zero, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        let sampleStatus = CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: image,
                                                                   formatDescription: format,
                                                                   sampleTiming: &timing, sampleBufferOut: &sample)
        guard sampleStatus == noErr, let sample else { throw CameraBoundaryFailure.frameAllocation }
        return sample
    }
}

@Suite("Camera backend lifecycle", .serialized)
@MainActor
struct CameraLifecycleTests {
    @Test func failedUnpublishAndDisconnectKeepStartedCaptureUntilRetry() async throws {
        let f = CameraLifecycleFixture()
        try await f.backend.connect(config: .default)
        try await f.backend.injectVideoFrame(f.frame())
        let original = try #require(f.tracks.first)
        #expect(original.capturer._state.startStopCounter == 1)
        f.failUnpublish = true
        f.stopFailures = 2
        await #expect(throws: CameraBoundaryFailure.self) { try await f.backend.stopCamera() }
        #expect(f.rooms[0].localParticipant.trackPublications.isEmpty)
        await #expect(throws: CancellationError.self) { try await f.backend.injectVideoFrame(f.frame()) }
        await #expect(throws: CameraBoundaryFailure.self) { try await f.backend.disconnect() }
        #expect(f.disconnectCount == 1)
        #expect(original.capturer._state.startStopCounter == 1)
        await #expect(throws: StreamKit.StreamError.self) { try await f.backend.injectVideoFrame(f.frame()) }
        f.failUnpublish = false
        try await f.backend.connect(config: .default)
        #expect(original.capturer._state.startStopCounter == 0)
        try await f.backend.injectVideoFrame(f.frame())
        #expect(f.tracks.count == 2 && f.tracks[1] !== original)
        try await f.backend.disconnect()
    }

    @Test func concurrentStopsJoinAndLateFramesCannotRecreateCamera() async throws {
        let f = CameraLifecycleFixture()
        try await f.backend.connect(config: .default)
        try await f.backend.injectVideoFrame(f.frame())
        let gate = CameraBoundaryGate()
        f.unpublishGate = gate
        let first = Task { try await f.backend.stopCamera() }
        defer { gate.release() }
        await gate.waitUntilEntered()
        // The entry continuation resumes on MainActor; stopCamera registers its
        // task before this observer can resume, so the second call is admitted.
        var second: Task<Void, any Error>!
        await withCheckedContinuation { (entered: CheckedContinuation<Void, Never>) in
            second = Task { @MainActor in
                entered.resume()
                try await f.backend.stopCamera()
            }
        }
        await #expect(throws: CancellationError.self) { try await f.backend.injectVideoFrame(f.frame()) }
        gate.release()
        try await first.value
        try await second.value
        #expect(f.unpublishCount == 1 && f.tracks.count == 1)
        #expect(f.tracks[0].capturer._state.startStopCounter == 0)
        f.unpublishGate = nil
        try await f.backend.disconnect()
    }

    @Test func lateJoinCannotReplaceTheNewRoomOrItsCamera() async throws {
        let f = CameraLifecycleFixture()
        let gate = CameraBoundaryGate()
        f.connectGate = gate
        let first = Task { try await f.backend.connect(config: .default) }
        defer { gate.release() }
        await gate.waitUntilEntered()
        let oldRoom = try #require(f.rooms.first)
        f.connectGate = nil
        try await f.backend.connect(config: .default)
        try await f.backend.injectVideoFrame(f.frame())
        let current = try #require(f.tracks.first)
        gate.release()
        await #expect(throws: StreamKit.StreamError.self) { try await first.value }
        #expect(f.rooms.count == 2 && f.rooms[1].connectionState == .connected)
        #expect(oldRoom.connectionState == .disconnected)
        #expect(current.capturer._state.startStopCounter == 1)
        try await f.backend.disconnect()
        #expect(current.capturer._state.startStopCounter == 0)
    }

    @Test func partialTrackStartBalancesCapturerEvenThoughTrackNeverStarted() async throws {
        let f = CameraLifecycleFixture()
        try await f.backend.connect(config: .default)
        f.failPartialStart = true
        await #expect(throws: CameraBoundaryFailure.self) { try await f.backend.injectVideoFrame(f.frame()) }
        #expect(f.tracks.isEmpty)
        #expect(f.starts.count == 1)
        #expect(f.starts[0].capturer._state.startStopCounter == 0)
        f.failPartialStart = false
        try await f.backend.injectVideoFrame(f.frame())
        #expect(f.tracks.count == 1)
        #expect(f.tracks[0].capturer._state.startStopCounter == 1)
        try await f.backend.disconnect()
        #expect(f.tracks[0].capturer._state.startStopCounter == 0)
    }

    @Test func concurrentStartsAndStopDrainBlockedCaptureBeforeAnyNewPublication() async throws {
        let f = CameraLifecycleFixture()
        try await f.backend.connect(config: .default)
        let gate = CameraBoundaryGate()
        f.startGate = gate
        let first = Task { try await f.backend.startCamera(config: .default) }
        defer { gate.release() }
        await gate.waitUntilEntered()
        var second: Task<Void, any Error>!
        await withCheckedContinuation { (entered: CheckedContinuation<Void, Never>) in
            second = Task { @MainActor in
                entered.resume()
                try await f.backend.startCamera(config: .default)
            }
        }
        var stopped: Task<Void, any Error>!
        await withCheckedContinuation { (entered: CheckedContinuation<Void, Never>) in
            stopped = Task { @MainActor in
                entered.resume()
                try await f.backend.stopCamera()
            }
        }
        await #expect(throws: CancellationError.self) { try await f.backend.injectVideoFrame(f.frame()) }
        #expect(f.tracks.isEmpty)
        gate.release()
        await #expect(throws: CancellationError.self) { try await first.value }
        await #expect(throws: CancellationError.self) { try await second.value }
        try await stopped.value
        #expect(f.tracks.isEmpty)
        #expect(f.starts.count == 1 && f.starts[0].capturer._state.startStopCounter == 0)
        f.startGate = nil
        try await f.backend.startCamera(config: .default)
        #expect(f.tracks.count == 1)
        try await f.backend.disconnect()
        #expect(f.tracks[0].capturer._state.startStopCounter == 0)
    }

    @Test func terminalRoomCallbackAttemptsCleanupAndRetainsFailedExactCapture() async throws {
        let f = CameraLifecycleFixture()
        try await f.backend.connect(config: .default)
        try await f.backend.injectVideoFrame(f.frame())
        let original = try #require(f.tracks.first)
        let closedRoom = f.rooms[0]
        // Model native terminal cleanup losing its registry entry before a
        // failed stop, then delivering the RoomDelegate terminal callback.
        closedRoom.localParticipant._state.mutate { $0.trackPublications.removeAll() }
        closedRoom._state.mutate { $0.connectionState = .disconnected }
        f.stopFailures = 1
        await withCheckedContinuation { (closed: CheckedContinuation<Void, Never>) in
            f.backend.onConnectionStateChanged = { state in
                if state == .disconnected { closed.resume() }
            }
            f.backend.room(closedRoom, didUpdateConnectionState: .disconnected, from: .connected)
        }
        #expect(f.stopCount == 1)
        #expect(original.capturer._state.startStopCounter == 1)
        await #expect(throws: StreamKit.StreamError.self) { try await f.backend.injectVideoFrame(f.frame()) }
        // No registry is needed to retry the retained actual resource.
        try await f.backend.disconnect()
        #expect(original.capturer._state.startStopCounter == 0)
    }

    @Test func startDuringStopWaitsForExactEarlierCleanupBeforeAllocatingCapture() async throws {
        let f = CameraLifecycleFixture()
        try await f.backend.connect(config: .default)
        try await f.backend.injectVideoFrame(f.frame())
        let original = try #require(f.tracks.first)
        let gate = CameraBoundaryGate()
        f.unpublishGate = gate
        let stopped = Task { try await f.backend.stopCamera() }
        defer { gate.release() }
        await gate.waitUntilEntered()
        var started: Task<Void, any Error>!
        await withCheckedContinuation { (entered: CheckedContinuation<Void, Never>) in
            started = Task { @MainActor in
                entered.resume()
                try await f.backend.startCamera(config: .default)
            }
        }
        #expect(f.unpublishCount == 1 && f.starts.count == 1 && f.tracks.count == 1)
        gate.release()
        try await stopped.value
        try await started.value
        #expect(f.unpublishCount == 1 && f.starts.count == 2 && f.tracks.count == 2)
        #expect(original.capturer._state.startStopCounter == 0)
        #expect(f.tracks[1] !== original && f.tracks[1].capturer._state.startStopCounter == 1)
        f.unpublishGate = nil
        try await f.backend.disconnect()
        #expect(f.tracks[1].capturer._state.startStopCounter == 0)
    }

    @Test func stopStartStopCancelsInterveningStartInsteadOfJoiningOldIntent() async throws {
        let f = CameraLifecycleFixture()
        try await f.backend.connect(config: .default)
        try await f.backend.injectVideoFrame(f.frame())
        let gate = CameraBoundaryGate()
        f.unpublishGate = gate
        let firstStop = Task { try await f.backend.stopCamera() }
        defer { gate.release() }
        await gate.waitUntilEntered()
        var start: Task<Void, any Error>!
        await withCheckedContinuation { (entered: CheckedContinuation<Void, Never>) in
            start = Task { @MainActor in entered.resume(); try await f.backend.startCamera(config: .default) }
        }
        var lastStop: Task<Void, any Error>!
        await withCheckedContinuation { (entered: CheckedContinuation<Void, Never>) in
            lastStop = Task { @MainActor in entered.resume(); try await f.backend.stopCamera() }
        }
        gate.release()
        try await firstStop.value
        await #expect(throws: CancellationError.self) { try await start.value }
        try await lastStop.value
        #expect(f.starts.count == 1 && f.tracks.count == 1 && f.unpublishCount == 1)
        #expect(f.tracks[0].capturer._state.startStopCounter == 0)
        try await f.backend.stopCamera() // idle stop is an observable no-op
        #expect(f.unpublishCount == 1)
        f.unpublishGate = nil
        try await f.backend.disconnect()
    }

    @Test func startStopStartOnlyAllowsIntentAdmittedAfterStop() async throws {
        let f = CameraLifecycleFixture()
        try await f.backend.connect(config: .default)
        let gate = CameraBoundaryGate()
        f.startGate = gate
        let firstStart = Task { try await f.backend.startCamera(config: .default) }
        defer { gate.release() }
        await gate.waitUntilEntered()
        var stop: Task<Void, any Error>!
        await withCheckedContinuation { (entered: CheckedContinuation<Void, Never>) in
            stop = Task { @MainActor in entered.resume(); try await f.backend.stopCamera() }
        }
        f.startGate = nil
        var lastStart: Task<Void, any Error>!
        await withCheckedContinuation { (entered: CheckedContinuation<Void, Never>) in
            lastStart = Task { @MainActor in entered.resume(); try await f.backend.startCamera(config: .default) }
        }
        gate.release()
        await #expect(throws: CancellationError.self) { try await firstStart.value }
        try await stop.value
        try await lastStart.value
        #expect(f.starts.count == 2 && f.tracks.count == 1)
        #expect(f.starts[0].capturer._state.startStopCounter == 0)
        #expect(f.tracks[0].capturer._state.startStopCounter == 1)
        try await f.backend.disconnect()
    }

    @Test func persistentCameraCleanupDoesNotPreventTransportReconnect() async throws {
        let f = CameraLifecycleFixture()
        try await f.backend.connect(config: .default)
        try await f.backend.injectVideoFrame(f.frame())
        f.failUnpublish = true
        f.stopFailures = 3
        await #expect(throws: CameraBoundaryFailure.self) { try await f.backend.stopCamera() }
        try await f.backend.connect(config: .default)
        #expect(f.rooms.count == 2 && f.rooms[1].connectionState == .connected)
        await #expect(throws: CameraBoundaryFailure.self) { try await f.backend.startCamera(config: .default) }
        #expect(f.tracks.count == 1) // no replacement discarded the failed owner
        f.failUnpublish = false
        try await f.backend.disconnect()
        #expect(f.tracks[0].capturer._state.startStopCounter == 0)
    }

    @Test func callerCancellationAfterSDKPublicationRetainsExactRollbackOwner() async throws {
        let f = CameraLifecycleFixture()
        try await f.backend.connect(config: .default)
        let gate = CameraBoundaryGate()
        f.publishGate = gate
        let publication = Task { try await f.backend.injectVideoFrame(f.frame()) }
        defer { gate.release() }
        await gate.waitUntilEntered()
        #expect(f.tracks.count == 1 && !f.rooms[0].localParticipant.trackPublications.isEmpty)
        publication.cancel() // no stop/generation change hides caller cancellation
        gate.release()
        await #expect(throws: CancellationError.self) { try await publication.value }
        #expect(f.rooms[0].localParticipant.trackPublications.isEmpty)
        #expect(f.tracks[0].capturer._state.startStopCounter == 0)
        #expect(f.unpublishCount == 1)
        f.publishGate = nil
        try await f.backend.disconnect()
    }
}
