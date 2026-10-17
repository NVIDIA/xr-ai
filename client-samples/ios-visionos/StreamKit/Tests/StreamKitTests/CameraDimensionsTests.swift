// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import CoreMedia
import CoreVideo
@testable import LiveKit
import Testing
@testable import StreamKit

@Suite("Camera first-frame metadata")
struct CameraDimensionsTests {
    @Test @MainActor func realBufferCapturerResolvesDimensionsAndDetachesObserver() async throws {
        let track = await LocalVideoTrack.createBufferTrack(name: "test", source: .camera)
        let capturer = try #require(track.capturer as? BufferCapturer)
        let before = capturer.delegates.countDelegates
        let waiting = Task { try await cameraDimensions(capturer) }
        var pixel: CVPixelBuffer?
        #expect(CVPixelBufferCreate(kCFAllocatorDefault, 640, 480, kCVPixelFormatType_32BGRA, nil, &pixel) == kCVReturnSuccess)
        let image = try #require(pixel)
        var format: CMVideoFormatDescription?
        #expect(CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
                                                            imageBuffer: image, formatDescriptionOut: &format) == noErr)
        var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: .zero, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        let descriptor = try #require(format)
        #expect(CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: image,
                                                        formatDescription: descriptor,
                                                        sampleTiming: &timing, sampleBufferOut: &sample) == noErr)
        capturer.capture(try #require(sample))
        let dimensions = try await waiting.value
        #expect(dimensions.width == 640 && dimensions.height == 480)
        #expect(capturer.delegates.countDelegates == before)
        try await track.stop()
    }

    @Test @MainActor func cancellationBeforeAnyFrameDetachesObserver() async throws {
        let track = await LocalVideoTrack.createBufferTrack(name: "cancel", source: .camera)
        let before = track.capturer.delegates.countDelegates
        let waiting = Task { try await cameraDimensions(track.capturer) }
        waiting.cancel()
        await #expect(throws: CancellationError.self) { try await waiting.value }
        #expect(track.capturer.delegates.countDelegates == before)
        try await track.stop()
    }

    @Test @MainActor func captureDeadlineFailureDetachesObserver() async throws {
        let track = await LocalVideoTrack.createBufferTrack(name: "deadline", source: .camera)
        let before = track.capturer.delegates.countDelegates
        await #expect(throws: LiveKitError.self) { try await cameraDimensions(track.capturer) }
        #expect(track.capturer.delegates.countDelegates == before)
        try await track.stop()
    }

    @Test @MainActor func SDKTrackStartAndStopBalanceCaptureExactlyOnce() async throws {
        let track = await LocalVideoTrack.createBufferTrack(name: "ownership", source: .camera)
        try await track.start()
        try await track.start()
        #expect(track.capturer._state.startStopCounter == 1)
        try await track.stop()
        #expect(track.capturer._state.startStopCounter == 0)
    }
}
