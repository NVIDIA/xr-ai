// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import LiveKit

private final class CameraDimensionsObserver: NSObject, VideoCapturerDelegate, Sendable {
    let stream: AsyncStream<Dimensions>
    private let continuation: AsyncStream<Dimensions>.Continuation

    override init() {
        let pair = AsyncStream<Dimensions>.makeStream(bufferingPolicy: .bufferingNewest(1))
        stream = pair.stream
        continuation = pair.continuation
        super.init()
    }

    func capturer(_ capturer: VideoCapturer, didUpdate dimensions: Dimensions?) {
        if let dimensions { continuation.yield(dimensions) }
    }

    func capturer(_ capturer: VideoCapturer, didUpdate state: VideoCapturer.CapturerState) {
        if state == .stopped { continuation.finish() }
    }
}

/// Await the actual first frame using the SDK's own capture-start deadline.
/// Cancellation removes this observer and does not leave a raw continuation.
func cameraDimensions(_ capturer: VideoCapturer) async throws -> Dimensions {
    if let dimensions = capturer.dimensions { return dimensions }
    let observer = CameraDimensionsObserver()
    capturer.delegates.add(delegate: observer)
    defer { capturer.delegates.remove(delegate: observer) }
    // Capture may have resolved dimensions between the first read and attach.
    if let dimensions = capturer.dimensions { return dimensions }
    return try await withThrowingTaskGroup(of: Dimensions.self) { group in
        group.addTask {
            for await dimensions in observer.stream { return dimensions }
            throw CancellationError()
        }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(TimeInterval.defaultCaptureStart * 1_000_000_000))
            throw LiveKitError(.capturerDimensionsNotResolved)
        }
        defer { group.cancelAll() }
        guard let dimensions = try await group.next() else { throw CancellationError() }
        return dimensions
    }
}
