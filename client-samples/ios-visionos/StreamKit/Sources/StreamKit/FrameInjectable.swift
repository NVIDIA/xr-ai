// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * StreamKit — FrameInjectable
 *
 * Protocol for streaming video from an external camera source — such as the
 * Meta wearables SDK — by pushing CMSampleBuffers directly into the LiveKit track.
 *
 * ## Device workflow (e.g. Meta wearables SDK)
 *
 * ```swift
 * // 1. Connect to the session.
 * try await session.connect()
 *
 * // 2. Route frames according to the user's saved camera mode:
 * metaSDK.onFrame = { sampleBuffer in
 *     switch cameraMode {
 *     case .live:
 *         try? await session.injectVideoFrame(sampleBuffer)
 *     case .onDemand:
 *         latestCameraFrame = sampleBuffer // encode from the request handler
 *     case .off:
 *         break
 *     }
 * }
 * ```
 *
 * ## Simulator workflow
 *
 * On the simulator `startCamera()` automatically generates synthetic test frames
 * at ~30 fps using this same injection path, so the full video pipeline can be
 * exercised without wearable hardware.
 *
 * ## Frame publication
 *
 * The first `injectVideoFrame(_:)` call creates a `BufferCapturer`-backed
 * `LocalVideoTrack` and publishes it to the LiveKit room. The track is published
 * after the first frame (not before) because LiveKit requires at least one captured
 * frame to resolve the stream's dimensions before it can complete the publish handshake.
 * Applications with a camera-authorization selector should therefore call this API only
 * in their live-video mode. For on-demand images, retain or request a frame in the
 * external-camera adapter and return its encoded bytes from `onImageCaptureRequested`;
 * leaving the handler unset disables that path.
 */

import CoreMedia
import Foundation

// MARK: - FrameInjectable

/// Implemented by backends that accept externally produced video frames.
///
/// ``LiveKitBackend`` conforms to this protocol. Custom ``StreamingBackend``
/// implementations can optionally conform to integrate their own video pipeline.
public protocol FrameInjectable: AnyObject, Sendable {

    /// Push a video frame from an external camera source into the published video track.
    ///
    /// A `BufferCapturer`-backed LiveKit track is created on the first call and
    /// published to the room automatically once the frame dimensions are known.
    /// Call this only after the user has selected live video; the method always
    /// represents a publishing path, not an unpublished still-frame cache.
    ///
    /// - Parameter sampleBuffer: A `CMSampleBuffer` containing a `CVPixelBuffer`.
    ///   The pixel format must be one of LiveKit's supported formats
    ///   (`kCVPixelFormatType_420YpCbCr8BiPlanarFullRange`,
    ///    `kCVPixelFormatType_32BGRA`, etc.).
    /// - Throws: ``StreamError/notConnected`` if not connected.
    func injectVideoFrame(_ sampleBuffer: sending CMSampleBuffer) async throws
}
