// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * StreamKit — FrameSink
 *
 * Optional mixin for backends that accept externally produced video frames —
 * e.g. from a game engine render target, a hardware capture card, or an XR
 * headset camera SDK.
 *
 * Mirror of Swift `FrameInjectable`.
 *
 * ## Workflow
 *
 *   // 1. Connect to the session.
 *   session.Connect();
 *
 *   // 2. In your frame callback, inject each buffer:
 *   if (auto* sink = dynamic_cast<FrameSink*>(session.GetBackend())) {
 *       sink->InjectVideoFrame(pixels, width, height, format, timestamp_us);
 *   }
 *
 * ## Frame publication
 *
 * The first InjectVideoFrame() call creates a BufferCapturer-backed LiveKit
 * track and publishes it to the room. The track is published after the first
 * frame (not before) because LiveKit requires at least one captured frame to
 * resolve stream dimensions before the publish handshake can complete.
 */

#include <cstddef>
#include <cstdint>
#include <span>

namespace streamkit {

/// Pixel format of an injected video frame.
enum class PixelFormat {
    kI420,      ///< Planar YUV 4:2:0 — native WebRTC format, zero-copy path.
    kNV12,      ///< Semi-planar YUV 4:2:0 — common on camera hardware.
    kRGBA,      ///< Packed 8-bit RGBA.
    kBGRA,      ///< Packed 8-bit BGRA.
};

/// Implemented by backends that accept raw video frames from external sources.
///
/// LiveKitBackend will implement this once the stub is filled in.
class FrameSink {
public:
    virtual ~FrameSink() = default;

    /// Push a single video frame into the published video track.
    ///
    /// \param data         Pointer to the first byte of pixel data.
    /// \param width        Frame width in pixels.
    /// \param height       Frame height in pixels.
    /// \param stride       Row stride in bytes (may be > width * bytes_per_pixel).
    /// \param format       Pixel layout of `data`.
    /// \param timestamp_us Capture timestamp in microseconds (monotonic clock).
    ///
    /// A BufferCapturer-backed LocalVideoTrack is created and published on the
    /// first call. Subsequent calls deliver frames to the already-published track.
    ///
    /// Throws NotConnectedError if not connected.
    virtual void InjectVideoFrame(std::span<const std::byte> data,
                                  int width,
                                  int height,
                                  int stride,
                                  PixelFormat format,
                                  int64_t timestamp_us) = 0;
};

} // namespace streamkit
