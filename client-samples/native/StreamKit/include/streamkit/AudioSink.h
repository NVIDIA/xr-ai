// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * StreamKit — AudioSink
 *
 * Optional mixin for backends that accept externally produced PCM audio
 * frames — e.g. from a platform mic HAL, a USB audio capture device, or a
 * loopback ring on an embedded host where the C++ SDK ships no built-in
 * mic capture path.
 *
 * Sibling of FrameSink. Mirror of Swift `AudioInjectable`.
 *
 * ## Workflow
 *
 *   // 1. Connect to the session.
 *   session.Connect();
 *
 *   // 2. Arm the audio track.
 *   session.StartAudio();
 *
 *   // 3. In your mic callback, push each buffer:
 *   if (auto* sink = dynamic_cast<AudioSink*>(session.GetBackend())) {
 *       sink->InjectAudioFrame(pcm, sample_rate, channels,
 *                              samples_per_channel, timestamp_us);
 *   }
 */

#include <cstddef>
#include <cstdint>
#include <span>

namespace streamkit {

/// Implemented by backends that accept raw PCM audio frames from external
/// sources. The built-in `LiveKitBackend` implements this; subclassing the
/// backend to reach a private `AudioSource` is no longer required.
class AudioSink {
public:
    virtual ~AudioSink() = default;

    /// Push a single PCM audio frame into the published audio track.
    ///
    /// \param pcm                  Interleaved signed-16-bit little-endian PCM
    ///                             samples, viewed as `std::span<const int16_t>`.
    ///                             `pcm.size()` is the **sample (element) count**
    ///                             and must equal `channels * samples_per_channel`.
    ///                             (Equivalently, the underlying byte length is
    ///                             `channels * samples_per_channel * sizeof(int16_t)`,
    ///                             but the span carries element count, not bytes.)
    /// \param sample_rate          Sample rate in Hz (e.g. 48000).
    /// \param channels             Channel count (1 = mono, 2 = stereo).
    /// \param samples_per_channel  Samples per channel in this frame
    ///                             (e.g. 480 for a 10 ms @ 48 kHz frame).
    /// \param timestamp_us         Capture timestamp in microseconds
    ///                             (monotonic clock).
    ///
    /// WebRTC expects ~10 ms frames; callers driving from a HAL that delivers
    /// longer chunks should slice before calling. Backends validate
    /// `pcm.size() == channels * samples_per_channel` and throw
    /// `std::invalid_argument` on mismatch.
    ///
    /// Throws `NotConnectedError` if not connected. Silently drops the frame
    /// when audio capture is not armed (e.g. after `StopAudio()` or before
    /// the first `StartAudio()`) — matches `FrameSink::InjectVideoFrame`'s
    /// behaviour so real-time capture threads don't see exception spam
    /// during stop/restart sequences.
    virtual void InjectAudioFrame(std::span<const std::int16_t> pcm,
                                  int sample_rate,
                                  int channels,
                                  int samples_per_channel,
                                  int64_t timestamp_us) = 0;
};

} // namespace streamkit
