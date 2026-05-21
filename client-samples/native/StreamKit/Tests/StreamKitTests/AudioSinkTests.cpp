// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Verifies the AudioSink contract:
 *
 *   1. A subclass that implements InjectAudioFrame receives the call with
 *      every parameter (rate / channels / samples_per_channel / timestamp)
 *      delivered verbatim.
 *   2. The interface is dispatchable through an `AudioSink&` reference —
 *      the typical path callers take via `dynamic_cast<AudioSink*>` on
 *      `session.GetBackend()`. Catches regressions where AudioSink's vtable
 *      gets sliced by a missing-virtual mistake.
 *
 * The AudioSink interface intentionally has a single pure-virtual entry
 * point (no zero-copy second overload — the i16 PCM volume per frame is
 * ~960 bytes for 10 ms @ 48 kHz mono, well below the threshold that made
 * the FrameSink move-overload worth its asymmetry).
 */

#include "test_assert.h"

#include "streamkit/AudioSink.h"

#include <cstdint>
#include <span>
#include <vector>

namespace {

struct RecordingAudioSink : streamkit::AudioSink {
    int calls = 0;
    int last_sample_rate = 0;
    int last_channels = 0;
    int last_samples_per_channel = 0;
    int64_t last_ts = -1;
    std::size_t last_sample_count = 0;
    std::int16_t last_first_sample = 0;

    void InjectAudioFrame(std::span<const std::int16_t> pcm,
                          int sample_rate,
                          int channels,
                          int samples_per_channel,
                          int64_t timestamp_us) override {
        ++calls;
        last_sample_rate = sample_rate;
        last_channels = channels;
        last_samples_per_channel = samples_per_channel;
        last_ts = timestamp_us;
        last_sample_count = pcm.size();
        last_first_sample = pcm.empty() ? std::int16_t{0} : pcm.front();
    }
};

}  // namespace

int main() {
    // 1. Direct dispatch via the concrete subclass.
    {
        RecordingAudioSink sink;
        std::vector<std::int16_t> pcm(480, std::int16_t{1234});  // 10 ms @ 48 kHz mono
        sink.InjectAudioFrame(pcm, 48000, 1, 480, 5000);
        SK_EXPECT_EQ(sink.calls, 1);
        SK_EXPECT_EQ(sink.last_sample_rate, 48000);
        SK_EXPECT_EQ(sink.last_channels, 1);
        SK_EXPECT_EQ(sink.last_samples_per_channel, 480);
        SK_EXPECT_EQ(sink.last_ts, int64_t{5000});
        SK_EXPECT_EQ(sink.last_sample_count, std::size_t{480});
        SK_EXPECT_EQ(sink.last_first_sample, std::int16_t{1234});
    }

    // 2. Dispatch through an AudioSink& reference — the path
    //    `dynamic_cast<AudioSink*>(session.GetBackend())` callers use.
    {
        RecordingAudioSink sink;
        streamkit::AudioSink& base = sink;
        std::vector<std::int16_t> pcm(960, std::int16_t{-7});  // 10 ms @ 48 kHz stereo
        base.InjectAudioFrame(pcm, 48000, 2, 480, 99);
        SK_EXPECT_EQ(sink.calls, 1);
        SK_EXPECT_EQ(sink.last_channels, 2);
        SK_EXPECT_EQ(sink.last_samples_per_channel, 480);
        SK_EXPECT_EQ(sink.last_sample_count, std::size_t{960});
        SK_EXPECT_EQ(sink.last_first_sample, std::int16_t{-7});
    }

    return 0;
}
