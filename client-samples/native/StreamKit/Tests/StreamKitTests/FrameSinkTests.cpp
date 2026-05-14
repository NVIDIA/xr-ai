// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Verifies the contract of FrameSink's two InjectVideoFrame overloads:
 *
 *   1. A subclass that overrides only the span overload still works when a
 *      caller invokes the move overload — the default impl forwards to span.
 *   2. A subclass that overrides both overloads gets the move overload
 *      called directly when the caller passes an rvalue vector (no fallback).
 *
 * The forwarding behaviour is what makes the move overload strictly
 * additive — existing FrameSink implementations keep working without
 * recompilation.
 */

#include "test_assert.h"

#include "streamkit/FrameSink.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace {

struct SpanOnlySink : streamkit::FrameSink {
    int span_calls = 0;
    std::size_t last_span_size = 0;
    int last_width = 0;
    int last_height = 0;
    streamkit::PixelFormat last_format = streamkit::PixelFormat::kI420;
    int64_t last_ts = -1;

    void InjectVideoFrame(std::span<const std::byte> data,
                          int width, int height,
                          streamkit::PixelFormat format,
                          int64_t timestamp_us) override {
        ++span_calls;
        last_span_size = data.size();
        last_width = width;
        last_height = height;
        last_format = format;
        last_ts = timestamp_us;
    }
};

struct BothOverloadsSink : streamkit::FrameSink {
    int span_calls = 0;
    int move_calls = 0;
    std::size_t last_move_capacity = 0;

    void InjectVideoFrame(std::span<const std::byte> /*data*/,
                          int /*width*/, int /*height*/,
                          streamkit::PixelFormat /*format*/,
                          int64_t /*timestamp_us*/) override {
        ++span_calls;
    }

    void InjectVideoFrame(std::vector<std::uint8_t>&& data,
                          int /*width*/, int /*height*/,
                          streamkit::PixelFormat /*format*/,
                          int64_t /*timestamp_us*/) override {
        ++move_calls;
        last_move_capacity = data.capacity();
    }
};

}  // namespace

int main() {
    using streamkit::PixelFormat;

    // 1. Span-only subclass — the move overload should fall through the
    //    default impl to the span overload. The call goes through a
    //    FrameSink& reference because that's how StreamSession actually
    //    invokes the backend, and because C++'s name-hiding rule otherwise
    //    masks the inherited move overload at the derived static type.
    {
        SpanOnlySink sink;
        streamkit::FrameSink& base = sink;
        std::vector<std::uint8_t> buffer(1024, std::uint8_t{0xAB});
        base.InjectVideoFrame(std::move(buffer), 32, 16,
                              PixelFormat::kI420, 12345);
        SK_EXPECT_EQ(sink.span_calls, 1);
        SK_EXPECT_EQ(sink.last_span_size, std::size_t{1024});
        SK_EXPECT_EQ(sink.last_width, 32);
        SK_EXPECT_EQ(sink.last_height, 16);
        SK_EXPECT(sink.last_format == PixelFormat::kI420);
        SK_EXPECT_EQ(sink.last_ts, int64_t{12345});
    }

    // 2. Backend that overrides both — the move overload is called directly,
    //    span is not (no extra copy in the hot path).
    {
        BothOverloadsSink sink;
        std::vector<std::uint8_t> buffer(2048);
        sink.InjectVideoFrame(std::move(buffer), 64, 32,
                              PixelFormat::kNV12, 999);
        SK_EXPECT_EQ(sink.move_calls, 1);
        SK_EXPECT_EQ(sink.span_calls, 0);
        SK_EXPECT(sink.last_move_capacity >= std::size_t{2048});
    }

    // 3. Span overload on a both-overloads backend still routes through span
    //    (callers with read-only / shared buffers keep working unchanged).
    {
        BothOverloadsSink sink;
        std::vector<std::uint8_t> buffer(512);
        std::span<const std::byte> view(
            reinterpret_cast<const std::byte*>(buffer.data()), buffer.size());
        sink.InjectVideoFrame(view, 16, 8, PixelFormat::kRGBA, 1);
        SK_EXPECT_EQ(sink.span_calls, 1);
        SK_EXPECT_EQ(sink.move_calls, 0);
    }

    return 0;
}
