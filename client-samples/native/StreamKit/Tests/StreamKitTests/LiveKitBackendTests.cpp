// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Regression tests for LiveKitBackend's state-machine behaviour, exercised
 * through the stub-mode build (no LiveKit SDK linked). Stub mode synthesises
 * the connect / disconnect transitions deterministically, which is exactly
 * what we want for verifying the FireStateChanged dedupe — that consumers
 * never see a duplicate kConnected or a spurious kDisconnected on first
 * Connect.
 */

#include "test_assert.h"

#include "streamkit/Config/BackendConfiguration.h"
#include "streamkit/ConnectionState.h"
#include "streamkit/FrameSink.h"
#include "streamkit/StreamSession.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
    using streamkit::ConnectionState;

    streamkit::LiveKitConfig lk;
    lk.host  = "localhost";
    lk.token = "stub-mode-token";

    streamkit::StreamSession session{
        streamkit::BackendConfiguration{lk}};

    std::vector<ConnectionState> states;
    session.on_connection_state_changed = [&](ConnectionState s) {
        states.push_back(s);
    };

    // ── First Connect — must NOT emit a spurious initial kDisconnected
    //    from TearDown(), and must emit exactly one kConnected. ────────────
    session.Connect();
    SK_EXPECT_EQ(states.size(), std::size_t{2});
    SK_EXPECT(states[0] == ConnectionState::kConnecting);
    SK_EXPECT(states[1] == ConnectionState::kConnected);

    // ── Disconnect — exactly one kDisconnected. ──────────────────────────
    session.Disconnect();
    SK_EXPECT_EQ(states.size(), std::size_t{3});
    SK_EXPECT(states[2] == ConnectionState::kDisconnected);

    // ── Reconnect — still no spurious leading kDisconnected. The
    //    TearDown() at the top of Connect sees last_fired_state_ ==
    //    kDisconnected and skips. ────────────────────────────────────────
    session.Connect();
    SK_EXPECT_EQ(states.size(), std::size_t{5});
    SK_EXPECT(states[3] == ConnectionState::kConnecting);
    SK_EXPECT(states[4] == ConnectionState::kConnected);

    // ── Idempotent disconnect — second Disconnect after the first is a
    //    no-op for the callback. ──────────────────────────────────────────
    session.Disconnect();
    SK_EXPECT_EQ(states.size(), std::size_t{6});
    SK_EXPECT(states[5] == ConnectionState::kDisconnected);

    session.Disconnect();
    SK_EXPECT_EQ(states.size(), std::size_t{6});  // unchanged

    // ── Buffer-size validation in the move overload — packed contract.
    //    Reconnect, arm the camera, then push a deliberately-wrong-size
    //    buffer. LiveKitBackend throws std::invalid_argument. Validation
    //    runs in stub mode because it sits before the
    //    STREAMKIT_HAVE_LIVEKIT-gated SDK calls.
    session.Connect();
    session.StartCamera();
    auto* sink = dynamic_cast<streamkit::FrameSink*>(session.GetBackend());
    SK_EXPECT(sink != nullptr);

    // 16×16 I420 packed = 16*16 + 2 * (8*8) = 384 bytes.
    // Pass 100 bytes to force the mismatch.
    bool threw_invalid = false;
    try {
        std::vector<std::uint8_t> too_small(100);
        sink->InjectVideoFrame(std::move(too_small), 16, 16,
                               streamkit::PixelFormat::kI420, 0);
    } catch (const std::invalid_argument&) {
        threw_invalid = true;
    }
    SK_EXPECT(threw_invalid);

    // Correct size goes through (stub-mode no-op past the validation).
    std::vector<std::uint8_t> packed(384);
    sink->InjectVideoFrame(std::move(packed), 16, 16,
                           streamkit::PixelFormat::kI420, 0);

    session.Disconnect();

    return 0;
}
