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

#include "streamkit/Backends/LiveKit/LiveKitBackend.h"
#include "streamkit/Config/BackendConfiguration.h"
#include "streamkit/ConnectionState.h"
#include "streamkit/FrameSink.h"
#include "streamkit/StreamSession.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

int main() {
    using streamkit::ConnectionState;
    using streamkit::test::Expect;
    using streamkit::test::ExpectEq;

    streamkit::LiveKitConfig lk;
    lk.host  = "localhost";
    lk.token = "stub-mode-token";

    streamkit::StreamSession session{
        streamkit::BackendConfiguration{lk}};
    const auto* livekit_backend =
        dynamic_cast<streamkit::LiveKitBackend*>(session.GetBackend());
    Expect(livekit_backend != nullptr);
    Expect(!livekit_backend->GetRoom());

    std::vector<ConnectionState> states;
    session.on_connection_state_changed = [&states](ConnectionState s) {
        states.push_back(s);
    };

    // ── First Connect — must NOT emit a spurious initial kDisconnected
    //    from TearDown(), and must emit exactly one kConnected. ────────────
    session.Connect();
    ExpectEq(states.size(), std::size_t{2});
    Expect(states[0] == ConnectionState::kConnecting);
    Expect(states[1] == ConnectionState::kConnected);
    Expect(!livekit_backend->GetRoom());

    // ── Disconnect — exactly one kDisconnected. ──────────────────────────
    session.Disconnect();
    ExpectEq(states.size(), std::size_t{3});
    Expect(states[2] == ConnectionState::kDisconnected);
    Expect(!livekit_backend->GetRoom());

    // ── Reconnect — still no spurious leading kDisconnected. The
    //    TearDown() at the top of Connect sees last_fired_state_ ==
    //    kDisconnected and skips. ────────────────────────────────────────
    session.Connect();
    ExpectEq(states.size(), std::size_t{5});
    Expect(states[3] == ConnectionState::kConnecting);
    Expect(states[4] == ConnectionState::kConnected);

    // ── Idempotent disconnect — second Disconnect after the first is a
    //    no-op for the callback. ──────────────────────────────────────────
    session.Disconnect();
    ExpectEq(states.size(), std::size_t{6});
    Expect(states[5] == ConnectionState::kDisconnected);

    session.Disconnect();
    ExpectEq(states.size(), std::size_t{6});  // unchanged

    // ── Buffer-size validation in the move overload — packed contract.
    //    Reconnect, arm the camera, then push a deliberately-wrong-size
    //    buffer. LiveKitBackend throws std::invalid_argument. Validation
    //    runs in stub mode because it sits before the
    //    STREAMKIT_HAVE_LIVEKIT-gated SDK calls.
    session.Connect();
    session.StartCamera();
    auto* sink = dynamic_cast<streamkit::FrameSink*>(session.GetBackend());
    Expect(sink != nullptr);

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
    Expect(threw_invalid);

    // Correct size goes through (stub-mode no-op past the validation).
    std::vector<std::uint8_t> packed(384);
    sink->InjectVideoFrame(std::move(packed), 16, 16,
                           streamkit::PixelFormat::kI420, 0);

    session.Disconnect();

    // A telemetry callback may disconnect while an app-thread disconnect is
    // already joining that worker. Both calls must share ownership safely and
    // neither may wait on the other while holding the ownership mutex.
    streamkit::LiveKitBackend callback_backend{lk};
    std::promise<void> callback_started;
    auto callback_started_future = callback_started.get_future();
    std::promise<void> release_callback;
    auto release_callback_future = release_callback.get_future();
    std::promise<void> callback_finished;
    auto callback_future = callback_finished.get_future();
    callback_backend.on_network_metrics = [&](const streamkit::NetworkMetrics&) {
        callback_started.set_value();
        release_callback_future.wait();
        callback_backend.Disconnect();
        callback_finished.set_value();
    };
    callback_backend.Connect(streamkit::SessionConfig::Default());
    Expect(callback_started_future.wait_for(std::chrono::seconds(2)) ==
           std::future_status::ready);

    std::thread app_disconnect([&callback_backend]() {
        callback_backend.Disconnect();
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    release_callback.set_value();
    Expect(callback_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    app_disconnect.join();

    return 0;
}
