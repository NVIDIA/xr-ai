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
#include "streamkit/StreamSession.h"

#include <cstddef>
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

    return 0;
}
