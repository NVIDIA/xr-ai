// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Demonstrates and verifies the "custom StreamingBackend for tests" pattern
 * documented in client-samples/native/README.md — that the host can replace
 * the LiveKit-backed transport with a mock to exercise the StreamSession
 * lifecycle deterministically and without a WebRTC stack.
 */

#include "test_assert.h"

#include "streamkit/Backends/StreamingBackend.h"
#include "streamkit/ConnectionState.h"
#include "streamkit/StreamSession.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct MockBackend : streamkit::StreamingBackend {
    int connect_calls = 0;
    int disconnect_calls = 0;
    int start_audio_calls = 0;
    int stop_audio_calls = 0;
    int start_camera_calls = 0;
    int stop_camera_calls = 0;
    std::vector<std::string> sent_topics;
    std::vector<std::string> sent_payloads;

    void Connect(const streamkit::SessionConfig&) override {
        ++connect_calls;
        if (on_connection_state_changed) {
            on_connection_state_changed(streamkit::ConnectionState::kConnecting);
            on_connection_state_changed(streamkit::ConnectionState::kConnected);
        }
    }
    void Disconnect() override {
        ++disconnect_calls;
        if (on_connection_state_changed) {
            on_connection_state_changed(streamkit::ConnectionState::kDisconnected);
        }
    }
    void StartAudio(const streamkit::AudioConfig&) override { ++start_audio_calls; }
    void StopAudio() override { ++stop_audio_calls; }
    void StartCamera(const streamkit::CameraConfig&) override { ++start_camera_calls; }
    void StopCamera() override { ++stop_camera_calls; }
    void Send(std::span<const std::byte> data, bool, std::string_view topic) override {
        sent_topics.emplace_back(topic);
        sent_payloads.emplace_back(reinterpret_cast<const char*>(data.data()), data.size());
    }

    // Helpers for tests to drive the event hooks the backend would normally
    // fire from its event loop.
    void fire_data(std::string_view topic, std::string_view payload) {
        if (!on_data_received) return;
        std::span<const std::byte> bytes(
            reinterpret_cast<const std::byte*>(payload.data()), payload.size());
        on_data_received(topic, bytes);
    }
    void fire_agent_status(std::string_view status) {
        if (on_agent_status) on_agent_status(status);
    }
};

}  // namespace

int main() {
    auto backend = std::make_unique<MockBackend>();
    auto* raw = backend.get();
    streamkit::StreamSession session(std::move(backend));

    // ── Wire session-level callbacks ───────────────────────────────────────
    int state_changes = 0;
    streamkit::ConnectionState last_state = streamkit::ConnectionState::kDisconnected;
    session.on_connection_state_changed = [&](streamkit::ConnectionState s) {
        ++state_changes;
        last_state = s;
    };

    int data_calls = 0;
    std::string last_topic;
    std::string last_payload;
    session.on_data_received = [&](std::string_view topic,
                                   std::span<const std::byte> data) {
        ++data_calls;
        last_topic = std::string(topic);
        last_payload.assign(reinterpret_cast<const char*>(data.data()), data.size());
    };

    int agent_calls = 0;
    std::string last_agent_status;
    session.on_agent_status = [&](std::string_view status) {
        ++agent_calls;
        last_agent_status = std::string(status);
    };

    // ── Connect lifecycle ──────────────────────────────────────────────────
    SK_EXPECT(session.connection_state() == streamkit::ConnectionState::kDisconnected);
    session.Connect(streamkit::SessionConfig{"test-identity"});
    SK_EXPECT_EQ(raw->connect_calls, 1);
    SK_EXPECT_EQ(state_changes, 2);
    SK_EXPECT(last_state == streamkit::ConnectionState::kConnected);
    SK_EXPECT(session.connection_state() == streamkit::ConnectionState::kConnected);

    // ── Media independent of connection ────────────────────────────────────
    session.StartAudio();
    session.StartCamera();
    SK_EXPECT_EQ(raw->start_audio_calls, 1);
    SK_EXPECT_EQ(raw->start_camera_calls, 1);

    // ── Send + data delivery ───────────────────────────────────────────────
    const std::string msg = "hello";
    std::vector<std::byte> payload(msg.size());
    std::memcpy(payload.data(), msg.data(), msg.size());
    session.Send(payload, /*reliable=*/true, "test.topic");
    SK_EXPECT_EQ(raw->sent_topics.size(), std::size_t{1});
    SK_EXPECT_EQ(raw->sent_topics[0], std::string("test.topic"));
    SK_EXPECT_EQ(raw->sent_payloads[0], std::string("hello"));

    raw->fire_data("incoming.topic", "world");
    SK_EXPECT_EQ(data_calls, 1);
    SK_EXPECT_EQ(last_topic, std::string("incoming.topic"));
    SK_EXPECT_EQ(last_payload, std::string("world"));

    raw->fire_agent_status("processing");
    SK_EXPECT_EQ(agent_calls, 1);
    SK_EXPECT_EQ(last_agent_status, std::string("processing"));

    // ── Stop media + disconnect ────────────────────────────────────────────
    session.StopAudio();
    session.StopCamera();
    SK_EXPECT_EQ(raw->stop_audio_calls, 1);
    SK_EXPECT_EQ(raw->stop_camera_calls, 1);

    session.Disconnect();
    SK_EXPECT_EQ(raw->disconnect_calls, 1);
    SK_EXPECT(last_state == streamkit::ConnectionState::kDisconnected);
    SK_EXPECT(session.connection_state() == streamkit::ConnectionState::kDisconnected);

    return 0;
}
