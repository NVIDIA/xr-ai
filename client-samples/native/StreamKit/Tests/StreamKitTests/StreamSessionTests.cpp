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

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string BytesToString(std::span<const std::byte> data) {
    std::string payload(data.size(), '\0');
    std::ranges::transform(data, payload.begin(), [](std::byte byte) {
        return static_cast<char>(std::to_integer<unsigned char>(byte));
    });
    return payload;
}

struct MockBackend : streamkit::StreamingBackend {
    int connect_calls = 0;
    int disconnect_calls = 0;
    int start_audio_calls = 0;
    int stop_audio_calls = 0;
    int start_camera_calls = 0;
    int stop_camera_calls = 0;
    streamkit::CameraConfig last_camera_config;
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
    void StartCamera(const streamkit::CameraConfig& config) override {
        ++start_camera_calls;
        last_camera_config = config;
    }
    void StopCamera() override { ++stop_camera_calls; }
    void Send(std::span<const std::byte> data, bool, std::string_view topic) override {
        sent_topics.emplace_back(topic);
        sent_payloads.emplace_back(BytesToString(data));
    }

    // Helpers for tests to drive the event hooks the backend would normally
    // fire from its event loop.
    void fire_data(std::string_view topic, std::string_view payload) const {
        if (!on_data_received) return;
        auto bytes = std::as_bytes(std::span<const char>(payload.data(), payload.size()));
        on_data_received(topic, bytes);
    }
    void fire_agent_status(std::string_view status) const {
        if (on_agent_status) on_agent_status(status);
    }
};

}  // namespace

int main() {
    using streamkit::test::Expect;
    using streamkit::test::ExpectEq;

    auto backend = std::make_unique<MockBackend>();
    auto* raw = backend.get();
    streamkit::StreamSession session(std::move(backend));

    // ── Wire session-level callbacks ───────────────────────────────────────
    int state_changes = 0;
    streamkit::ConnectionState last_state = streamkit::ConnectionState::kDisconnected;
    session.on_connection_state_changed =
        [&state_changes, &last_state](streamkit::ConnectionState s) {
        ++state_changes;
        last_state = s;
    };

    int data_calls = 0;
    std::string last_topic;
    std::string last_payload;
    session.on_data_received =
        [&data_calls, &last_topic, &last_payload](
            std::string_view topic,
            std::span<const std::byte> data) {
        ++data_calls;
        last_topic = std::string(topic);
        last_payload = BytesToString(data);
    };

    int agent_calls = 0;
    std::string last_agent_status;
    session.on_agent_status = [&agent_calls, &last_agent_status](std::string_view status) {
        ++agent_calls;
        last_agent_status = std::string(status);
    };

    // ── Connect lifecycle ──────────────────────────────────────────────────
    Expect(session.connection_state() == streamkit::ConnectionState::kDisconnected);
    session.Connect(streamkit::SessionConfig{"test-identity"});
    ExpectEq(raw->connect_calls, 1);
    ExpectEq(state_changes, 2);
    Expect(last_state == streamkit::ConnectionState::kConnected);
    Expect(session.connection_state() == streamkit::ConnectionState::kConnected);

    // ── Media independent of connection ────────────────────────────────────
    session.StartAudio();
    session.StartCamera();
    ExpectEq(raw->start_audio_calls, 1);
    ExpectEq(raw->start_camera_calls, 1);

    streamkit::CameraConfig encoded_camera;
    encoded_camera.encoding = streamkit::CameraEncodingConfig{
        .max_bitrate_bps = 2'500'000,
        .max_framerate = 21.0,
        .simulcast = false,
    };
    session.StartCamera(encoded_camera);
    ExpectEq(raw->start_camera_calls, 2);
    Expect(raw->last_camera_config.encoding.has_value());
    ExpectEq(raw->last_camera_config.encoding->max_bitrate_bps,
             std::uint64_t{2'500'000});
    ExpectEq(raw->last_camera_config.encoding->max_framerate, 21.0);
    Expect(raw->last_camera_config.encoding->simulcast.has_value());
    Expect(!*raw->last_camera_config.encoding->simulcast);

    // ── Send + data delivery ───────────────────────────────────────────────
    const std::string msg = "hello";
    auto payload = std::as_bytes(std::span<const char>(msg.data(), msg.size()));
    session.Send(payload, /*reliable=*/true, "test.topic");
    ExpectEq(raw->sent_topics.size(), std::size_t{1});
    ExpectEq(raw->sent_topics[0], std::string("test.topic"));
    ExpectEq(raw->sent_payloads[0], std::string("hello"));

    raw->fire_data("incoming.topic", "world");
    ExpectEq(data_calls, 1);
    ExpectEq(last_topic, std::string("incoming.topic"));
    ExpectEq(last_payload, std::string("world"));

    raw->fire_agent_status("processing");
    ExpectEq(agent_calls, 1);
    ExpectEq(last_agent_status, std::string("processing"));

    // ── Stop media + disconnect ────────────────────────────────────────────
    session.StopAudio();
    session.StopCamera();
    ExpectEq(raw->stop_audio_calls, 1);
    ExpectEq(raw->stop_camera_calls, 1);

    session.Disconnect();
    ExpectEq(raw->disconnect_calls, 1);
    Expect(last_state == streamkit::ConnectionState::kDisconnected);
    Expect(session.connection_state() == streamkit::ConnectionState::kDisconnected);

    return 0;
}
