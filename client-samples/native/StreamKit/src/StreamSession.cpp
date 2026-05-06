// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "streamkit/StreamSession.h"
#include "streamkit/Backends/LiveKit/LiveKitBackend.h"
#include "streamkit/Config/BackendConfiguration.h"

#include <stdexcept>
#include <utility>

namespace streamkit {

// ─────────────────────────────────────────────────────────────────────────────
// MakeBackend (BackendConfiguration.h factory)
// ─────────────────────────────────────────────────────────────────────────────

std::unique_ptr<StreamingBackend> MakeBackend(const BackendConfiguration& config) {
    return std::visit([](auto&& cfg) -> std::unique_ptr<StreamingBackend> {
        using T = std::decay_t<decltype(cfg)>;
        if constexpr (std::is_same_v<T, LiveKitConfig>) {
            return std::make_unique<LiveKitBackend>(cfg);
        }
    }, config);
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamSession
// ─────────────────────────────────────────────────────────────────────────────

StreamSession::StreamSession(const BackendConfiguration& config)
    : backend_(MakeBackend(config)) {
    WireCallbacks();
}

StreamSession::StreamSession(std::unique_ptr<StreamingBackend> backend)
    : backend_(std::move(backend)) {
    WireCallbacks();
}

// ── Connection ────────────────────────────────────────────────────────────────

void StreamSession::Connect(const SessionConfig& config) {
    backend_->Connect(config);
}

void StreamSession::Disconnect() {
    backend_->Disconnect();
    // The backend fires kDisconnected via on_connection_state_changed;
    // agent_status is implicitly stale once disconnected.
}

// ── Audio ─────────────────────────────────────────────────────────────────────

void StreamSession::StartAudio(const AudioConfig& config) {
    backend_->StartAudio(config);
}

void StreamSession::StopAudio() {
    backend_->StopAudio();
}

// ── Camera ────────────────────────────────────────────────────────────────────

void StreamSession::StartCamera(const CameraConfig& config) {
    backend_->StartCamera(config);
}

void StreamSession::StopCamera() {
    backend_->StopCamera();
}

// ── Data channel ──────────────────────────────────────────────────────────────

void StreamSession::Send(std::span<const std::byte> data,
                         bool reliable,
                         std::string_view topic) {
    backend_->Send(data, reliable, topic);
}

// ── Private ───────────────────────────────────────────────────────────────────

/// Subscribe to the backend's event hooks and forward them to this session's
/// own public callbacks. Called once immediately after the backend is set.
void StreamSession::WireCallbacks() {
    backend_->on_connection_state_changed = [this](ConnectionState state) {
        connection_state_ = state;
        if (on_connection_state_changed) {
            on_connection_state_changed(state);
        }
    };

    backend_->on_data_received = [this](std::string_view topic,
                                        std::span<const std::byte> data) {
        if (on_data_received) {
            on_data_received(topic, data);
        }
    };

    backend_->on_agent_status = [this](std::string_view status) {
        if (on_agent_status) {
            on_agent_status(status);
        }
    };
}

} // namespace streamkit
