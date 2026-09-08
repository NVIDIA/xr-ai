// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "streamkit/StreamSession.h"
#include "streamkit/Backends/LiveKit/LiveKitBackend.h"
#include "streamkit/Config/BackendConfiguration.h"

#include <charconv>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace streamkit {

namespace {

std::string JsonString(std::string_view json, std::string_view key) {
    const auto marker = std::string{"\""} + std::string(key) + "\"";
    auto position = json.find(marker);
    if (position == std::string_view::npos) return {};
    position = json.find(':', position + marker.size());
    position = json.find('"', position);
    if (position == std::string_view::npos) return {};
    const auto end = json.find('"', position + 1);
    if (end == std::string_view::npos) return {};
    return std::string(json.substr(position + 1, end - position - 1));
}

std::int64_t JsonInteger(std::string_view json, std::string_view key) {
    const auto marker = std::string{"\""} + std::string(key) + "\"";
    auto position = json.find(marker);
    if (position == std::string_view::npos) return 0;
    position = json.find(':', position + marker.size());
    if (position == std::string_view::npos) return 0;
    ++position;
    while (position < json.size() && json[position] == ' ') ++position;
    std::int64_t value = 0;
    std::from_chars(json.data() + position, json.data() + json.size(), value);
    return value;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// MakeBackend (BackendConfiguration.h factory)
// ─────────────────────────────────────────────────────────────────────────────

std::unique_ptr<StreamingBackend> MakeBackend(const BackendConfiguration& config) {
    return std::visit([]<typename T>(const T& cfg) -> std::unique_ptr<StreamingBackend> {
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

void StreamSession::SendImage(const CapturedImage& image, std::string_view request_id) {
    backend_->SendImage(image.data, request_id, image.mime_type, image.name);
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
        if (topic == "camera.capture.request") {
            if (!on_image_capture_requested) return;
            const auto payload = std::string_view(
                reinterpret_cast<const char*>(data.data()), data.size());
            if (JsonInteger(payload, "version") != 1) return;
            ImageCaptureRequest request{
                .request_id = JsonString(payload, "request_id"),
                .timeout_ms = JsonInteger(payload, "timeout_ms"),
            };
            if (request.request_id.empty()) return;
            try {
                auto image = on_image_capture_requested(request);
                if (!image.data.empty() &&
                    (image.mime_type == "image/jpeg" ||
                     image.mime_type == "image/png" ||
                     image.mime_type == "image/webp")) {
                    SendImage(image, request.request_id);
                }
            } catch (...) {
                // Capability errors are request-local and must not escape the SDK callback.
            }
            return;
        }
        if (topic == "camera.capture.cancel") return;
        if (on_data_received) {
            on_data_received(topic, data);
        }
    };

    backend_->on_agent_status = [this](std::string_view status) {
        if (on_agent_status) {
            on_agent_status(status);
        }
    };

    backend_->on_network_metrics = [this](const NetworkMetrics& metrics) {
        if (on_network_metrics) {
            on_network_metrics(metrics);
        }
    };
}

} // namespace streamkit
