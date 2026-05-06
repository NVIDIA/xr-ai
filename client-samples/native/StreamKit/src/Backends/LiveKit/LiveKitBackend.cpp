// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * StreamKit — LiveKitBackend (stub)
 *
 * This file is intentionally left unimplemented. Each method has TODO comments
 * that describe exactly what needs to be called in the LiveKit C++ SDK / livekit-ffi.
 *
 * ## Dependency
 *
 * LiveKit provides two C/C++ integration paths:
 *
 *   1. livekit-ffi  — Rust-based, cross-platform, exposes a C ABI via a
 *                     generated header (livekit_ffi.h). Suitable for most
 *                     native targets.
 *                     Repo: https://github.com/livekit/rust-sdks
 *
 *   2. WebRTC native — Build LiveKit's patched libwebrtc directly and use the
 *                     room-level C++ API. Higher effort, more control.
 *
 * See the README for CMake integration instructions for both paths.
 *
 * ## How to complete this file
 *
 *   1. Add livekit-ffi (or your chosen SDK) via CMake FetchContent — see the
 *      placeholder block in the root CMakeLists.txt.
 *   2. Replace `#include "livekit_ffi_stub.h"` with the real SDK header.
 *   3. Replace `LKRoom*` in LiveKitBackend.h with the real room type.
 *   4. Fill in each TODO block below, following the pattern in the Swift and
 *      Kotlin LiveKitBackend implementations for reference.
 */

#include "streamkit/Backends/LiveKit/LiveKitBackend.h"
#include "streamkit/StreamError.h"

#include <stdexcept>
#include <string>

// TODO: Replace with the real livekit-ffi or SDK header once the dependency
//       is added to CMakeLists.txt.
//
// #include <livekit/livekit_ffi.h>

namespace streamkit {

// ─────────────────────────────────────────────────────────────────────────────
// Construction / destruction
// ─────────────────────────────────────────────────────────────────────────────

LiveKitBackend::LiveKitBackend(const LiveKitConfig& config)
    : config_(config) {}

LiveKitBackend::~LiveKitBackend() {
    TearDown();
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection
// ─────────────────────────────────────────────────────────────────────────────

void LiveKitBackend::Connect(const SessionConfig& session_config) {
    session_config_ = session_config;
    TearDown();  // clean up any previous session

    // ── Validate host ─────────────────────────────────────────────────────
    if (config_.host.empty()) {
        throw InvalidHostError(config_.host);
    }

    // ── Build WebSocket URL ───────────────────────────────────────────────
    const std::string scheme = config_.secure ? "wss" : "ws";
    const std::string ws_url = scheme + "://" + config_.host + ":"
                               + std::to_string(config_.port);

    // ── Acquire JWT ───────────────────────────────────────────────────────
    std::string token;
    if (config_.token.has_value() && !config_.token->empty()) {
        token = *config_.token;
    } else if (config_.token_url.has_value() && !config_.token_url->empty()) {
        token = FetchToken(*config_.token_url, session_config_.identity);
    } else {
        throw MissingTokenError{};
    }

    // ── Create Room and register event listeners ──────────────────────────
    //
    // TODO: Create the LiveKit room and register callbacks.
    //
    // Using livekit-ffi (C API):
    //
    //   LKRoomOptions opts = lk_room_options_default();
    //   room_ = lk_room_create(&opts);
    //
    //   lk_room_set_callback(room_, LK_EVENT_CONNECTION_STATE_CHANGED,
    //       [](LKConnectionState state, void* ctx) {
    //           static_cast<LiveKitBackend*>(ctx)->HandleConnectionStateChange(state);
    //       }, this);
    //
    //   lk_room_set_callback(room_, LK_EVENT_DATA_RECEIVED,
    //       [](const uint8_t* data, size_t len, const char* topic, void* ctx) {
    //           auto* self = static_cast<LiveKitBackend*>(ctx);
    //           self->HandleDataReceived(topic,
    //               std::span(reinterpret_cast<const std::byte*>(data), len));
    //       }, this);

    // ── Fire CONNECTING before the async handshake ────────────────────────
    if (on_connection_state_changed) {
        on_connection_state_changed(ConnectionState::kConnecting);
    }

    // ── Connect ───────────────────────────────────────────────────────────
    //
    // TODO: Call the SDK's connect method and block until connected or throw.
    //
    // Using livekit-ffi:
    //
    //   LKConnectOptions conn_opts = lk_connect_options_default();
    //   conn_opts.timeout_ms = 5000;
    //
    //   LKError err = lk_room_connect(room_, ws_url.c_str(), token.c_str(), &conn_opts);
    //   if (err.code != LK_OK) {
    //       throw StreamError(err.message);
    //   }
    //
    // For async SDKs, block with a promise/future or a condition variable:
    //
    //   std::promise<void> connected;
    //   // ... wire connection callback to call connected.set_value() ...
    //   connected.get_future().get();  // blocks until kConnected or throws

    (void)ws_url;
    (void)token;

    is_connected_.store(true);

    if (on_connection_state_changed) {
        on_connection_state_changed(ConnectionState::kConnected);
    }
}

void LiveKitBackend::Disconnect() {
    TearDown();
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio
// ─────────────────────────────────────────────────────────────────────────────

void LiveKitBackend::StartAudio(const AudioConfig& config) {
    if (!is_connected_.load()) {
        throw NotConnectedError{};
    }

    // ── Map MicrophoneMode to WebRTC AudioOptions ─────────────────────────
    //
    // TODO: Create a local audio track with the appropriate DSP settings and
    //       publish it to the room.
    //
    // Using livekit-ffi:
    //
    //   LKAudioOptions audio_opts = lk_audio_options_default();
    //
    //   switch (config.mode) {
    //     case AudioConfig::MicrophoneMode::kVoiceProcessing:
    //       // Hardware AEC: disable WebRTC's own AEC/AGC/NS to avoid double-processing.
    //       audio_opts.echo_cancellation  = false;
    //       audio_opts.auto_gain_control  = false;
    //       audio_opts.noise_suppression  = false;
    //       break;
    //     case AudioConfig::MicrophoneMode::kSoftwareProcessing:
    //       audio_opts.echo_cancellation  = true;
    //       audio_opts.auto_gain_control  = true;
    //       audio_opts.noise_suppression  = true;
    //       audio_opts.highpass_filter    = config.highpass_filter;
    //       break;
    //     case AudioConfig::MicrophoneMode::kRaw:
    //     case AudioConfig::MicrophoneMode::kDisabled:
    //       audio_opts.echo_cancellation  = false;
    //       audio_opts.auto_gain_control  = false;
    //       audio_opts.noise_suppression  = false;
    //       break;
    //   }
    //
    //   if (config.mode != AudioConfig::MicrophoneMode::kDisabled) {
    //       LKError err = lk_local_participant_set_microphone_enabled(
    //           lk_room_local_participant(room_), true, &audio_opts);
    //       if (err.code != LK_OK) throw StreamError(err.message);
    //   }

    (void)config;
}

void LiveKitBackend::StopAudio() {
    // TODO: Disable the microphone track.
    //
    // Using livekit-ffi:
    //   lk_local_participant_set_microphone_enabled(
    //       lk_room_local_participant(room_), false, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera
// ─────────────────────────────────────────────────────────────────────────────

void LiveKitBackend::StartCamera(const CameraConfig& config) {
    if (!is_connected_.load()) {
        throw CameraRequiresConnectionError{};
    }

    StopCamera();  // unpublish any existing camera track first

    // ── Create and publish a local video track ────────────────────────────
    //
    // TODO: Create a video track for the requested device and publish it.
    //
    // Using livekit-ffi:
    //
    //   LKVideoCaptureOptions cap_opts = lk_video_capture_options_default();
    //   if (config.device_id.has_value()) {
    //       cap_opts.device_id = config.device_id->c_str();
    //   } else {
    //       cap_opts.facing_mode = (config.facing == CameraConfig::Facing::kFront)
    //                               ? LK_FACING_FRONT : LK_FACING_BACK;
    //   }
    //
    //   LKLocalVideoTrack* track = lk_local_video_track_create_camera(&cap_opts);
    //   LKError err = lk_local_participant_publish_video_track(
    //       lk_room_local_participant(room_), track, nullptr);
    //   if (err.code != LK_OK) throw StreamError(err.message);
    //
    // For frame injection from an external source (see FrameSink.h), create a
    // BufferCapturer-backed track instead:
    //
    //   LKLocalVideoTrack* track = lk_local_video_track_create_buffer_capturer("camera");
    //   // Store the track handle so InjectVideoFrame() can push frames into it.

    (void)config;
}

void LiveKitBackend::StopCamera() {
    // TODO: Unpublish and release the camera / buffer track.
    //
    // Using livekit-ffi:
    //   if (camera_track_) {
    //       lk_local_participant_unpublish_video_track(
    //           lk_room_local_participant(room_), camera_track_);
    //       lk_local_video_track_release(camera_track_);
    //       camera_track_ = nullptr;
    //   }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data channel
// ─────────────────────────────────────────────────────────────────────────────

void LiveKitBackend::Send(std::span<const std::byte> data,
                          bool reliable,
                          std::string_view topic) {
    if (!is_connected_.load()) {
        throw NotConnectedError{};
    }
    if (topic == kAgentStatusTopic) {
        throw std::invalid_argument(
            "topic '" + std::string(topic) + "' is reserved for internal SDK use");
    }

    // TODO: Publish data via the LiveKit data channel.
    //
    // Using livekit-ffi:
    //
    //   LKDataPublishOptions opts{};
    //   opts.reliable = reliable;
    //   opts.topic    = topic.empty() ? nullptr : std::string(topic).c_str();
    //
    //   LKError err = lk_local_participant_publish_data(
    //       lk_room_local_participant(room_),
    //       reinterpret_cast<const uint8_t*>(data.data()),
    //       data.size(),
    //       &opts);
    //   if (err.code != LK_OK) throw StreamError(err.message);

    (void)data;
    (void)reliable;
}

// ─────────────────────────────────────────────────────────────────────────────
// Event handlers
// ─────────────────────────────────────────────────────────────────────────────

void LiveKitBackend::HandleConnectionStateChange(int lk_state) {
    // TODO: Map the SDK's connection-state type to ConnectionState and fire
    //       on_connection_state_changed.
    //
    // Using livekit-ffi (LKConnectionState enum):
    //
    //   ConnectionState sk_state;
    //   switch (static_cast<LKConnectionState>(lk_state)) {
    //     case LK_CONNECTION_STATE_CONNECTED:    sk_state = ConnectionState::kConnected;    break;
    //     case LK_CONNECTION_STATE_CONNECTING:   sk_state = ConnectionState::kConnecting;   break;
    //     case LK_CONNECTION_STATE_RECONNECTING: sk_state = ConnectionState::kReconnecting; break;
    //     default:                               sk_state = ConnectionState::kDisconnected; break;
    //   }
    //   is_connected_.store(sk_state == ConnectionState::kConnected);
    //   if (on_connection_state_changed) on_connection_state_changed(sk_state);

    (void)lk_state;
}

void LiveKitBackend::HandleDataReceived(std::string_view topic,
                                        std::span<const std::byte> payload) {
    // Intercept the reserved agent-status topic — never forward to on_data_received.
    if (topic == kAgentStatusTopic) {
        // TODO: Parse the JSON payload and extract the "status" string.
        //
        // The payload is: {"status": "idle"} or {"status": "processing"}
        //
        // Using nlohmann/json or a minimal parse:
        //
        //   auto json = nlohmann::json::parse(payload.begin(), payload.end(),
        //                                     nullptr, /*exceptions=*/false);
        //   if (!json.is_discarded() && json.contains("status")) {
        //       std::string status = json["status"].get<std::string>();
        //       if (!status.empty() && on_agent_status) on_agent_status(status);
        //   }
        return;
    }

    if (on_data_received) {
        on_data_received(topic, payload);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Teardown
// ─────────────────────────────────────────────────────────────────────────────

void LiveKitBackend::TearDown() {
    is_connected_.store(false);

    // TODO: Disconnect the room and release the room handle.
    //
    // Using livekit-ffi:
    //
    //   if (room_) {
    //       lk_room_disconnect(room_);
    //       lk_room_release(room_);
    //       room_ = nullptr;
    //   }

    if (on_connection_state_changed) {
        on_connection_state_changed(ConnectionState::kDisconnected);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Token fetch
// ─────────────────────────────────────────────────────────────────────────────

std::string LiveKitBackend::FetchToken(const std::string& token_url,
                                       const std::string& identity) {
    // TODO: HTTP GET <token_url>?identity=<identity>, return the JWT string.
    //
    // The endpoint must return either:
    //   - A plain JWT string in the response body, or
    //   - A JSON object: { "token": "eyJ…" }
    //
    // Use libcurl, Poco::Net, cpp-httplib, or any other HTTP client available
    // in your project. The Swift and Kotlin implementations both handle both
    // response formats — the C++ version must too.
    //
    // Example with cpp-httplib:
    //
    //   httplib::Client cli(token_url);
    //   auto res = cli.Get("/?identity=" + UrlEncode(identity));
    //   if (!res || res->status != 200) throw TokenFetchFailedError(token_url);
    //
    //   // Try JSON first.
    //   auto json = nlohmann::json::parse(res->body, nullptr, false);
    //   if (!json.is_discarded() && json.contains("token"))
    //       return json["token"].get<std::string>();
    //
    //   // Fall back to plain string.
    //   auto trimmed = Trim(res->body);
    //   if (!trimmed.empty()) return trimmed;
    //
    //   throw TokenFetchFailedError(token_url);

    throw TokenFetchFailedError(token_url + " — FetchToken not yet implemented");
}

} // namespace streamkit
