// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * StreamKit — StreamSession
 *
 * The single public entry-point of the SDK.
 *
 * StreamSession is transport-agnostic: it delegates all network operations to
 * a StreamingBackend. Application code never imports LiveKit directly.
 *
 * ## Lifecycle
 *
 *   // 1. Connect — WebRTC peer connection + data channel only.
 *   session.Connect(SessionConfig{.identity = "workstation-1"});
 *
 *   // 2. Start media independently — each throws its own error,
 *   //    never drops the connection.
 *   session.StartAudio();
 *   session.StartCamera();
 *
 *   // 3. Send / receive data.
 *   session.on_data_received = [](auto topic, auto data) { ... };
 *   session.Send(payload);
 *
 *   // 4. Stop media / disconnect.
 *   session.StopAudio();
 *   session.StopCamera();
 *   session.Disconnect();
 *
 * Mirror of Swift `StreamSession`, Kotlin `StreamSession`, and JS `StreamSession`.
 */

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "streamkit/Backends/StreamingBackend.h"
#include "streamkit/Config/AudioConfig.h"
#include "streamkit/Config/BackendConfiguration.h"
#include "streamkit/Config/CameraConfig.h"
#include "streamkit/Config/SessionConfig.h"
#include "streamkit/ConnectionState.h"

namespace streamkit {

class StreamSession {
public:
    // ── Construction ───────────────────────────────────────────────────────

    /// Create a session backed by a built-in transport.
    explicit StreamSession(const BackendConfiguration& config);

    /// Create a session backed by a custom StreamingBackend implementation.
    explicit StreamSession(std::unique_ptr<StreamingBackend> backend);

    ~StreamSession() = default;

    // Non-copyable; movable.
    StreamSession(const StreamSession&)             = delete;
    StreamSession& operator=(const StreamSession&)  = delete;
    StreamSession(StreamSession&&)                  = default;
    StreamSession& operator=(StreamSession&&)       = default;

    // ── Event hooks (wire before calling Connect) ──────────────────────────

    /// Fired when the connection lifecycle state changes.
    std::function<void(ConnectionState)> on_connection_state_changed;

    /// Fired when data is received from remote participants.
    /// `topic` identifies the logical channel; `data` is the raw payload.
    std::function<void(std::string_view topic,
                       std::span<const std::byte> data)> on_data_received;

    /// Fired when an agent publishes a status update.
    /// Common values: "idle", "processing".
    std::function<void(std::string_view status)> on_agent_status;

    // ── State ──────────────────────────────────────────────────────────────

    ConnectionState connection_state() const { return connection_state_; }

    // ── Connection ─────────────────────────────────────────────────────────

    /// Establishes a WebRTC peer connection and data channel.
    /// Does NOT start audio or camera — call StartAudio() and StartCamera()
    /// explicitly once connected.
    void Connect(const SessionConfig& config = SessionConfig::Default());

    /// Disconnects and releases all resources.
    void Disconnect();

    // ── Audio ──────────────────────────────────────────────────────────────

    /// Starts microphone capture and publishes an audio track.
    /// Throws NotConnectedError if not connected. Never drops the connection.
    void StartAudio(const AudioConfig& config = AudioConfig::Default());

    /// Stops microphone capture.
    void StopAudio();

    // ── Camera ─────────────────────────────────────────────────────────────

    /// Starts camera capture and publishes a video track.
    /// Throws CameraRequiresConnectionError if not connected.
    /// Never drops the connection.
    void StartCamera(const CameraConfig& config = CameraConfig::Default());

    /// Stops camera capture.
    void StopCamera();

    // ── Data channel ───────────────────────────────────────────────────────

    /// Sends binary data to remote participants.
    ///
    /// \param data     Payload. Keep individual messages ≤ 15 KB (LiveKit MTU).
    /// \param reliable Ordered + guaranteed delivery when true (default).
    /// \param topic    Optional logical channel label.
    void Send(std::span<const std::byte> data,
              bool reliable = true,
              std::string_view topic = "");

    // ── Advanced ──────────────────────────────────────────────────────────

    /// Returns the underlying backend. Cast to FrameSink* to inject video
    /// frames from an external camera source.
    StreamingBackend* GetBackend() { return backend_.get(); }

private:
    void WireCallbacks();

    std::unique_ptr<StreamingBackend> backend_;
    ConnectionState connection_state_ = ConnectionState::kDisconnected;
};

} // namespace streamkit
