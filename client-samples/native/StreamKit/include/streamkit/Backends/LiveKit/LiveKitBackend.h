// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * StreamKit — LiveKitBackend
 *
 * Implements StreamingBackend using the LiveKit C++ SDK / livekit-ffi.
 * This header is the only place in StreamKit that should include LiveKit
 * headers directly — all other StreamKit code depends only on StreamingBackend.
 *
 * NOTE: The implementation in LiveKitBackend.cpp is currently a stub.
 *       See that file and the README for instructions on wiring it up.
 *
 * Mirror of Swift `LiveKitBackend` and Kotlin `LiveKitBackend`.
 */

#include <atomic>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "streamkit/Backends/StreamingBackend.h"
#include "streamkit/Config/BackendConfiguration.h"

// Forward-declare the LiveKit room handle so that users of this header
// don't need to include LiveKit headers. Replace with the actual type
// once the LiveKit C++ SDK dependency is in place.
//
// e.g. #include <livekit/room.h>  →  livekit::Room
struct LKRoom;  // TODO: replace with the real LiveKit room type

namespace streamkit {

/// StreamingBackend implementation using the LiveKit C++ SDK.
///
/// Do not construct this directly — use BackendConfiguration{LiveKitConfig{…}}
/// passed to StreamSession, or call MakeBackend().
///
/// ## Status
/// The public interface is complete and matches the Swift / Kotlin backends
/// exactly. The method bodies in LiveKitBackend.cpp are stubs annotated with
/// TODO comments that reference the exact LiveKit C++ / livekit-ffi API calls
/// needed to complete each one.
class LiveKitBackend final : public StreamingBackend {
public:
    explicit LiveKitBackend(const LiveKitConfig& config);
    ~LiveKitBackend() override;

    // Non-copyable, non-movable (holds live SDK state).
    LiveKitBackend(const LiveKitBackend&)             = delete;
    LiveKitBackend& operator=(const LiveKitBackend&)  = delete;
    LiveKitBackend(LiveKitBackend&&)                  = delete;
    LiveKitBackend& operator=(LiveKitBackend&&)       = delete;

    // ── StreamingBackend ───────────────────────────────────────────────────

    void Connect(const SessionConfig& config) override;
    void Disconnect() override;

    void StartAudio(const AudioConfig& config = AudioConfig::Default()) override;
    void StopAudio() override;

    void StartCamera(const CameraConfig& config = CameraConfig::Default()) override;
    void StopCamera() override;

    void Send(std::span<const std::byte> data,
              bool reliable = true,
              std::string_view topic = "") override;

private:
    // ── Private helpers ────────────────────────────────────────────────────

    /// Disconnects the room, stops all tracks, fires kDisconnected.
    void TearDown();

    /// Fetches a LiveKit JWT from `config_.token_url`.
    /// GET <url>?identity=<identity> → plain string or {"token":"eyJ…"}.
    std::string FetchToken(const std::string& url, const std::string& identity);

    /// Maps LiveKit connection-state events to StreamKit's ConnectionState
    /// and fires on_connection_state_changed.
    void HandleConnectionStateChange(/* livekit state */ int lk_state);

    /// Routes incoming data-channel messages: intercepts "_agent.status",
    /// fires on_data_received for everything else.
    void HandleDataReceived(std::string_view topic,
                            std::span<const std::byte> payload);

    // ── State ──────────────────────────────────────────────────────────────

    LiveKitConfig config_;
    SessionConfig session_config_;

    // TODO: Replace LKRoom* with the actual LiveKit room type once the SDK
    // dependency is in place. Using a raw forward-declared pointer here so
    // this header compiles without LiveKit headers.
    LKRoom* room_ = nullptr;

    std::atomic<bool> is_connected_{false};

    // Reserved topic for internal agent-status messages. Must match the
    // server-side constant in xr-ai-pipecat.
    static constexpr std::string_view kAgentStatusTopic = "_agent.status";
};

} // namespace streamkit
