// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * StreamKit — StreamingBackend
 *
 * This is the single seam between StreamSession and any networking technology.
 * The library ships with a LiveKit implementation; to use a different transport
 * just subclass this and pass your instance to StreamSession.
 *
 * Mirror of Swift `StreamingBackend` protocol and Kotlin `StreamingBackend`
 * interface.
 */

#include <cstddef>
#include <functional>
#include <span>
#include <string_view>

#include "streamkit/Config/AudioConfig.h"
#include "streamkit/Config/CameraConfig.h"
#include "streamkit/Config/SessionConfig.h"
#include "streamkit/ConnectionState.h"

namespace streamkit {

/// The contract that every networking backend must satisfy.
///
/// StreamSession delegates all network operations to an object inheriting
/// this class. The call-site never depends on a specific transport technology.
///
/// ## Lifecycle
///
/// ```
/// Connect()             → WebRTC peer connection + data channel only
/// StartAudio(config)    → mic capture + publish   (independent, throws on failure)
/// StartCamera(config)   → cam capture + publish   (independent, throws on failure)
/// Send(data, reliable)  → data channel message
/// StopAudio()           → stop microphone
/// StopCamera()          → stop camera
/// Disconnect()          → tear down everything
/// ```
///
/// Audio and camera failures never affect the connection itself.
///
/// ## Threading
///
/// Methods are called from whichever thread the application chooses.
/// The implementation is responsible for any required synchronisation.
/// Callbacks fire from whatever thread the backend's event loop runs on;
/// use the mechanism appropriate for your platform (dispatch queue, strand,
/// UI-thread post, etc.) to forward them to your application.
///
/// ## Implementing a custom backend
///
/// ```cpp
/// class MyBackend : public streamkit::StreamingBackend {
/// public:
///     void Connect(const SessionConfig& config) override {
///         on_connection_state_changed(ConnectionState::kConnected);
///     }
///     void Disconnect() override {}
///     void StartAudio(const AudioConfig&) override {}
///     void StopAudio() override {}
///     void StartCamera(const CameraConfig&) override {}
///     void StopCamera() override {}
///     void Send(std::span<const std::byte>, bool, std::string_view) override {}
/// };
///
/// auto session = StreamSession(std::make_unique<MyBackend>());
/// ```
class StreamingBackend {
public:
    virtual ~StreamingBackend() = default;

    // ── Event hooks ────────────────────────────────────────────────────────
    //
    // StreamSession assigns these before calling Connect().
    // Fire from any thread; StreamSession re-dispatches if needed.

    /// Fired when the connection state changes.
    std::function<void(ConnectionState)> on_connection_state_changed;

    /// Fired when binary data arrives from the remote end.
    /// `topic` identifies the logical channel; `data` is the raw payload.
    std::function<void(std::string_view topic,
                       std::span<const std::byte> data)> on_data_received;

    /// Fired when an agent publishes a status update on the reserved
    /// `_agent.status` topic. Common values: "idle", "processing".
    /// These messages are NOT forwarded to on_data_received.
    std::function<void(std::string_view status)> on_agent_status;

    // ── Connection ─────────────────────────────────────────────────────────

    /// Establish a WebRTC peer connection and data channel.
    /// Does NOT start audio or camera capture.
    /// Throws a StreamError subclass on failure.
    virtual void Connect(const SessionConfig& config) = 0;

    /// Cleanly disconnect and release all resources.
    virtual void Disconnect() = 0;

    // ── Audio ──────────────────────────────────────────────────────────────

    /// Begin microphone capture and publish an audio track.
    /// Throws NotConnectedError if called before Connect() succeeds.
    /// A failure here does NOT affect the connection.
    virtual void StartAudio(const AudioConfig& config = AudioConfig::Default()) = 0;

    /// Stop microphone capture and unpublish the audio track.
    virtual void StopAudio() = 0;

    // ── Camera ─────────────────────────────────────────────────────────────

    /// Begin camera capture and publish a video track.
    /// Throws CameraRequiresConnectionError if called before Connect() succeeds.
    /// A failure here does NOT affect the connection.
    ///
    /// `CameraConfig::facing` and `CameraConfig::device_id` are only honoured
    /// by backends that open a camera themselves. Backends without a
    /// portable camera-open path (the built-in C++ `LiveKitBackend`) ignore
    /// both fields and expect frames to be pushed via
    /// `FrameSink::InjectVideoFrame`. See `CameraConfig.h`.
    virtual void StartCamera(const CameraConfig& config = CameraConfig::Default()) = 0;

    /// Stop camera capture and unpublish the video track.
    virtual void StopCamera() = 0;

    // ── Data channel ───────────────────────────────────────────────────────

    /// Send binary data to remote participants.
    ///
    /// \param data     Payload. Keep individual messages ≤ 15 KB (LiveKit MTU).
    /// \param reliable Ordered + guaranteed delivery when true (default).
    /// \param topic    Optional logical channel label. The reserved topic
    ///                 "_agent.status" is rejected with std::invalid_argument.
    ///
    /// Throws NotConnectedError if not connected.
    virtual void Send(std::span<const std::byte> data,
                      bool reliable = true,
                      std::string_view topic = "") = 0;
};

} // namespace streamkit
