// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * StreamKit — LiveKitBackend
 *
 * Implements StreamingBackend using the LiveKit C++ SDK
 * (https://github.com/livekit/rust-sdks → `cpp/`). The SDK headers are kept
 * out of this header so that consumers only need StreamKit's own includes.
 * All LiveKit types are forward-declared and stored as opaque smart pointers;
 * the destructor lives in the .cpp where the full types are visible.
 *
 * Mirror of Swift `LiveKitBackend` and Kotlin `LiveKitBackend`.
 */

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>

#include "streamkit/AudioSink.h"
#include "streamkit/Backends/StreamingBackend.h"
#include "streamkit/Config/BackendConfiguration.h"
#include "streamkit/FrameSink.h"

namespace livekit {
class Room;
class AudioSource;
class VideoSource;
class LocalAudioTrack;
class LocalVideoTrack;
} // namespace livekit

namespace streamkit {

/// StreamingBackend implementation using the upstream LiveKit C++ SDK.
///
/// Do not construct this directly — use `BackendConfiguration{LiveKitConfig{…}}`
/// passed to StreamSession, or call `MakeBackend()`.
///
/// ## Frame ingestion
///
/// The C++ SDK requires a fixed resolution at `livekit::VideoSource`
/// construction time, so this backend cannot publish a camera track until it
/// has seen the first frame. Application code should:
///   1. Open its platform camera / capture source (out of scope here).
///   2. Call `session.StartCamera()` to arm the backend.
///   3. Push frames via the `FrameSink` interface — the first call lazily
///      creates the LocalVideoTrack and publishes it.
///
/// Audio follows the same shape via the `AudioSink` interface:
///   1. Open the platform mic / audio capture source (out of scope here).
///   2. Call `session.StartAudio()` to arm the backend and publish the track.
///   3. Push PCM frames via `AudioSink::InjectAudioFrame`.
class LiveKitBackend : public StreamingBackend,
                       public FrameSink,
                       public AudioSink {
public:
    explicit LiveKitBackend(const LiveKitConfig& config);
    ~LiveKitBackend() noexcept override;

    LiveKitBackend(const LiveKitBackend&)            = delete;
    LiveKitBackend& operator=(const LiveKitBackend&) = delete;
    LiveKitBackend(LiveKitBackend&&)                 = delete;
    LiveKitBackend& operator=(LiveKitBackend&&)      = delete;

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

    // ── FrameSink ──────────────────────────────────────────────────────────

    /// Push a video frame into the published video track. The first call
    /// after StartCamera() creates and publishes the track at the frame's
    /// dimensions; subsequent calls deliver into the same track. NV12 / RGBA
    /// / BGRA inputs are converted to I420 before reaching the SDK.
    void InjectVideoFrame(std::span<const std::byte> data,
                          int width,
                          int height,
                          PixelFormat format,
                          int64_t timestamp_us) override;

    /// Zero-copy variant — moves the buffer all the way through to the
    /// SDK's livekit::VideoFrame without an intermediate allocation.
    /// Real-time embedded callers should prefer this overload; the span
    /// overload allocates and memcpies 1.4 MB per 720p frame.
    void InjectVideoFrame(std::vector<std::uint8_t>&& data,
                          int width,
                          int height,
                          PixelFormat format,
                          int64_t timestamp_us) override;

    // ── AudioSink ──────────────────────────────────────────────────────────

    /// Push a PCM audio frame into the published audio track. The track is
    /// created in StartAudio(); InjectAudioFrame requires StartAudio() to
    /// have been called and silently drops otherwise (matches FrameSink).
    void InjectAudioFrame(std::span<const std::int16_t> pcm,
                          int sample_rate,
                          int channels,
                          int samples_per_channel,
                          int64_t timestamp_us) override;

    // ── LiveKit-specific integrations ───────────────────────────────────────

    /// Returns the underlying LiveKit room for advanced receiver-side
    /// integrations such as remote-audio rendering or AEC reference capture.
    /// Returns nullptr before Connect(), after Disconnect(), and in stub mode.
    std::shared_ptr<livekit::Room> GetRoom() const { return room_; }

protected:
    /// Fetches a LiveKit JWT from `config_.token_url`.
    ///
    /// The default implementation throws `TokenFetchFailedError` — the C++
    /// SDK doesn't ship a portable HTTP client and the embedded path
    /// supplies an inline `LiveKitConfig::token` instead. Subclass and
    /// override with whichever HTTP client your target already links
    /// against (libcurl, cpp-httplib, Poco::Net) when a token endpoint is
    /// actually required. Expected response format: plain JWT string or
    /// `{"token":"eyJ…"}`; the SDK appends `?identity=<identity>` to the
    /// configured URL.
    virtual std::string FetchToken(const std::string& url,
                                   const std::string& identity);

private:
    // Forward-declared in the .cpp; subclasses livekit::RoomDelegate and
    // bridges its event callbacks into this backend's on_* event hooks.
    class Delegate;

    /// Disconnects the room, stops all tracks, fires kDisconnected.
    void TearDown();

    /// Maps livekit::ConnectionState → StreamKit's ConnectionState and fires
    /// on_connection_state_changed. Called by the Delegate's event hooks.
    void HandleConnectionStateChange(int lk_state);

    /// Fires on_connection_state_changed only on a real transition. Used by
    /// every state-change site (TearDown, Connect, Delegate) to keep
    /// consumers from seeing spurious duplicates — both the redundant
    /// kDisconnected that TearDown would emit on a never-connected backend,
    /// and the doubled kConnected from "Delegate fired + Connect()
    /// explicitly fires too" after a blocking Room::Connect.
    void FireStateChanged(ConnectionState state);

    /// Routes incoming data packets: intercepts "_agent.status",
    /// fires on_data_received for everything else.
    void HandleDataReceived(std::string_view topic,
                            std::span<const std::byte> payload) const;

    LiveKitConfig config_;
    SessionConfig session_config_;
    CameraConfig camera_config_;

    // shared_ptr (not unique_ptr) because the type-erased deleter is
    // captured at construction. That keeps the class destructor valid when
    // the SDK headers are not included (stub mode) — unique_ptr<Forward>
    // would force the full type to be visible in every translation unit
    // that destructs the backend.
    std::shared_ptr<livekit::Room>             room_;
    std::shared_ptr<Delegate>                  delegate_;
    std::shared_ptr<livekit::AudioSource>      audio_source_;
    std::shared_ptr<livekit::LocalAudioTrack>  audio_track_;
    std::shared_ptr<livekit::VideoSource>      video_source_;
    std::shared_ptr<livekit::LocalVideoTrack>  video_track_;

    // Guards lazy track creation in InjectVideoFrame and unpublishing in
    // StopCamera / TearDown. Without this two frames racing through the
    // first-frame path would each create a track and only the second would
    // be reachable for cleanup.
    std::mutex tracks_mutex_;

    std::atomic<bool> is_connected_{false};
    std::atomic<bool> camera_armed_{false};
    std::atomic<bool> audio_armed_{false};
    std::atomic<ConnectionState> last_fired_state_{ConnectionState::kDisconnected};

    static constexpr std::string_view kAgentStatusTopic = "_agent.status";
};

} // namespace streamkit
