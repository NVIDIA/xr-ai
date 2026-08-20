// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * StreamKit — LiveKitBackend
 *
 * The single bridge between StreamKit and the upstream LiveKit C++ SDK
 * (https://github.com/livekit/rust-sdks → `cpp/`). All `livekit::` includes
 * live in this file; the rest of StreamKit never sees the SDK.
 *
 * When `STREAMKIT_HAVE_LIVEKIT` is not defined (i.e. the SDK was not found
 * by CMake), this file compiles in stub mode: Connect() fires kConnected
 * immediately without opening a real session. CI can still build the rest
 * of StreamKit on machines without the SDK.
 */

#include "streamkit/Backends/LiveKit/LiveKitBackend.h"
#include "streamkit/StreamError.h"

#include "AgentStatusParser.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <future>
#include <iterator>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#if STREAMKIT_HAVE_LIVEKIT
#include "livekit/audio_frame.h"
#include "livekit/audio_source.h"
#include "livekit/livekit.h"
#include "livekit/local_audio_track.h"
#include "livekit/local_participant.h"
#include "livekit/local_video_track.h"
#include "livekit/participant.h"
#include "livekit/remote_participant.h"
#include "livekit/remote_track_publication.h"
#include "livekit/room.h"
#include "livekit/room_delegate.h"
#include "livekit/room_event_types.h"
#include "livekit/stats.h"
#include "livekit/track.h"
#include "livekit/video_frame.h"
#include "livekit/video_source.h"
#endif

namespace streamkit {

namespace {

#if STREAMKIT_HAVE_LIVEKIT
// Lazy one-shot initialise of the SDK's global state. livekit::initialize()
// returns false if already initialised, so this is safe to call from each
// LiveKitBackend ctor — but doing it once removes the per-instance log line.
void EnsureSdkInitialised() {
    static const bool initialised = []() {
        livekit::initialize();
        return true;
    }();
    (void)initialised;
}

ConnectionState MapState(livekit::ConnectionState lk) {
    switch (lk) {
        case livekit::ConnectionState::Connected:    return ConnectionState::kConnected;
        case livekit::ConnectionState::Reconnecting: return ConnectionState::kReconnecting;
        case livekit::ConnectionState::Disconnected: return ConnectionState::kDisconnected;
    }
    return ConnectionState::kDisconnected;
}

livekit::VideoBufferType MapPixelFormat(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::kI420: return livekit::VideoBufferType::I420;
        case PixelFormat::kNV12: return livekit::VideoBufferType::NV12;
        case PixelFormat::kRGBA: return livekit::VideoBufferType::RGBA;
        case PixelFormat::kBGRA: return livekit::VideoBufferType::BGRA;
    }
    return livekit::VideoBufferType::I420;
}
#endif // STREAMKIT_HAVE_LIVEKIT

// Required tightly-packed buffer size for a frame of the given dimensions
// and format. I420 / NV12 round chroma plane dimensions up to the next
// even pixel, matching the 4:2:0 subsampling spec.
std::size_t PackedFrameSize(int width, int height, PixelFormat format) {
    const auto pixels =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    using enum PixelFormat;
    switch (format) {
        case kI420:
        case kNV12: {
            const auto chroma_w =
                (static_cast<std::size_t>(width)  + 1) / 2;
            const auto chroma_h =
                (static_cast<std::size_t>(height) + 1) / 2;
            return pixels + 2 * chroma_w * chroma_h;
        }
        case kRGBA:
        case kBGRA:
            return pixels * 4;
    }
    return 0;  // unreachable
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Delegate
// ─────────────────────────────────────────────────────────────────────────────

#if STREAMKIT_HAVE_LIVEKIT
class LiveKitBackend::Delegate final : public livekit::RoomDelegate {
public:
    explicit Delegate(LiveKitBackend* owner) : owner_(owner) {}

    void onConnectionStateChanged(livekit::Room&,
                                  const livekit::ConnectionStateChangedEvent& e) override {
        owner_->HandleConnectionStateChange(static_cast<int>(e.state));
    }

    void onDisconnected(livekit::Room&, const livekit::DisconnectedEvent&) override {
        owner_->HandleConnectionStateChange(
            static_cast<int>(livekit::ConnectionState::Disconnected));
    }

    void onReconnecting(livekit::Room&, const livekit::ReconnectingEvent&) override {
        owner_->HandleConnectionStateChange(
            static_cast<int>(livekit::ConnectionState::Reconnecting));
    }

    void onReconnected(livekit::Room&, const livekit::ReconnectedEvent&) override {
        owner_->HandleConnectionStateChange(
            static_cast<int>(livekit::ConnectionState::Connected));
    }

    void onUserPacketReceived(livekit::Room&,
                              const livekit::UserDataPacketEvent& e) override {
        // Paren init (not brace) — some embedded toolchains ship a
        // pre-final std::span whose ctor takes int, so brace init from a
        // size_type that's narrower than ptrdiff_t trips
        // -Wc++11-narrowing.
        std::span<const std::byte> bytes(
            reinterpret_cast<const std::byte*>(e.data.data()), e.data.size());
        owner_->HandleDataReceived(e.topic, bytes);
    }

    void onConnectionQualityChanged(
        livekit::Room& room,
        const livekit::ConnectionQualityChangedEvent& e) override {
        const auto* local = room.localParticipant();
        if (local && e.participant && e.participant->identity() == local->identity()) {
            owner_->HandleNetworkQualityChange(static_cast<int>(e.quality));
        }
    }

private:
    LiveKitBackend* owner_;
};
#else
class LiveKitBackend::Delegate final {};
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Construction / destruction
// ─────────────────────────────────────────────────────────────────────────────

LiveKitBackend::LiveKitBackend(const LiveKitConfig& config) : config_(config) {
#if STREAMKIT_HAVE_LIVEKIT
    EnsureSdkInitialised();
#endif
}

LiveKitBackend::~LiveKitBackend() noexcept {
    try {
        TearDown();
    } catch (...) {
        // Swallowed: a destructor must not propagate, and a failed teardown
        // here has no recoverable action.
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection
// ─────────────────────────────────────────────────────────────────────────────

void LiveKitBackend::Connect(const SessionConfig& session_config) {
    session_config_ = session_config;
    TearDown();

    if (config_.host.empty()) {
        throw InvalidHostError(config_.host);
    }

    const std::string scheme = config_.secure ? "wss" : "ws";
    const std::string ws_url =
        std::format("{}://{}:{}", scheme, config_.host, config_.port);

    std::string token;
    if (config_.token.has_value() && !config_.token->empty()) {
        token = *config_.token;
    } else if (config_.token_url.has_value() && !config_.token_url->empty()) {
        token = FetchToken(*config_.token_url, session_config_.identity);
    } else {
        throw MissingTokenError{};
    }

    FireStateChanged(ConnectionState::kConnecting);

#if STREAMKIT_HAVE_LIVEKIT
    room_ = std::make_shared<livekit::Room>();
    delegate_ = std::make_shared<Delegate>(this);
    room_->setDelegate(delegate_.get());

    livekit::RoomOptions opts;
    // auto_subscribe=true is the SDK default; spelled out so a future
    // upstream default flip doesn't silently break remote audio playback.
    opts.auto_subscribe = true;

    const bool ok = room_->connect(ws_url, token, opts);
    if (!ok) {
        room_.reset();
        delegate_.reset();
        FireStateChanged(ConnectionState::kDisconnected);
        throw StreamError("LiveKit Room::connect returned false for " + ws_url);
    }
    is_connected_.store(true);
    // The SDK delegate may have already fired kConnected during the
    // blocking Room::Connect above. FireStateChanged dedupes so consumers
    // see exactly one transition into kConnected regardless of which path
    // fired first.
    FireStateChanged(ConnectionState::kConnected);
    StartNetworkMetricsReporting();
#else
    (void)ws_url;
    (void)token;
    is_connected_.store(true);
    FireStateChanged(ConnectionState::kConnected);
    StartNetworkMetricsReporting();
#endif
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
    if (config.mode == AudioConfig::MicrophoneMode::kDisabled) {
        return;
    }

#if STREAMKIT_HAVE_LIVEKIT
    // 48 kHz mono, 0 ms queue: the SDK's documented real-time-capture mode.
    // Mic frames are pushed by the host via AudioSink::InjectAudioFrame —
    // the C++ SDK ships no built-in platform capture.
    // MicrophoneMode is not applied: the C++ SDK does not surface
    // AEC / AGC / NS toggles on AudioSource — software DSP would go
    // through AudioProcessingModule, tracked as a follow-up.
    StopAudio();
    std::lock_guard<std::mutex> lock(tracks_mutex_);
    audio_source_ = std::make_shared<livekit::AudioSource>(48000, 1, 0);
    audio_track_ = room_->localParticipant()->publishAudioTrack(
        "mic", audio_source_, livekit::TrackSource::SOURCE_MICROPHONE);
    audio_armed_.store(true);
#else
    (void)config;
    audio_armed_.store(true);
#endif
}

void LiveKitBackend::StopAudio() {
#if STREAMKIT_HAVE_LIVEKIT
    std::lock_guard<std::mutex> lock(tracks_mutex_);
    if (audio_track_ && room_) {
        room_->localParticipant()->unpublishTrack(audio_track_->sid());
    }
    audio_track_.reset();
    audio_source_.reset();
#endif
    // Disarm in both real and stub builds — `StartAudio()` sets `audio_armed_`
    // in both branches, so `StopAudio()` must clear it in both branches too,
    // otherwise stub builds never satisfy the "silently drops…after
    // `StopAudio()`" contract documented on `AudioSink::InjectAudioFrame`.
    audio_armed_.store(false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera
// ─────────────────────────────────────────────────────────────────────────────

void LiveKitBackend::StartCamera(const CameraConfig& config) {
    if (!is_connected_.load()) {
        throw CameraRequiresConnectionError{};
    }
    // VideoSource ctor requires explicit width/height, so the track cannot
    // be created here. Arm the backend; the first FrameSink::InjectVideoFrame
    // call creates the source and publishes lazily.
    StopCamera();
    camera_config_ = config;
    camera_armed_.store(true);
}

void LiveKitBackend::StopCamera() {
#if STREAMKIT_HAVE_LIVEKIT
    std::lock_guard<std::mutex> lock(tracks_mutex_);
    if (video_track_ && room_) {
        room_->localParticipant()->unpublishTrack(video_track_->sid());
    }
    video_track_.reset();
    video_source_.reset();
    camera_armed_.store(false);
#else
    camera_armed_.store(false);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// FrameSink
// ─────────────────────────────────────────────────────────────────────────────

void LiveKitBackend::InjectVideoFrame(std::span<const std::byte> data,
                                      int width,
                                      int height,
                                      PixelFormat format,
                                      int64_t timestamp_us) {
    // Span entrypoint: own the buffer so we can move it through. Hot-path
    // callers should use the std::vector&& overload to avoid this copy.
    std::vector<std::uint8_t> buffer(data.size());
    std::memcpy(buffer.data(), data.data(), data.size());
    InjectVideoFrame(std::move(buffer), width, height, format, timestamp_us);
}

void LiveKitBackend::InjectVideoFrame(std::vector<std::uint8_t>&& data,
                                      int width,
                                      int height,
                                      PixelFormat format,
                                      int64_t timestamp_us) {
    if (!is_connected_.load()) {
        throw NotConnectedError{};
    }
    if (!camera_armed_.load()) {
        return;
    }

    // FrameSink's contract is packed buffers — validate the byte count
    // matches what a packed frame of the declared dimensions / format
    // requires. Catches the common "did the caller forget to repack
    // their padded HAL or GPU readback buffer?" mistake.
    if (const auto expected = PackedFrameSize(width, height, format);
        data.size() != expected) {
        throw std::invalid_argument(std::format(
            "InjectVideoFrame: buffer size {}"
            " does not match the packed size {}"
            " expected for the given dimensions and format. FrameSink "
            "requires tightly packed input - repack padded buffers before "
            "calling.",
            data.size(), expected));
    }

#if STREAMKIT_HAVE_LIVEKIT
    const auto lk_format = MapPixelFormat(format);
    livekit::VideoFrame frame(width, height, lk_format, std::move(data));

    // Convert to I420 if the input is NV12 / RGBA / BGRA. The LiveKit SDK
    // accepts non-I420 buffers but most downstream pipelines want I420 and
    // the conversion FFI is cheap relative to encode.
    std::optional<livekit::VideoFrame> i420;
    if (lk_format != livekit::VideoBufferType::I420) {
        i420 = frame.convert(livekit::VideoBufferType::I420);
    }
    const livekit::VideoFrame& outgoing = i420 ? *i420 : frame;

    std::shared_ptr<livekit::VideoSource> source;
    {
        std::lock_guard<std::mutex> lock(tracks_mutex_);
        if (!video_source_) {
            video_source_ = std::make_shared<livekit::VideoSource>(width, height);
            video_track_ = livekit::LocalVideoTrack::createLocalVideoTrack(
                "camera", video_source_);
            livekit::TrackPublishOptions options;
            options.source = livekit::TrackSource::SOURCE_CAMERA;
            if (camera_config_.encoding.has_value()) {
                const auto& encoding = *camera_config_.encoding;
                if (encoding.max_bitrate_bps > 0 || encoding.max_framerate > 0.0) {
                    livekit::VideoEncodingOptions video_encoding;
                    video_encoding.max_bitrate = encoding.max_bitrate_bps;
                    video_encoding.max_framerate = encoding.max_framerate;
                    options.video_encoding = video_encoding;
                }
                if (encoding.simulcast.has_value()) {
                    options.simulcast = encoding.simulcast;
                }
            }
            room_->localParticipant()->publishTrack(video_track_, options);
        }
        source = video_source_;
    }
    source->captureFrame(outgoing, timestamp_us);
#else
    std::vector<std::uint8_t> ignored = std::move(data);
    (void)ignored;
    (void)width;
    (void)height;
    (void)format;
    (void)timestamp_us;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// AudioSink
// ─────────────────────────────────────────────────────────────────────────────

void LiveKitBackend::InjectAudioFrame(std::span<const std::int16_t> pcm,
                                      int sample_rate,
                                      int channels,
                                      int samples_per_channel,
                                      int64_t timestamp_us) {
    if (!is_connected_.load()) {
        throw NotConnectedError{};
    }
    if (!audio_armed_.load()) {
        return;
    }

    if (const auto expected =
            static_cast<std::size_t>(channels) *
            static_cast<std::size_t>(samples_per_channel);
        pcm.size() != expected) {
        throw std::invalid_argument(std::format(
            "InjectAudioFrame: sample count {}"
            " does not match channels * samples_per_channel = {}",
            pcm.size(), expected));
    }

#if STREAMKIT_HAVE_LIVEKIT
    // AudioSink timestamps are media timestamps; AudioSource::captureFrame's
    // second parameter is a bounded-wait timeout in milliseconds.
    (void)timestamp_us;
    std::vector<std::int16_t> data(pcm.begin(), pcm.end());
    livekit::AudioFrame frame(std::move(data), sample_rate, channels,
                              samples_per_channel);
    std::shared_ptr<livekit::AudioSource> source;
    {
        std::lock_guard<std::mutex> lock(tracks_mutex_);
        source = audio_source_;
    }
    if (source) {
        source->captureFrame(frame);
    }
#else
    (void)sample_rate;
    (void)channels;
    (void)samples_per_channel;
    (void)timestamp_us;
#endif
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

#if STREAMKIT_HAVE_LIVEKIT
    std::vector<std::uint8_t> payload(data.size());
    std::memcpy(payload.data(), data.data(), data.size());
    room_->localParticipant()->publishData(payload, reliable, {}, std::string(topic));
#else
    (void)data;
    (void)reliable;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Event handlers
// ─────────────────────────────────────────────────────────────────────────────

void LiveKitBackend::HandleConnectionStateChange(int lk_state) {
#if STREAMKIT_HAVE_LIVEKIT
    const auto sk = MapState(static_cast<livekit::ConnectionState>(lk_state));
    is_connected_.store(sk == ConnectionState::kConnected);
    FireStateChanged(sk);
#else
    (void)lk_state;
#endif
}

void LiveKitBackend::FireStateChanged(ConnectionState state) {
    if (last_fired_state_.exchange(state) == state) {
        // Same state as last fire — skip the callback.
        return;
    }
    if (on_connection_state_changed) {
        on_connection_state_changed(state);
    }
}

void LiveKitBackend::HandleDataReceived(std::string_view topic,
                                        std::span<const std::byte> payload) const {
    if (topic == kAgentStatusTopic) {
        if (auto status = internal::ExtractAgentStatus(payload)) {
            if (!status->empty() && on_agent_status) {
                on_agent_status(*status);
            }
        }
        return;
    }
    if (on_data_received) {
        on_data_received(topic, payload);
    }
}

void LiveKitBackend::HandleNetworkQualityChange(int lk_quality) {
#if STREAMKIT_HAVE_LIVEKIT
    switch (static_cast<livekit::ConnectionQuality>(lk_quality)) {
        case livekit::ConnectionQuality::Excellent:
            network_quality_.store(NetworkQuality::kExcellent);
            break;
        case livekit::ConnectionQuality::Good:
            network_quality_.store(NetworkQuality::kGood);
            break;
        case livekit::ConnectionQuality::Poor:
            network_quality_.store(NetworkQuality::kPoor);
            break;
        case livekit::ConnectionQuality::Lost:
            network_quality_.store(NetworkQuality::kLost);
            break;
        default:
            network_quality_.store(NetworkQuality::kUnknown);
            break;
    }
#else
    (void)lk_quality;
#endif
}

void LiveKitBackend::StartNetworkMetricsReporting() {
    StopNetworkMetricsReporting();
    auto stop = std::make_shared<std::atomic<bool>>(false);
    network_metrics_stop_ = stop;
    std::promise<void> assigned;
    auto assigned_future = assigned.get_future();
    std::thread worker([this, stop, assigned = std::move(assigned_future)]() mutable {
        // The worker must not call Disconnect() before its std::thread has
        // been moved into network_metrics_thread_.
        assigned.wait();
        while (!stop->load()) {
            if (is_connected_.load()) {
                PublishNetworkMetrics();
            }
            // PublishNetworkMetrics may invoke a callback that destroys this
            // backend. Only the separately-owned stop flag is safe to touch
            // after that callback returns.
            for (int tenth = 0; tenth < 10 && !stop->load(); ++tenth) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    });
    network_metrics_thread_ = std::move(worker);
    assigned.set_value();
}

void LiveKitBackend::StopNetworkMetricsReporting() {
    if (network_metrics_stop_) {
        network_metrics_stop_->store(true);
    }
    if (network_metrics_thread_.joinable()) {
        if (network_metrics_thread_.get_id() == std::this_thread::get_id()) {
            network_metrics_thread_.detach();
        } else {
            network_metrics_thread_.join();
        }
    }
    network_metrics_stop_.reset();
}

void LiveKitBackend::PublishNetworkMetrics() {
    const auto callback = on_network_metrics;
    if (!callback) return;

    NetworkMetrics metrics{
        .quality = network_quality_.load(),
        .round_trip_time_ms = std::nullopt,
        .receive_jitter_ms = std::nullopt,
    };
#if STREAMKIT_HAVE_LIVEKIT
    const auto room = room_;
    if (!room) return;

    try {
        std::vector<std::shared_ptr<livekit::Track>> tracks;
        {
            std::lock_guard<std::mutex> lock(tracks_mutex_);
            if (audio_track_) tracks.push_back(audio_track_);
            if (video_track_) tracks.push_back(video_track_);
        }
        for (const auto& participant : room->remoteParticipants()) {
            if (!participant) continue;
            for (const auto& [_, publication] : participant->trackPublications()) {
                if (publication && publication->track()) {
                    tracks.push_back(publication->track());
                }
            }
        }

        std::vector<std::future<std::vector<livekit::RtcStats>>> futures;
        futures.reserve(tracks.size());
        for (const auto& track : tracks) {
            futures.push_back(track->getStats());
        }

        std::vector<livekit::RtcStats> records;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(750);
        for (auto& future : futures) {
            if (future.wait_until(deadline) != std::future_status::ready) continue;
            auto track_records = future.get();
            records.insert(records.end(),
                           std::make_move_iterator(track_records.begin()),
                           std::make_move_iterator(track_records.end()));
        }

        std::unordered_set<std::string> selected_pair_ids;
        for (const auto& record : records) {
            if (const auto* transport = std::get_if<livekit::RtcTransportStats>(&record.stats);
                transport && !transport->transport.selected_candidate_pair_id.empty()) {
                selected_pair_ids.insert(transport->transport.selected_candidate_pair_id);
            }
        }

        const livekit::RtcCandidatePairStats* fallback_pair = nullptr;
        for (const auto& record : records) {
            if (const auto* pair = std::get_if<livekit::RtcCandidatePairStats>(&record.stats)) {
                if (selected_pair_ids.contains(pair->rtc.id)) {
                    const double seconds = pair->candidate_pair.current_round_trip_time;
                    if (std::isfinite(seconds) && seconds > 0.0) {
                        metrics.round_trip_time_ms = seconds * 1'000.0;
                    }
                    break;
                }
                if (!fallback_pair && pair->candidate_pair.nominated) fallback_pair = pair;
            }
        }
        if (!metrics.round_trip_time_ms && fallback_pair) {
            const double seconds = fallback_pair->candidate_pair.current_round_trip_time;
            if (std::isfinite(seconds) && seconds > 0.0) {
                metrics.round_trip_time_ms = seconds * 1'000.0;
            }
        }

        for (const auto& record : records) {
            if (const auto* inbound = std::get_if<livekit::RtcInboundRtpStats>(&record.stats)) {
                const double seconds = inbound->received.jitter;
                if (!std::isfinite(seconds) || seconds < 0.0) continue;
                const double jitter_ms = seconds * 1'000.0;
                if (!metrics.receive_jitter_ms || jitter_ms > *metrics.receive_jitter_ms) {
                    metrics.receive_jitter_ms = jitter_ms;
                }
            }
        }
    } catch (...) {
        // Quality remains useful while RTCStats is temporarily unavailable.
    }
#endif
    callback(metrics);
}

// ─────────────────────────────────────────────────────────────────────────────
// Teardown
// ─────────────────────────────────────────────────────────────────────────────

void LiveKitBackend::TearDown() {
    is_connected_.store(false);
    StopNetworkMetricsReporting();
    network_quality_.store(NetworkQuality::kUnknown);
    camera_armed_.store(false);
    audio_armed_.store(false);

#if STREAMKIT_HAVE_LIVEKIT
    {
        std::lock_guard<std::mutex> lock(tracks_mutex_);
        video_track_.reset();
        video_source_.reset();
        audio_track_.reset();
        audio_source_.reset();
    }
    if (room_) {
        room_->setDelegate(nullptr);
        room_.reset();
    }
    delegate_.reset();
#endif

    // FireStateChanged dedupes — a fresh ctor + first Connect goes
    // kDisconnected -> kConnecting -> kConnected without an initial
    // spurious kDisconnected from this call.
    FireStateChanged(ConnectionState::kDisconnected);
}

// ─────────────────────────────────────────────────────────────────────────────
// Token fetch
// ─────────────────────────────────────────────────────────────────────────────

std::string LiveKitBackend::FetchToken(
    const std::string& token_url,
    const std::string& /*identity*/) {
    // No HTTP client shipped — callers must pass an inline JWT via
    // `LiveKitConfig::token` (computed server-side).
    throw TokenFetchFailedError(
        token_url + " - FetchToken is not implemented in this backend; supply "
        "an inline token in LiveKitConfig::token");
}

} // namespace streamkit
