<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# StreamKit for native C++ — LiveKit-backed client

> **Audience:** Developers who have used LiveKit before and want to embed StreamKit in a native C++ host — e.g. an embedded device, a native game engine plugin, or a CloudXR client.

The `LiveKitBackend` in `StreamKit/src/Backends/LiveKit/LiveKitBackend.cpp` is a working implementation against the upstream LiveKit C++ SDK (`livekit::Room` API). Build it by pointing CMake at a LiveKit SDK install:

```bash
cmake -S . -B build -DLIVEKIT_SDK_ROOT=/path/to/livekit-cpp-sdk
cmake --build build
./build/bin/streamkit_sample --host 192.168.1.100 --token <jwt>
```

If `LIVEKIT_SDK_ROOT` is not set, the backend compiles in stub mode: `Connect()` reports `kConnected` immediately without opening a real session. This keeps CI green on machines without the SDK and lets you build the rest of StreamKit header-only.

## Running the tests

A small unit-test suite lives under `StreamKit/Tests/StreamKitTests/`. Tests are off by default; pass `-DSTREAMKIT_BUILD_TESTS=ON` to build them and run via CTest. Stub mode is the supported path — tests do not need (and should not be paired with) a `LIVEKIT_SDK_ROOT` pointing at an ABI-incompatible build.

```bash
cmake -S . -B build -DSTREAMKIT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

What's covered:

| Test | What it asserts |
|---|---|
| `streamkit_mapping_tests` | `ConnectionState` enum basic ops. |
| `streamkit_agent_status_tests` | `_agent.status` JSON extractor — canonical / missing-key / truncated / empty-value / empty-payload cases. |
| `streamkit_frame_sink_tests` | `FrameSink`'s move-overload default impl correctly forwards to the span overload; backends that override both bypass the forwarder. |
| `streamkit_audio_sink_tests` | `AudioSink::InjectAudioFrame` delivers every parameter verbatim and dispatches correctly through an `AudioSink&` reference. |
| `streamkit_session_tests` | Full `StreamSession` lifecycle through a `MockBackend` — connect / start audio / start camera / send / receive / agent status / disconnect, verifying event-hook fan-out. |
| `streamkit_livekit_backend_tests` | `LiveKitBackend` state-change dedupe — no spurious initial `kDisconnected` on first `Connect()`, no doubled `kConnected` after a successful connect, idempotent `Disconnect()`. |

Useful variants:

```bash
# Run a single test by name
ctest --test-dir build -R streamkit_frame_sink_tests --output-on-failure

# Run a test binary directly (no ctest wrapper)
./build/StreamKit/Tests/StreamKitTests/streamkit_frame_sink_tests

# Re-run only failing tests
ctest --test-dir build --rerun-failed --output-on-failure
```

Each test is a standalone executable that asserts and exits non-zero on failure — no third-party test framework, no GoogleTest dependency.

## Constraints in the current native backend

| Area | Status |
|---|---|
| Connect / Disconnect + state mapping | ✅ implemented |
| Data channel `Send` + `_agent.status` interception | ✅ implemented |
| Video publish via `FrameSink::InjectVideoFrame` | ✅ implemented — first frame creates the track. Real-time callers should use the `std::vector<uint8_t>&&` overload to avoid a 1.4 MB per-frame copy. See finding #12 in [issue #134](https://github.com/NVIDIA/xr-ai/issues/134). |
| Audio publish via `AudioSink::InjectAudioFrame` | ✅ implemented — `StartAudio()` creates the track; host pushes PCM frames |
| Platform mic open | ⚠️ no built-in path; host opens its mic and pushes PCM frames via AudioSink |
| Platform camera open | ⚠️ no built-in path; host opens its camera and pushes frames via FrameSink. `CameraConfig::facing` / `device_id` are inert here — the host chooses the camera |
| `LiveKitConfig::token_url` HTTP fetch | ⚠️ not implemented; pass `LiveKitConfig::token` inline or override `FetchToken` |
| `AudioConfig::MicrophoneMode` mapping | ⚠️ unapplied — the C++ SDK has no AEC/AGC/NS toggles on `AudioSource` |

The mismatches above against the Swift / Kotlin / JS backends are summarised in [issue #134](https://github.com/NVIDIA/xr-ai/issues/134) — a partner-side audit of the integration with one entry per friction point.

---

## What StreamKit is (and isn't)

StreamKit is a thin transport-agnostic wrapper that sits on top of LiveKit.
It does **not** replace LiveKit — it constrains how you use it.

The existing iOS, Android, and web clients all share the same shape:

```
Application code
      │
      ▼
 StreamSession          ← single public entry-point, transport-agnostic
      │  delegates to
      ▼
 StreamingBackend       ← interface/protocol — the seam between StreamKit and LiveKit
      │  implemented by
      ▼
 LiveKitBackend         ← the only file that imports LiveKit directly
      │
      ▼
 LiveKit SDK            ← Room, LocalParticipant, tracks, data channel
```

To add a C++ StreamKit library, you implement `StreamingBackend` against the LiveKit C++ SDK (or `livekit-ffi`), and all application code stays the same.

---

## What StreamKit adds on top of LiveKit

### 1. A single entry-point with decoupled media

Raw LiveKit lets you publish tracks as part of `room.connect()`. StreamKit splits the lifecycle into three independent phases:

| Phase | StreamKit call | What it does |
|---|---|---|
| Transport | `connect(config)` | WebRTC peer connection + data channel only |
| Audio | `startAudio(config)` / `stopAudio()` | Mic capture + publish; throws without dropping the connection |
| Video | `startCamera(config)` / `stopCamera()` | Camera capture + publish; throws without dropping the connection |

**Why this matters:** Audio/camera failures are isolated. A bad camera never kills the session.

In C++, `connect()` calls `room->Connect(url, token)` and nothing else. `startAudio()` and `startCamera()` are separate calls made by the application after the room is connected.

### 2. A typed `ConnectionState` enum

LiveKit's connection state is an SDK-specific enum or string. StreamKit maps it to four values that are the same across every platform:

```
DISCONNECTED  →  CONNECTING  →  CONNECTED
                                    │
                              RECONNECTING
```

In `LiveKitBackend` (C++), subscribe to the room's connection-state delegate/callback and forward through the mapping:

```cpp
room->AddListener([this](livekit::ConnectionState state) {
    if (state == livekit::ConnectionState::kConnected)
        on_connection_state_changed_(ConnectionState::CONNECTED);
    else if (state == livekit::ConnectionState::kReconnecting)
        on_connection_state_changed_(ConnectionState::RECONNECTING);
    // …
});
```

### 3. Typed errors

Instead of propagating LiveKit's internal error types, StreamKit defines a small set of errors that apply regardless of transport:

| Error | When |
|---|---|
| `InvalidHost` | Empty or unparseable host string |
| `NotConnected` | Method called before `connect()` succeeded |
| `MissingToken` | Neither `token` nor `tokenURL` was provided |
| `TokenFetchFailed` | HTTP request to token endpoint failed |
| `CameraRequiresConnection` | `startCamera()` called while not connected |

In C++, express these as an enum, `std::error_code`, or a `std::variant` — whichever fits your project's convention. The important thing is that callers never see LiveKit-specific error types.

### 4. The agent status channel

The xr-ai server publishes internal status messages (`"idle"`, `"processing"`, etc.) on a reserved data-channel topic: `_agent.status`. The payload is a JSON object `{"status": "…"}`.

`LiveKitBackend` intercepts this topic in the `RoomEvent.DataReceived` handler and fires `onAgentStatus` instead of `onDataReceived`. Application code never sees `_agent.status` on the raw data callback.

```cpp
// In your C++ data-received handler:
static constexpr std::string_view kAgentStatusTopic = "_agent.status";

void OnDataReceived(std::span<const uint8_t> payload,
                    std::string_view topic) {
    if (topic == kAgentStatusTopic) {
        // parse JSON, extract "status", call on_agent_status_
        return;
    }
    on_data_received_(topic, payload);
}
```

### 5. `AudioConfig` and `MicrophoneMode`

Rather than exposing LiveKit's raw `AudioCaptureOptions`, StreamKit presents four presets:

| Mode | What the backend does |
|---|---|
| `VOICE_PROCESSING` | Use hardware echo cancellation (AUVoiceIO on Apple, platform equivalent elsewhere). Disable WebRTC's own AEC/AGC/NS to avoid double-processing. |
| `SOFTWARE_PROCESSING` | Enable WebRTC's software AEC, AGC, and noise suppression. |
| `RAW` | No processing. Use when the server-side agent handles DSP. |
| `DISABLED` | Don't capture or publish a microphone track at all. |

In C++, map `MicrophoneMode` to the appropriate `AudioOptions` fields when calling `local_participant->SetMicrophoneEnabled()` or when creating a local audio track.

### 6. Token acquisition

`LiveKitConfig` carries either a pre-signed JWT (`token`) or a URL to fetch one from (`tokenURL`). The token endpoint contract is:

```
GET <tokenURL>?identity=<identity>
→ 200 OK, body: "eyJ…"          (plain JWT string)
→ 200 OK, body: {"token":"eyJ…"} (JSON envelope)
```

`LiveKitBackend` handles both response shapes. In C++ you can use `libcurl` or any HTTP client; the logic is straightforward:

```cpp
std::string FetchToken(const std::string& token_url,
                       const std::string& identity) {
    std::string url = token_url + "?identity=" + UrlEncode(identity);
    std::string body = HttpGet(url);  // your HTTP client here

    // Try JSON envelope first.
    auto json = ParseJson(body);
    if (json.contains("token")) return json["token"];

    // Fall back to plain string.
    return Trim(body);
}
```

### 7. Frame injection (optional, for external video sources)

The iOS backend has a `FrameInjectable` extension that lets you push raw video frames from any external camera (e.g. the Meta wearables SDK, a game engine texture, a hardware capture card) directly into a LiveKit `BufferCapturer`-backed track:

```
external camera callback
        │
        ▼
session.injectVideoFrame(buffer)
        │
        ▼
BufferCapturer.Capture(buffer)  ← first call also publishes the track
```

For C++, implement the equivalent using `livekit::LocalVideoTrack` with a custom `VideoSource`. Publish the track lazily on the first frame (the track must have at least one frame before LiveKit can complete the publish handshake and resolve stream dimensions).

---

## The `StreamingBackend` interface you need to implement

Here is the interface in C++ terms. Every other platform has an identical surface area:

```cpp
// streamkit/streaming_backend.h

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>

enum class ConnectionState { kDisconnected, kConnecting, kConnected, kReconnecting };

struct SessionConfig {
    std::string identity;
};

struct AudioConfig {
    enum class MicrophoneMode { kVoiceProcessing, kSoftwareProcessing, kRaw, kDisabled };
    MicrophoneMode mode = MicrophoneMode::kVoiceProcessing;
};

struct CameraConfig {
    enum class Facing { kFront, kBack };
    Facing facing = Facing::kFront;
    std::string device_id;  // optional; overrides facing when set
};

class StreamingBackend {
public:
    virtual ~StreamingBackend() = default;

    // ── Event hooks (set by StreamSession before calling Connect) ────────────
    std::function<void(ConnectionState)>                       on_connection_state_changed;
    std::function<void(std::string_view, std::span<const uint8_t>)> on_data_received;
    std::function<void(std::string_view)>                      on_agent_status;

    // ── Connection ───────────────────────────────────────────────────────────
    virtual void Connect(const SessionConfig& config)    = 0;  // async
    virtual void Disconnect()                            = 0;  // async

    // ── Audio ────────────────────────────────────────────────────────────────
    virtual void StartAudio(const AudioConfig& config)   = 0;  // throws NotConnected
    virtual void StopAudio()                             = 0;

    // ── Camera ───────────────────────────────────────────────────────────────
    virtual void StartCamera(const CameraConfig& config) = 0;  // throws CameraRequiresConnection
    virtual void StopCamera()                            = 0;

    // ── Data channel ─────────────────────────────────────────────────────────
    virtual void Send(std::span<const uint8_t> data,
                      bool reliable = true,
                      std::string_view topic = "") = 0;  // throws NotConnected
};
```

`StreamSession` is a thin wrapper around this: it stores callbacks, wires them to the backend, and provides the same calls as public API. None of that logic changes when you swap to the C++ backend.

---

## Implementing `LiveKitBackend` in C++

Walk through the same structure used in Swift and Kotlin:

**Construction**: Accept a `LiveKitConfig` (host, port, secure, token/tokenURL).

**`Connect()`**:
1. Call `TearDown()` to clean up any previous session.
2. Validate `config.host`; throw `InvalidHost` if empty.
3. Acquire a JWT — either from `config.token` or by calling `FetchToken()`.
4. Build the WebSocket URL (`ws://host:port` or `wss://`).
5. Create a `livekit::Room`, register event listeners.
6. Call `room->Connect(url, token)`.
7. Fire `on_connection_state_changed(kConnecting)` before the call, `kConnected` after.

**`StartAudio()`**:
1. Guard: throw `NotConnected` if the room is not connected.
2. Map `AudioConfig::MicrophoneMode` to the appropriate `AudioOptions`.
3. Call `room->local_participant()->SetMicrophoneEnabled(true, options)`.

**`StartCamera()`**:
1. Guard: throw `CameraRequiresConnection` if not connected.
2. Call `StopCamera()` first to tear down any existing track.
3. Create a `LocalVideoTrack` with the appropriate device/facing constraint.
4. Publish it via `room->local_participant()->PublishVideoTrack(track)`.

**`Send()`**:
1. Guard: throw `NotConnected` if not connected.
2. Reject `topic == "_agent.status"` (reserved).
3. Call `room->local_participant()->PublishData(data, reliable, topic)`.

**Room event handler**:
- `ConnectionStateChanged` → map and fire `on_connection_state_changed_`.
- `DataReceived` → intercept `_agent.status` → parse JSON → fire `on_agent_status_`, else fire `on_data_received_`.
- `TrackSubscribed` (audio) → attach the remote audio track to your platform's audio renderer.

**`TearDown()`**:
- Cancel any background tasks (simulator frame loop, pending publishes).
- Disconnect and null out the room.
- Fire `on_connection_state_changed(kDisconnected)`.

---

## What you get for free once the backend is done

Once `LiveKitBackend` satisfies `StreamingBackend`, the rest of StreamKit — `StreamSession`, all configs, all error types, the agent-status channel — works without modification. You can also swap in a mock backend for unit tests without touching any LiveKit code.

```cpp
// Production
auto session = StreamSession(std::make_unique<LiveKitBackend>(config));

// Test
auto session = StreamSession(std::make_unique<MockBackend>());
```

The mock backend just fires `on_connection_state_changed(kConnected)` from `Connect()` and records every `Send()` call — no WebRTC stack needed.
