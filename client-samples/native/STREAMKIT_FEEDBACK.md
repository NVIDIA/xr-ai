<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# StreamKit native C++ — integration feedback

Findings from filling in the `LiveKitBackend.cpp` stub against the upstream LiveKit C++ SDK (`livekit::Room`, `LocalParticipant`, `AudioSource`, `VideoSource`). Captured here for NVIDIA review before deciding whether to ship the gaps as backend-side workarounds or change the StreamKit API surface.

## API-surface mismatches

### 1. Stub TODOs reference an FFI shape that does not exist

The pre-existing stub uses an aspirational C API (`lk_room_create`, `lk_room_connect`, `lk_local_participant_set_microphone_enabled`, …) that does not match the real LiveKit FFI. The real `livekit-ffi` is a single `livekit_ffi_request` call serializing protobuf-encoded `FfiRequest`/`FfiResponse` messages, and the LiveKit C++ SDK wraps that in a class hierarchy (`livekit::Room`, `LocalParticipant`, etc.). Anyone following the stub's TODOs verbatim would write code that does not compile against either layer.

**Suggestion:** Replace the pseudo-code TODOs in the stub with references to the real `livekit::Room::Connect`, `LocalParticipant::publishVideoTrack`, etc., or delete the pseudo-code and link to the C++ SDK's `main_sample.cpp`.

### 2. `LiveKitBackend` did not inherit `FrameSink`

`FrameSink.h` says "LiveKitBackend will implement this once the stub is filled in" and documents the `dynamic_cast<FrameSink*>(session.GetBackend())` pattern. The stub's class header didn't actually inherit from `FrameSink`, so the cast would silently return `nullptr` even after the stub was filled in.

**Suggestion:** Either make `LiveKitBackend` inherit `FrameSink` (this PR does), or document that `FrameSink` is opt-in per backend and the cast is the host's contract check.

### 3. `livekit::ConnectionState` has no `Connecting` value

The SDK enum is only `Disconnected / Connected / Reconnecting`. StreamKit's `kConnecting` has no upstream signal — the backend has to synthesise it by firing the callback manually before `Room::Connect`.

**Suggestion:** Document this StreamKit-side convention, or surface a `Connecting` event in the C++ SDK to match Swift / Kotlin / JS.

### 4. `AudioConfig::MicrophoneMode` does not map cleanly to the C++ SDK

`AudioSource` ctor signature is `(sample_rate, num_channels, queue_size_ms)`. AEC / AGC / noise-suppression live in a separate `AudioProcessingModule`, with a different lifecycle than the audio track. There is no direct knob equivalent to `lk_audio_options_default()` in the stub's pseudo-code.

**Suggestion:** Either extend `AudioConfig` with explicit `sample_rate_hz` / `frame_size_ms` for embedded targets, or document that `MicrophoneMode` is platform-managed and ignored in native unless the backend explicitly wires up `AudioProcessingModule`.

### 5. `CameraConfig::Facing` / `device_id` is unactionable in pure C++

JS, iOS, and Android each ask the OS for a platform camera by facing-mode. There is no portable C++ camera-open API, so the native backend has to defer to the host (via `FrameSink::InjectVideoFrame`). `CameraConfig::Facing` becomes inert.

**Suggestion:** Document that `CameraConfig` only takes effect when the backend ships a platform-camera capturer; otherwise the host must drive frames through `FrameSink`. Possibly split `CameraConfig` into a "platform" subtree that other backends override.

### 6. `VideoSource(width, height)` conflicts with FrameSink's first-frame-defines-dimensions contract

`FrameSink.h` says "The first `InjectVideoFrame()` call creates a BufferCapturer-backed LiveKit track and publishes it to the room … because LiveKit requires at least one captured frame to resolve stream dimensions". That matches WebRTC's BufferCapturer model on iOS / JS, but the C++ `VideoSource` ctor requires dimensions up-front. The native backend has to lazily create the source on first frame to honour the documented contract; calling `StartCamera()` cannot eagerly publish.

**Suggestion:** Either align the C++ SDK with the buffer-capturer model (defer dimension resolution), or revise `FrameSink`'s docs to allow eager dimensioning via a hint in `StartCamera` (e.g. `CameraConfig.preferred_resolution`).

### 7. No `AudioSink` companion to `FrameSink`

`FrameSink::InjectVideoFrame` lets the host drive video frames. There is no parallel mechanism for audio. Embedded hosts that capture mic frames from a platform HAL must either subclass `LiveKitBackend` to reach the private `audio_source_` member, or invent an out-of-band channel.

**Suggestion:** Add a sibling `AudioSink` interface with `InjectAudioFrame(span<int16_t> pcm, sample_rate, channels, samples_per_channel, timestamp_us)`. `LiveKitBackend` would implement it the same way it implements `FrameSink`.

### 8. `LiveKitConfig::token_url` HTTP fetch has no portable C++ implementation

The stub's pseudo-code suggested `cpp-httplib`; the JS/Swift/Kotlin equivalents use platform-native HTTP clients. Forcing a libcurl / cpp-httplib dependency on every StreamKit consumer is heavy, and on embedded targets often impractical (no TLS root store, no DNS resolver).

**Suggestion:** Either make `FetchToken` a `virtual` member that hosts override with their preferred HTTP client (this PR's approach — throws by default), or split it out into a free function in a `streamkit/HttpToken.h` opt-in header gated behind a `STREAMKIT_ENABLE_TOKEN_FETCH` CMake option.

### 9. `livekit::initialize()` global one-shot is undocumented in StreamKit

`livekit::initialize()` must be called once per process before any other LiveKit C++ SDK call. The stub didn't call it; a backend that omits it silently misses log routing and may misbehave on shutdown.

**Suggestion:** Document this requirement, or add a `streamkit::Initialize()` free function that backends route to.

## Build-system observations

### 10. `liblivekit.so` has DT_NEEDED on `liblivekit_ffi.so` + vendored protobuf

The main library has hard dependencies on both. Vendor-bundled SDK distributions ship them as separate `.so` files in unconventional paths. Hosts that don't carry these in `/usr/lib` need explicit `LD_LIBRARY_PATH` setup at runtime and explicit `target_link_libraries` calls at build time.

**Suggestion:** Document the transitive deps in the C++ SDK's README so consumers know what to ship alongside `liblivekit.so`. (This PR's `CMakeLists.txt` finds all three with separate `find_library` calls.)

### 11. Non-uniform lib layout across distributions

Some vendor SDK bundles ship the runtime libs under deeply nested subdirectories rather than the conventional `<sdk>/lib/`. Hardcoding `${LIVEKIT_SDK_ROOT}/lib` in CMake would break those consumers.

**Suggestion:** Document a canonical layout for LiveKit C++ SDK distributions, or accept `LIVEKIT_LIB_DIR` as a separate override in the StreamKit `CMakeLists.txt` (this PR's approach).

## Real-time performance findings

### 12. `FrameSink::InjectVideoFrame` forces a redundant per-frame copy (FIX INCLUDED)

The most consequential finding from the integration — affects every native consumer pushing camera or rendered frames into LiveKit, not just embedded.

#### Theory

`FrameSink::InjectVideoFrame` is declared with a non-owning view of the pixel buffer:

```cpp
virtual void InjectVideoFrame(std::span<const std::byte> data,
                              int width, int height, int stride,
                              PixelFormat format, int64_t timestamp_us) = 0;
```

`livekit::VideoFrame` — the type the LiveKit C++ SDK actually accepts — owns its pixel buffer by value:

```cpp
VideoFrame(int width, int height, VideoBufferType type,
           std::vector<std::uint8_t> data);
```

A backend bridging the two must materialise an owned buffer from the borrowed view before it can call into the SDK. With the span-only API, the only available bridge is:

```cpp
std::vector<std::uint8_t> buffer(data.size());   // alloc + value-init (zero-fill)
std::memcpy(buffer.data(), data.data(), data.size());
livekit::VideoFrame frame(w, h, fmt, std::move(buffer));
```

Two passes over the frame buffer per call (zero-fill + memcpy), plus a fresh allocation every frame. At 720p I420 (≈1.4 MB) that's **2.8 MB of writes + 1 allocation per frame**, regardless of platform. Owning callers (camera HALs that produce a `std::vector`, game engines holding `std::vector` of pixels in their render path) pay this cost even though they could hand ownership over cleanly.

#### Test methodology

Implemented the upstream backend against an embedded ARMv7-A target (32-bit, pre-C++20 libc++) and instrumented `LiveKitBackend::InjectVideoFrame` with a four-phase split over 30-frame windows:

| Phase | Work measured |
|---|---|
| `copy` | `std::vector<uint8_t>(size)` value-init + `memcpy` |
| `ctor` | `livekit::VideoFrame` construction |
| `pub`  | Mutex + lazy `publishVideoTrack` (after frame 1 should be ~0) |
| `cap`  | `livekit::VideoSource::captureFrame` (the SDK send call itself) |

A/B'd the same hardware, same LiveKit server, same operator UI, 720p30 I420 camera frames. Two binaries, identical except for which path the test daemon takes through this backend.

#### Findings

**Span overload (current API), 30-frame averages:**

```
copy avg=63,248 us peak=77,092 us
ctor avg=     5 us peak=    22 us
pub  avg=     5 us peak=     6 us
cap  avg= 6,632 us peak=30,183 us
```

`cap` is ~7 ms, matching the in-house baseline publisher (a thin wrapper around the same SDK's lower-level facade). The entire 70 ms per-frame regression is in `copy`. The HAL is configured for 30 fps but the camera callback thread blocks for 70 ms per frame → effective rate collapses to ~6 fps. Operator UI tile shows visible stutter and glass-to-glass latency increases proportionally.

**Move overload (proposed fix), 30-frame averages:**

```
copy avg=     0 us peak=     0 us
ctor avg=     4 us peak=     5 us
pub  avg=     4 us peak=    11 us
cap  avg= 4,467 us peak=31,145 us
```

`copy` is zero (the buffer arrives by move; no allocation, no memcpy). Total per-frame cost: **5–10 ms**, matching the legacy path. **14× speedup.** Operator-perceived latency returns to the legacy baseline.

Identical hardware, identical SDK build, identical room. The only difference is which overload the caller targets.

#### Estimated impact on other native consumers

The absolute copy cost scales linearly with frame size; the allocator cost scales with both size and frequency. Indicative numbers:

| Consumer profile | Frame × fps | Per-frame waste | Wasted CPU |
|---|---|---|---|
| ARMv7-A embedded (test target) | 1280×720 @ 30 | ~63 ms | dominates (collapses to 6 fps) |
| Desktop Linux x86-64 | 1280×720 @ 30 | ~0.5–1 ms | ~2–3% of one core |
| Desktop x86-64 | 1920×1080 @ 60 | ~1.5–3 ms | ~10–15% of one core |
| Jetson / CloudXR client | 1920×1080 @ 90 | ~3–5 ms | substantial; visible frame loss |
| Native game engine plugin | 4K @ 60 | ~4–6 ms | substantial |

The embedded case is the canary, not the only victim. On desktop the cost is "free CPU you didn't expect"; nobody profiles it because the frames still flow. On CloudXR / Jetson / 1080p+ it starts showing as frame drops.

iOS Swift, Android Kotlin, and Web JS clients don't hit this because they use platform-managed capture paths (`CMSampleBuffer`, `ImageReader`, `MediaStream`) — all reference-counted or shareable without copying. **Native C++ is structurally the only path that hits this**, but every native C++ caller hits it.

#### Fix (included in this PR)

Additive overload — no behaviour change for existing callers:

```cpp
// FrameSink.h
class FrameSink {
public:
    virtual ~FrameSink() = default;

    // Existing API — kept for callers with read-only / shared buffers.
    virtual void InjectVideoFrame(std::span<const std::byte> data, …) = 0;

    // New zero-copy overload for callers that own the buffer.
    // Default impl forwards to the span overload, so backends that don't
    // override the move overload behave exactly as before.
    virtual void InjectVideoFrame(std::vector<std::uint8_t>&& data, …) {
        std::span<const std::byte> as_span(
            reinterpret_cast<const std::byte*>(data.data()), data.size());
        InjectVideoFrame(as_span, …);
    }
};
```

`LiveKitBackend` overrides both. The move overload constructs `livekit::VideoFrame` directly from the moved buffer (no alloc, no memcpy). The span overload becomes a thin forwarder that copies once and delegates to the move overload, so legacy callers still work.

See `StreamKit/include/streamkit/FrameSink.h` and `StreamKit/src/Backends/LiveKit/LiveKitBackend.cpp` for the implementation. Empirical validation: the per-frame cost drops from ~70 ms to ~5 ms on the ARMv7-A target when callers use the move overload (see the diagnostic methodology and raw numbers above).

#### Why span isn't wrong, just incomplete

There's a legitimate case for the span overload — callers with read-only mmap'd memory, GPU-mapped buffers they can't hand over, or callers broadcasting one buffer to multiple sinks. Those cases need a copy and have to live with it. The bug is the absence of a move overload for callers that already own a movable buffer.

Both audiences are now served. Non-owning callers continue to use the span overload (unchanged behaviour); owning callers get a fast path.

## Open questions

- Should StreamKit grow `streamkit::Initialize()` / `streamkit::Shutdown()` free functions to encapsulate per-backend global init? Today the LiveKit backend does it implicitly in its ctor.
- For the `_agent.status` JSON parsing, is it acceptable to ship a hand-rolled `{"status":"…"}` extractor (this PR), or does StreamKit want to pin to a JSON library (nlohmann) project-wide?
- Should StreamKit's tests adopt the `MockBackend` pattern documented in the README? Today there is no test harness in `client-samples/native/`.

## What we shipped in this PR

- CMake wiring (`LIVEKIT_SDK_ROOT` + `LIVEKIT_LIB_DIR`) — finds the SDK, falls back to stub mode without it.
- `LiveKitBackend` implements `Connect / Disconnect / Send / StartCamera (arm only) / StopCamera / StartAudio (arm only) / StopAudio / TearDown` against the real `livekit::Room` API.
- `LiveKitBackend` implements `FrameSink::InjectVideoFrame` with lazy track creation on first frame.
- `_agent.status` interception parses `{"status":"…"}` and routes through `on_agent_status`; everything else goes through `on_data_received`.
- `LIVEKIT_SDK_ROOT` unset → stub mode → builds the rest of StreamKit header-only on CI machines.

Items left intentionally unimplemented (and called out in the README's "Constraints" table): platform mic/camera capture, `FetchToken` HTTP, `MicrophoneMode` DSP mapping.
