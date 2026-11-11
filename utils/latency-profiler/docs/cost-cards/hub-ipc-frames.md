<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Hub client IPC, codec, shared-memory frames, live frames, image capture (agent-sdk/xr-ai-hub/xr_ai_hub/_processor.py, _codec.py, _shm.py, _live_frames.py, _image_capture.py, _capture.py, _file_ordering.py; hub side for reference: services/device-io-hub/device_io_hub/ipc/_hub.py)

## Role in a turn
Transport between the LiveKit connector/hub process and agent processes. Carries mic audio in (critical path every turn, 100 msgs/s/pid), TTS audio and data out (critical path), and on-demand camera frames (critical path only for vision turns: the "look at this" turn pays a `request_frame` or an image capture on top of the VLM call). All local `ipc://` ZMQ sockets (PUB/SUB for hub->agents, PUSH/PULL for agents->hub); video pixels are in POSIX shared memory between connector and hub, and cross to agents as msgpack over ZMQ only on request.

## Critical-path calls
Audio in (per 10 ms chunk): hub `PUB` -> agent `SUB` `recv_multipart` -> `decode` (msgpack unpackb, `bytes()` copy of the float32 payload) -> `create_task(cb)` (_processor.py:743-755, 1037-1039; _codec.py:84-94, 110-117).
Audio/data out: `encode` (msgpack packb of a list, binary payload copy) + `await push.send` (ZMQ PUSH, non-blocking until HWM) (_processor.py:527-543; _codec.py:72-81). Hub forwards to the LiveKit return pipe.
Flush: one small message (`RETURN_AUDIO_FLUSH`), (_processor.py:545-557).
Vision (`LiveFrameSource.get`, `_live_frames.py:79-120`):
1. Wait for a fresh `FrameSignal` (`max_age_s=2.0`, `timeout_s=5.0`) - usually already cached (signals arrive at camera fps).
2. `await endpoint.request_frame(signal)` (`_processor.py:662-699`): push `FRAME_REQUEST`, hub copies the held SHM slot (`bytes(view.data[:data_sz])`) and publishes `FRAME_DATA` on `video_data.{pid}.{track}` (_hub.py:651-669); agent decodes (msgpack + bytes) and resolves the future. Concurrent requests for the same (pid, track) coalesce (`_pending`). Timeout 1.0 s -> returns None -> `FrameUnavailable`.
Client image capture (`_ClientImageCaptureSource.capture`, `_image_capture.py:42-81`): PUSH `IMAGE_CAPTURE_REQUEST` -> hub -> client app takes a photo -> `IMAGE_CAPTURE_DATA` (encoded JPEG/PNG) back; wait `timeout_s=10.0`. Dominated by the client device and encoded image size.
Files: separate bounded lane (`file_hwm=2`), `decode` offloaded to a thread (_processor.py:804), pending/join grace `_PENDING_FILE_TTL_S=5.0` (_file_ordering.py:17).

## Fixed waits & timers
| Constant | Value | Where | Effect |
|---|---|---|---|
| `_FRAME_REQUEST_TIMEOUT` | 1.0 s | _processor.py:89 | Max wait for pixels |
| `LiveFrameSource.max_age_s` / `timeout_s` | 2.0 s / 5.0 s | _live_frames.py:38-39 | Staleness accepted (frame may be up to 2 s old!); wait for a signal |
| Image capture `timeout_s` | 10.0 s | _image_capture.py:30 | Worst-case wait |
| `_PROBE_RETRY_INTERVAL` / `_PROBE_TIMEOUT` | 0.02 s / 5.0 s | _processor.py:97-98 | Subscription barrier retry cadence (startup/join) |
| `_ROSTER_HANDSHAKE_WAIT` | 0.1 s | _processor.py:705 | Delay before roster request |
| `file_hwm` | 2 | _processor.py:234 | File lane depth |
| `_PENDING_FILE_TTL_S` / `_DEPARTED_SESSION_TTL_S` | 5 s / 60 s | _file_orderer 17-18 | File ordering around joins |
| ZMQ HWM (agent SUB/PUSH) | ZMQ default 1000 msgs | not set in _processor.py (only file lane sets RCVHWM) | At 100 audio msgs/s this is ~10 s of audio before drops on the real-time lane |
| Return-audio bound at hub | 3.0 s | services/.../livekit/config.py:11 | |
| Slot ring | `num_slots` x `max_frame_bytes` (connector-defined) | _shm.py:119-151 | `_claim_slot` raises when all slots held (backpressure on producer, _shm.py:371-380) |
| Wire format | `[u8 type][msgpack]` | _codec.py:40-94 | |

## Cost drivers
- Audio: msgs/s = 100 x #participants (10 ms frames); bytes/msg = 4 B x samples x channels (480 samples @ 48 kHz mono float32 = 1.9 KB). msgpack encode/decode per message ~10-30 us each way (guess).
- Video signals: one `FrameSignal` per camera frame per participant (30 fps) -> one `create_task` per registered `on_frame` callback (_processor.py:1024-1026). Signals are small.
- Pixel payloads: frame bytes = W x H x bpp (1080p RGBA ~8.3 MB, 720p RGB ~2.8 MB, 640x480 RGB ~0.9 MB). Copies per `request_frame`: SHM->bytes (hub), msgpack pack (copy), kernel socket copy (send), kernel->user (recv), msgpack unpack (copy) = roughly 4-5 memcpys; plus the hub dispatches serially, so the copy+send blocks the hub loop (which also forwards audio).
- `FRAME_DATA` is published to every endpoint subscribed to VIDEO for that pid (e.g. the voice endpoint too, _transport.py:673) - each pays recv + decode.
- #images per turn: one `request_frame` per camera per image; no batching API.

## Estimate formula
Audio hop (hub PUB -> agent callback): `t = zmq_ipc (~20-60 us) + decode (~20 us) + create_task (~15 us) + loop scheduling (event-loop busy-ness)` ~ **<0.2 ms** unloaded (guess); under loop contention, equals loop lag (needs measurement).
`request_frame`: `t = push_send(~50 us) + hub_dispatch_queue + memcpy(S) + msgpack_pack(S) + ipc_send(S) + recv(S) + unpack(S)`, with S = frame bytes. At ~5-10 GB/s memcpy and ~2-4 GB/s socket throughput: 1 MB ~ 0.5-1 ms; 3 MB (720p RGB) ~ 2-4 ms; 8 MB (1080p RGBA) ~ 5-12 ms (guess, **needs-measurement**). Add hub loop wait if busy.
Staleness: returned frame age <= `max_age_s` (2 s) plus request time; **frame may be up to 2 s old by construction** if the camera paused.
`LiveFrameSource.get` worst case: `timeout_s` 5 s wait for a signal + 1.0 s request timeout. Image capture: client camera + JPEG encode + upload: **needs measurement** (hundreds of ms to seconds), cap 10 s.

## Hidden latency / improvement candidates
1. **msgpack decode and `bytes()` of large `FRAME_DATA` on the shared event loop** (_processor.py:753-755, _codec.py:163-168) - unlike the file lane (decode in `asyncio.to_thread`, _processor.py:804). Move large-message decode to a thread or send frames via the existing SHM instead of re-encoding over ZMQ. Saves a 3-10 ms loop stall per image request on the voice process (jitter on VAD/TTS pacing).
2. **Hub serializes FRAME_REQUEST copy+publish with audio forwarding** (_hub.py:651-669, `_dispatch` is a single loop): a 8 MB frame copy + send delays every audio message behind it by the same ms. Hub-side thread offload or SHM-by-reference would remove it (service-side finding).
3. **Every VIDEO-subscribed endpoint receives every requested frame** (topic `video_data.{pid}.{track}`, hub publishes broadcast): N endpoints = N x copy/decode of each frame. Voice endpoint should not subscribe to VIDEO unless required (_transport.py:673). Saves N-1 decodes per request.
4. **`LiveFrameSource` accepts 2 s-old frames** (`max_age_s=2.0`): for "what am I looking at now" the image can lag by up to 2 s; reduce for accuracy (zero latency cost).
5. **One `create_task` per audio chunk / per frame signal** (_processor.py:1024-1042) plus fatal-on-error `_spawn` (1105-1113): task churn ~100-130/s/pid. Batch audio chunks.
6. **ZMQ SUB HWM default (1000)** on real-time lane means a stalled agent loop (e.g. during STT, which blocks the VAD frame loop, but the endpoint recv loop is separate) can still buffer ~10 s of audio before silent drops - a burst of stale audio arrives after a stall (head-of-line, stale speech). Set a smaller RCVHWM or drop-old policy for audio.
7. `request_frame` coalesces concurrent requests: parallel vision tools for the same camera share one transfer (good). No coalescing across tracks.
8. Subscription barrier probes at 20 ms retry (_processor.py:97): only at startup/join; negligible.

## What to measure live
- Audio hop: compare `AudioChunk.pts_us` (hub stamp) to `time.time_ns()` in an `on_audio` callback (a profiling endpoint can register `ep.on_audio` alongside the voice one). Same for `FrameSignal.pts_us` for video age.
- `request_frame` duration vs payload: time `await ep.request_frame(signal)` for several resolutions (640x480, 720p, 1080p) in the harness; record `FrameData.width/height/fmt/len(data)`.
- Event-loop lag during image requests: run a 10 ms ticker task and request frames concurrently while streaming mic audio; correlate ticker lateness with frame size.
- Image capture round trip: time `_ClientImageCaptureSource.capture`; `ImageCaptureData.data` length gives encoded size.
- Hub loop stalls: `Subscription probe` round trip time via `wait_for_subscriptions` while frames are being requested (probe echo goes through the hub dispatch loop, _processor.py:427-473): a live head-of-line detector.
- ZMQ drop detection: monotonic gaps in `AudioChunk.pts_us` (expected +10 ms steps).

## Confidence
Structure, timeouts, copy chain: **code-derived** (high; hub side read only at the dispatch). Throughput/ms numbers for copies: **guess** based on typical memory/socket bandwidth; **needs-measurement**. ZMQ default HWM claim: standard libzmq default (1000), not set in this file - confirm no process-wide override.
