<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Voice transport, pipeline plumbing and VoiceAgent bridge (agent-sdk/xr-ai-voice/xr_ai_voice/_transport.py (input half), _pipeline.py, _session.py, _readiness.py, _processors/io.py, _runtime.py)

## Role in a turn
Glue: hub IPC audio in -> pipecat frames, query hand-off to the runtime, and runtime responses back to the TTS processor. On the critical path twice per turn: (1) mic frame -> VAD (per-frame cost, constant) and (2) accepted query -> `UserQuery` published to the application agent, and assistant text -> `TextFrame`. Startup/readiness constants are listed so profiling can exclude them.

## Critical-path calls
Input, per 10 ms frame (100/s/pid):
1. Hub PUB -> `ProcessorEndpoint.run` recv loop: `await sub.recv_multipart()`, `msgpack` decode inline on the loop, `_dispatch` -> `asyncio.create_task(cb(chunk))` per audio chunk (_processor.py:743-755, 1037-1039, 1106).
2. `_on_hub_audio`: float32->int16, downmix, `scipy.signal.resample_poly` if rate != 16 kHz (_transport.py:161-187, 78-97), build `_TrackedInputAudioFrame`, `pts = chunk.pts_us*1000`, `await push_frame` into pipecat.
3. Frames pass `vad_stt -> voice_gate -> io -> streaming_tts -> output`, each a pipecat processor with its own queue/task (pipeline.py:107-114).
Query hand-off (after gate accepts):
4. `_VoiceIOProcessor._spawn_query` (io.py:166-183): if a response for the pid is in flight, cancel it (and, with `interrupt_on_supersede`, push an `InterruptionFrame`, which flushes hub audio, streaming_tts.py:464-477), then `create_task(_run_query)`.
5. `_run_query`: if a response was cancelled, `await asyncio.gather(interrupted_response)` (io.py:193-194) before delivering. Then **parallel** `asyncio.gather(input_sink(query), display(DataMessage voice.user-query))` (io.py:203-210). `input_sink` = `VoiceAgent._publish_input` -> `runtime.publish(UserQuery)` (runtime.py:459-473), which **awaits all subscribers of the topic** (see runtime-events card) - i.e. this task is alive for the whole subscriber handling.
Output: `VoiceAgent.output` (runtime.py:344) takes `_output_lock`, `enqueue_response` -> new response task (io.py:231-294) -> `TextFrame`s into `StreamingTtsProcessor` (see TTS card).
Text input (typed): `endpoint.on_data(_on_data)` -> `enqueue_query` (runtime.py:744-757) bypasses VAD/STT/gate (the only path with no STT cost).

## Fixed waits & timers
| Constant | Value | Where | Effect |
|---|---|---|---|
| Hub frame size | 10 ms (livekit `AudioStream` default; comment at speaker_stt.py:113) | _room_client.py:903 | 100 msgs/s/pid, floor on VAD granularity |
| Input sample rate to pipeline | 16 kHz mono | _transport.py:53-54 | resample cost if hub != 16 kHz |
| `_HUB_PUB` / `_HUB_PUSH` | `ipc:///tmp/xr_hub_pub` / `xr_hub_in` | _transport.py:50-51 | Unix-domain ZMQ, no TCP |
| `_ROSTER_HANDSHAKE_WAIT` | 0.1 s | _processor.py:705 | Startup roster catch-up |
| `_PROBE_RETRY_INTERVAL` / `_PROBE_TIMEOUT` | 0.02 s / 5.0 s | _processor.py:97-98 | Subscription barrier before announcing ready |
| `_STATUS_REANNOUNCE_INTERVAL_S` | 2.0 s | _session.py:28 | Background status republish (each does a subscription-confirmation check) |
| `wait_for_services` poll | 5.0 s | _readiness.py:18 | Startup (up to 5 s of dead time after a probe fails) |
| `_RETURN_AUDIO_*` | 0.12 s target / 0.32 s pre-roll / 40 ms | _transport.py:61-67 | See TTS card |
| `_TRANSCRIPT_CAPACITY` | 32 | runtime.py:33 | Drop-oldest caption queue |
| `response_capacity` | 32 | runtime.py:182 | Stream backpressure |
| `_OPEN_STREAM_CAPACITY`, `_CLOSED_STREAM_CAPACITY` | 1024 each | runtime.py:31-32 | Bounded |
| `idle_timeout_secs` | None (disabled) | _pipeline.py:42, 115-124 | No idle cancel |
| pipecat default `audio_in/out` queues | pipecat internal | - | not read |

## Cost drivers
- #participants x 100 frames/s: each frame = ZMQ recv + msgpack decode + task creation + numpy conversions + pipecat hop through `transport.input -> vad_stt` (2 queue hops) + Silero. All on one event loop thread.
- Hub endpoint filter is `AUDIO | DATA | VIDEO` (_transport.py:673): the voice process also receives and decodes **every video FrameSignal (30 fps) and every FRAME_DATA pixel payload that any other endpoint requests** for the same participant (hub publishes `video_data.{pid}.{track}` to all subscribers, services/device-io-hub `_hub.py:668-669`). A 1080p RGBA frame is ~8 MB: msgpack unpack + copy on the voice loop is several ms and stalls audio dispatch/VAD.
- Per-query: pydantic validation, two tasks, one display data message over IPC.

## Estimate formula
`t_input_per_frame = zmq_recv + msgpack(~2 KB) + task_spawn + int16/resample + 3 pipecat hops` ~ **0.1-0.4 ms CPU (guess)**; at 100 fps/pid that is 1-4% of one core per participant before Silero (~0.3-1 ms per 32 ms window on CPU ONNX, guess).
`t_query_handoff = pipecat hops (<1 ms each) + create_task + (cancel+gather of old response, if any) + runtime.publish overhead (~0.1-0.5 ms + per-subscriber task)` ~ **~1-3 ms** (guess) plus `gather(interrupted_response)` which waits for the old response task to unwind (the TTS side `aclose` of the producer generator; **needs measurement**, may be tens of ms or unbounded if the producer's `aclose` blocks on an LLM stream).
Mic -> hub latency (LiveKit + network): needs measurement (client-dependent, typically 30-150 ms).

## Hidden latency / improvement candidates
1. **Task per 10 ms audio chunk** (_processor.py:1037-1039, 1106) with no ordering guarantee beyond FIFO scheduling, and `_spawn` kills the whole process on any callback exception (os._exit(1), _processor.py:1109-1112). Batch 4-10 chunks per callback: reduces ~100 task creations/s/pid to 10-25.
2. **msgpack decode of large FRAME_DATA on the shared loop** (_processor.py:753-755): voice endpoint need not subscribe to VIDEO unless the sample uses frame signals through the same endpoint; use `Subscribe.AUDIO | Subscribe.DATA` or a separate endpoint for vision. Saves a multi-ms event-loop stall per image request (jitters VAD/TTS pacing) - 3-10 ms per 1080p request (guess).
3. **`await gather(interrupted_response)` before delivering the new query** (io.py:193-194): the new turn cannot start until the old response task fully unwinds, including `await _close_response(...)` -> `aclose()` of the producer (io.py:278). If the producer's async generator is mid-LLM-stream, `aclose` time is added to the user's barge-in latency. Fire the new query concurrently and let the old task unwind in the background.
4. **`runtime.publish` awaits subscribers** so `_run_query` stays alive for an entire app turn; combined with `_spawn_query` not serializing input tasks (io.py:178-183), two quick utterances produce two concurrent application turns. Not a latency bug, but results in contention; document for profiling (concurrency = number of overlapping turns).
5. **Voice output serializes per pid** (io.py:236-241): a second response waits for the first to finish *including its TTS playback only indirectly* - the response task ends when the *text* stream ends, not when audio ends; TTS keeps queuing audio. OK for latency but means "queue depth" in the profiler must include TTS sender backlog.
6. **`wait_for_services` polls every 5 s** (_readiness.py:18): startup can waste up to 5 s after a dependency becomes ready. Set `poll_interval` ~0.5 s. Startup only.
7. **Status reannounce every 2 s** runs `wait_for_subscriptions` per participant (_processor.py:624-660) - a subscription probe round trip at most every 2 s per pid if generations changed; negligible.
8. Per-frame `float32_to_int16` + `resample_poly(float64)` (_transport.py:93-96) allocates float64 arrays; for 48 kHz input ~480 samples: ~50-150 us. Could use a stateful `soxr`/`resample_poly` over longer chunks; saves CPU not latency.

## What to measure live
- Hub-to-pipeline frame latency: `AudioChunk.pts_us` (hub capture time, `chunk.pts_us`) vs `time.time_ns()` at `VadSttProcessor._handle_audio` (frame.pts is set from it, _transport.py:186). In the profiler harness wrap `DeviceIOHubInputTransport._on_hub_audio` or add a pipecat tap after `transport.input()`; no SDK edit needed.
- Event-loop health: asyncio slow-callback debug (`loop.slow_callback_duration`), or a 10 ms ticker task logging lateness. Needed to quantify items 1-2.
- Query hand-off: Loguru `voice input pid=` (io.py:168) -> `voice cancel pid= reason=` -> application subscriber start (runtime message metadata `timestamp_us`, runtime.py:325).
- `voice.user-query` data message (io.py:205) arrives at the client at ~ same time: client-side timestamp = end of STT+gate as the user sees it.
- Return path: `_capture.voice.tts` data message timestamps (_transport.py:609-621) vs client playback.
- CPU per participant: run 1, 2, 4 synthetic participants of silent audio and read process CPU.

## Confidence
Structure, constants: **code-derived** (high). Per-frame CPU and hand-off ms: **guess**. Mic-to-hub network: **needs-measurement**. The "10 ms frames" premise: strong (SDK comment + LiveKit default) but confirm by logging `chunk.samples`.
