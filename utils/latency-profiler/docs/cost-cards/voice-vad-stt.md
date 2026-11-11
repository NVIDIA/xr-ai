<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Voice input: VAD + batch STT / speaker-ASR (agent-sdk/xr-ai-voice/xr_ai_voice/_processors/vad_stt.py, _processors/speaker_stt.py, _speaker_client.py, utils/xr-ai-vad/xr_ai_vad/detector.py, agent-sdk/xr-ai-models/xr_ai_models/_speaker_stream.py)

> **Measured corrections (2026-10-22).** (1) The hypothesis that `VadDetector` accrues speech at about one third
> of real time with 10 ms frames is **confirmed**: a short utterance ("Stop.") never finalized at 10 ms frames and
> finalized at every frame size of 20 ms and above, and speech-start fired 0.89 s after onset at 10 ms against
> about 0.56 s at 32 ms and above. The production hub frame size is still inferred, not confirmed.
> (2) `silence_duration` 0.4 s split one long sentence into three utterances at 10 and 20 ms frames; 0.8 s kept
> it whole and finalized 0.76 s after the end of speech. (3) Speaker ASR end-of-speech to final is about 0.70 s,
> not the hop-based figure inferred for the speaker path elsewhere.

## Role in a turn
Head of the voice pipeline and the single largest fixed contributor to "end of user speech -> agent sees query". Everything from end-of-speech to `GatedQueryFrame` is serial: VAD hangover, then (optionally) a probe grace, then one batch STT call. Early partial probes run in the background and only matter for STOP/wake-chime latency, not for the normal query path. Two mutually exclusive backends: `VadSttProcessor` (local Silero + batch `STTService.transcribe`) or `_SpeakerSttProcessor` (streaming WebSocket to a speaker-ASR server that does its own VAD/STT).

## Critical-path calls
Backend A (`VadSttProcessor`, default):
1. Per 10 ms hub frame (`InputAudioRawFrame`), `await det.feed(...)` inside `process_frame` (vad_stt.py:282). CPU only: int16->float32, optional resample, Silero ONNX per 512-sample window (detector.py:164-190). Serial with every other audio frame of the pipeline.
2. Hangover: VAD finalizes only after `silent_s >= silence_duration` AND `speech_s >= min_speech` (detector.py:157-162). Pure wall-clock wait on audio.
3. `on_utterance` is called from inside `feed` -> inside `process_frame` (detector.py:218, vad_stt.py:205). First `await self._finish_probe_for_utterance(pid)` (vad_stt.py:208): if a probe STT is in flight, waits up to `_PARTIAL_PROBE_FINISH_GRACE_S`=0.15 s, else cancels probe (vad_stt.py:432-446).
4. Pushes `UserStoppedSpeakingFrame` (non-blocking queue put) (vad_stt.py:217).
5. `await self._stt.transcribe(audio_bytes, sample_rate)` (vad_stt.py:386 via `_transcribe` 358) - network call to the STT server, whole utterance incl. 320 ms pre-roll and trailing 0.8 s of silence. Serial; **blocks the VAD processor's frame loop while it runs** (see hidden latency).
6. `_on_final_transcript` callback (queues capture caption, non-blocking, runtime.py `_publish_transcript`), then `push_frame(TranscriptionFrame)` (vad_stt.py:244-250).

Background (parallel, `asyncio.create_task`): `_run_partial_probes` (vad_stt.py:192, 297). Each probe is a full batch STT call on the audio buffered so far + 120 ms zero padding (vad_stt.py:313). `on_speech_start` is itself spawned with `asyncio.create_task` (detector.py:140).

Backend B (`_SpeakerSttProcessor`): per 10 ms frame, a queue (`maxsize=200`) feeds one worker per pid that does `await client._feed(pid, audio, pts_us)` = one WebSocket send + wait-for-reply per frame (speaker_stt.py:134-136, _speaker_stream.py:62-66). Events (`speech_start`, `partial`, `transcript`, ...) come back in the reply and are turned into frames (speaker_stt.py:192-266). End-of-speech detection is server side.

## Fixed waits & timers
| Constant | Default | Where | Effect |
|---|---|---|---|
| `silence_duration` | 0.8 s | `VadConfig` vad_stt.py:71 | Hangover added to every turn (backend A) |
| `min_speech` | 0.15 s | vad_stt.py:74 | Min accumulated *classified-speech* time before finalize allowed (see 10 ms-frame caveat below) |
| `silero_threshold` | 0.5 | vad_stt.py:77 | |
| `_SILERO_WINDOW` | 512 samples (32 ms) | detector.py:37 | VAD decision granularity |
| `_PRE_ROLL_CHUNKS` | 10 chunks (320 ms only if chunks are 32 ms; with 10 ms hub frames it is only 100 ms) | detector.py:39 | Audio prepended to every utterance sent to STT |
| `_MAX_UTT_S` | 30 s | detector.py:38 | Force finalize |
| `stop_probe_after_s` | 0.25 s cadence | vad_stt.py:80 | First probe 0.25 s after speech-start event; <=0 disables |
| `_MAX_PARTIAL_PROBES` | 3 (1 if no partial handler) | vad_stt.py:52, 299 | Extra STT calls per utterance |
| `_PARTIAL_PROBE_TAIL_S` | 0.12 s zero pad | vad_stt.py:53, 313 | Added to every probe's audio |
| `_PARTIAL_PROBE_FINISH_GRACE_S` | 0.15 s | vad_stt.py:54, 440 | Max extra wait before final STT starts if a probe is in flight |
| Speaker backend: `silence_duration` | 0.6 s (server side) | utils/xr-ai-voicegate/_speaker.py:22 | Replaces VAD hangover |
| Speaker backend: `activity_threshold` | 0.7 | _speaker.py:21 | |
| Speaker backend: `max_utterance_s` | 30 s | _speaker.py:23 | |
| Speaker backend: `timeout_s` | 10 s per feed/open | _speaker.py:20, _speaker_stream.py:46,64 | Per-frame RPC timeout |
| Speaker per-pid queue | 200 frames (~2 s) and 64000 bytes | speaker_stt.py:115, 121 | Overflow => "enrollment reset" + 2.0 s `retry_after` blackout (speaker_stt.py:185) |
| Speaker frame limits | 16 kHz mono, <=32000 B/frame | speaker_stt.py:98-100 | |
| Speaker cleanup | `min(0.25, timeout_s)` | speaker_stt.py:150 | |
| Readiness poll | 5.0 s | _readiness.py:18 (startup only) | Startup, not per turn |
| Speaker availability HTTP | `min(2.0, timeout_s)` | _speaker_stream.py:24 | Startup only |

## Cost drivers
- Utterance audio length `A` (s): STT call scales with `A + 0.32 + 0.8` (pre-roll + hangover are sent to STT). Upload bytes = 32 kB/s (16 kHz int16): 10 s = 320 kB.
- STT server real-time factor (`stt_rtf`) and queueing: probes (up to 3 + final per utterance when a wake/conversation handler is bound, pipeline.py:98-100) hit the same server as the final call. Probe audio grows with each probe, so probe cost grows too.
- #participants: one `VadDetector` (and one Silero ONNX session) per pid, constructed lazily on first audio on the event loop (vad_stt.py:168-260, detector.py:91 `load_silero_vad`). Per-frame CPU is ~per-pid and runs on the single asyncio thread, shared with TTS/IPC.
- Speaker ASR: one WebSocket RPC per 10 ms frame per participant (100 RPC/s/pid), so cost is RTT-bound, not audio-length-bound.
- Frame rate: 100 frames/s/pid through pipecat queue, each with a Silero check, resample (if hub rate != 16 kHz) and an `asyncio.create_task` in the hub endpoint (_processor.py:1106).

## Estimate formula
Backend A, from end of user speech to `TranscriptionFrame` pushed:

`t_stt_path = silence_duration + vad_frame_quant + probe_grace + stt_final(A + 0.32 + silence_duration) + hop`

- `silence_duration` = 800 ms (config; the dominant floor).
- `vad_frame_quant` = 10-32 ms (hub frame 10 ms; Silero window 32 ms).
- `probe_grace` = 0 if no probe in flight; up to 150 ms if a probe STT is mid-flight at utterance end (likely for utterances 0.25-1.5 s long, or when the STT server is slow).
- `stt_final` = `stt_net_rtt + stt_rtf * (A + 1.12 s) + stt_queue`  -> **needs measurement** (`stt_rtf`, `stt_net_rtt`).
- `hop` = pipecat frame hops, <1 ms each (guess).

Best ~ 800 + 10 + 0 + stt (e.g. stt_rtf 0.05, A=2 s => ~160 ms) ~ **~1.0 s**; typical ~ 0.8 + 0.03 + 0.05 + stt(150-400 ms) ~ **1.0-1.3 s**; worst (probe grace + slow STT, long utterance) ~ 0.8 + 0.15 + (A+1.1)*rtf + queue, e.g. 10 s utterance at rtf 0.2 = ~3.3 s.

Backend B: `t = server silence (600 ms) + per_frame_RTT queueing + server final-ASR` -> 600 ms floor; remainder needs measurement.

Early STOP / wake path (not the query path): `t_stop = onset_to_speech_start + 0.25 * k + probe_stt(partial_audio) + gate regex` where `onset_to_speech_start` = time for `speech_s` to reach 0.15 s (see caveat), k = probe attempt (1..3). Best ~ 0.15 + 0.25 + probe_stt ~ 0.4 s + stt; with the 10 ms-frame accounting quirk below up to ~0.5 s + 0.25 + stt.

## Hidden latency / improvement candidates
1. **VAD frame loop is blocked during final STT** (vad_stt.py:282 -> detector.py:218 -> vad_stt.py:220): `on_utterance` is awaited inside `feed`, which is awaited inside `process_frame`. Until STT returns, no further audio frames are VAD-processed (they queue in pipecat's input queue). Consequences: speech-start of the next utterance/barge-in is detected only after STT finishes (+ catch-up). Saving: for follow-up speech/barge-in, up to `stt_final` (hundreds of ms). Fix: run `on_utterance` via `create_task` (the same way `on_speech_start` is spawned).
2. **Silero verdict only once per 512 samples** (detector.py:183-190): with the 10 ms/160-sample frames the hub delivers (livekit `rtc.AudioStream` default; _room_client.py:903), 2 of 3 chunks return `speech_prob = 0.0` and are counted as non-speech, so `speech_s` only accrues ~10 ms per ~32 ms of real speech. Effects (code-derived, **verify live**): `min_speech=0.15` is effectively ~0.45-0.5 s of voiced audio; very short utterances ("stop", "yes", "no") may never satisfy `speech_s >= min_speech` and then never finalize until later speech or the 30 s cap (detector.py:157-162); `on_speech_start` (hence probes, interrupts, wake chime) is delayed ~0.3 s. Fix: carry the last verdict across sub-window chunks or sum audio time on a positive verdict. Saving: up to ~300 ms on early probes, and removes a stuck-utterance failure mode.
3. **0.8 s hangover** is a fixed floor; a smarter end-of-turn (shorter hangover + semantic/ASR-endpoint confirmation or speculative STT on 0.3-0.4 s silence, cancel if speech resumes) could save 300-500 ms per turn.
4. **STT receives pre-roll + 0.8 s trailing silence** (vad_stt.py:206-224): trimming trailing silence before transcribe saves ~0.8 s of audio (STT time ~ `0.8*stt_rtf`, small for fast STT, but significant for batch Whisper-class models).
5. **Probe grace** up to 150 ms (vad_stt.py:440) serializes final STT behind an in-flight probe; better to start final STT immediately and ignore the probe result. Saving: 0-150 ms.
6. **Probes re-transcribe the whole buffer** (vad_stt.py:313): up to 3 extra STT calls/utterance when wake/conversation is configured (pipeline.py:98-105); they add GPU load and queueing on the STT server that can slow the final call. Fix: incremental/streaming STT or only probe when `could_match_magic_phrase`.
7. **Speaker ASR: 1 RPC per 10 ms frame, strictly serial** (speaker_stt.py:134-136, _speaker_stream.py:62-66). If mean RTT > 10 ms the queue grows to 200 frames (2 s) then the session is reset with a 2 s blackout (speaker_stt.py:121-123, 185). Batching 4-10 queued frames per send removes per-frame RTT from the critical path.
8. First audio from a new pid synchronously loads a Silero ONNX model on the event loop (detector.py:91): one-off stall (guess 20-200 ms, needs measurement) that also stalls TTS/IPC tasks.
9. `_detectors`/probe dicts are evicted on `ParticipantLeftFrame` (vad_stt.py:413) - no unbounded growth; probe buffer is unbounded within a 30 s utterance (30 s * 32 kB/s ~ 1 MB), `bytes(buf)` copy per probe (vad_stt.py:313).

## What to measure live
- `stt_rtf`, `stt_net_rtt`, `stt_queue` per call: existing nemo_relay scope `voice.stt` with `input.audio_duration_ms`, `metadata.mode` in {`final`,`partial-probe`} and `attempt` (vad_stt.py:373-391) gives duration and wall time per call without code changes.
- Loguru lines (parse log timestamps): `speech start pid=` (vad_stt.py:200), `early transcript probe fired ... latency_ms=` (332), `utterance finalize ... dur=` (211), `partial probe grace expired` (442), `gate decision ...` (gate.py) for the end of the chain. Speech-end-to-transcript = `utterance finalize` -> next `gate decision`/`voice input`.
- Frames carry `pts` (hub capture time, ns): `TranscriptionFrame.pts` = speech onset (vad_stt.py:243); wall-clock at `voice input` minus (`pts` + A) = true endpointing+STT latency.
- Event-loop stall probe: asyncio debug slow-callback logging, or a sidecar task logging `loop.time()` drift, to confirm item 1/8.
- Speaker backend: log RTT of `_SpeakerStream._feed` by wrapping the (private) `_make_speaker_stream` in the profiler harness, or watch for the "speaker ASR overloaded" warning (speaker_stt.py:122).
- Replay a known WAV through the hub at real time with `utils` tools and compare `VadDetector` finalize time to expected `speech_end + silence_duration` to verify item 2.

## Confidence
Constants, ordering, and serial structure: **code-derived** (high). `stt_*` terms: **needs-measurement**. Item 2 (10 ms-frame verdict accounting) is code-derived but rests on the assumption that the hub emits 10 ms frames (comment at speaker_stt.py:113 and livekit `AudioStream` default); treat as **hypothesis until measured**. ONNX load cost and per-frame CPU: **guess**.
