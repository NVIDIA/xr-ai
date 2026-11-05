<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Voice output: sentence batching + TTS + paced return audio (agent-sdk/xr-ai-voice/xr_ai_voice/_processors/streaming_tts.py, _audio.py, _transport.py (output half), _processors/io.py (response side))

## Role in a turn
Tail of the pipeline; it sets "first token of assistant text -> first audible sound". It is on the critical path for every spoken reply. Text arrives as `TextFrame`s (one per LLM chunk) from `_VoiceIOProcessor`, is cut into sentences, synthesized, converted to 20 ms frames, re-chunked to 40 ms by pipecat, paced to the hub at audio rate, then played by LiveKit with ~100 ms of client jitter buffer.

## Critical-path calls
1. Producer -> `VoiceAgent.output` (runtime.py:344) -> `_VoiceIOProcessor.enqueue_response` -> `_run_response` task (io.py:250-294). A new response waits for the previous response of the same pid to finish unless `interrupt=True` (io.py:236-241): per-participant serial speech.
2. `_consume_response` iterates the stream and `push_frame(TextFrame)` per chunk (io.py:309-315, 326-329). For streams, each chunk hop is `_ResponseStream.send/__anext__`, which creates 2 asyncio tasks + `asyncio.wait` per chunk on each side (runtime.py:85-130).
3. `StreamingTtsProcessor._handle_text` appends to `pending` and flushes complete sentences (streaming_tts.py:209-216, 264-283). Flush rule: regex `(?<=[.!?])\s+` (needs the whitespace after the punctuation, i.e. the *next* token) OR buffer ending in `[.!?]` + optional closing quote/paren + whitespace (streaming_tts.py:68-73, 279). No comma/length/time based flush. End-of-response flushes the tail (streaming_tts.py:218-230).
4. Dispatch (streaming_tts.py:285-301):
   - Batch TTS (`TTSService.synthesize`): `asyncio.create_task` **per sentence immediately** (parallel synth), tasks queued in order; sender loop `await task` in FIFO order (streaming_tts.py:344-360). First audio = full WAV of sentence 1 (not streaming).
   - Streaming TTS (`_StreamingTTSService.stream`): requests are put on the FIFO and **synthesized strictly one after another** (streaming_tts.py:336-343, 364-390); first audio = first PCM chunk of sentence 1.
5. First audio: `observe_tts_wav` (chime build, once) (streaming_tts.py:357, 376-399); `wav_to_output_frames` decodes the whole WAV into 20 ms int16 frames (`_audio.py:50-94`, O(audio length), CPU ~ms) (streaming_tts.py:409); `_CaptureTtsCaptionFrame` marker (caption data over IPC, async task).
6. Pipecat output: per-pid `MediaSender` resamples to 22,050 Hz and re-chunks to 40 ms (`audio_out_10ms_chunks=4`, pipecat base_transport.py:72) -> `write_audio_frame` (_transport.py:548-597).
7. `_send_paced_return_audio` (_transport.py:383-406): sleeps only when the estimated queued audio exceeds `_RETURN_AUDIO_TARGET_BUFFER_S`=0.12 s, then `await ep.send_return_audio(chunk)` (ZMQ PUSH, msgpack, float32 conversion `int16_to_float32`). So the first ~120 ms of audio is sent immediately; thereafter it flows at real-time rate.
8. Hub -> LiveKit return pipe (services/device-io-hub `_ReturnAudioPipe`, max buffer 3.0 s, config.py:11) -> client jitter buffer (~100 ms, documented in _processor.py:550-553).
9. Interrupt (barge-in/STOP): `_drain_on_interrupt` cancels sender task + synth tasks, then `await endpoint.flush_return_audio(pid)` (streaming_tts.py:447-477); pipecat sender `handle_interruptions` resets pacing deadline (_transport.py:512-521).

## Fixed waits & timers
| Constant | Value | Where | Effect |
|---|---|---|---|
| Sentence boundary | `[.!?]` + whitespace, or buffer ends with `[.!?]["')\]]*\s*` | streaming_tts.py:68-73 | First-audio waits for the whole first sentence of LLM text |
| Trailing-text flush | at `TextResponseEndFrame` | streaming_tts.py:218-230 | Unpunctuated tails wait for end of response |
| `_RETURN_AUDIO_TARGET_BUFFER_S` | 0.12 s | _transport.py:61 | Audio runs ahead of playout by up to 120 ms; also flush residue |
| `_RETURN_AUDIO_PREROLL_S` / chunk | 0.32 s of silence, 40 ms chunks | _transport.py:66-67, 295-338 | One-off after join; real audio waits behind it (`_wait_return_audio_preroll` 340-360) for ~0.28 s if the first reply comes right after joining |
| Pipecat output chunk | 40 ms (4 x 10 ms) | pipecat base_transport.py:72 | Min packetization delay, plus padded final chunk at `TTSStoppedFrame` |
| Output sample rate | 22,050 Hz (`TTS_NATIVE_SAMPLE_RATE`) | _transport.py:55, 682 | TTS output at other rates is resampled by pipecat |
| `_CHUNK_MS` | 20 ms | _audio.py:18 | WAV -> frame size |
| Sender queue | unbounded `asyncio.Queue()` | streaming_tts.py:197 | No backpressure; all sentences' synth tasks start at once (batch path) |
| `response_capacity` | 32 chunks per response stream | runtime.py:182, _ResponseStream queue | Producer blocks on `send` when 32 chunks are pending |
| `_OPEN_STREAM_CAPACITY` | 1024 | runtime.py:31 | |
| Return-audio hub buffer | 3.0 s max (hub config) | services/.../livekit/config.py:11 | Faulty/bursty producer drops past 3 s |
| Client jitter buffer | ~100 ms | _processor.py:553 | Floor on stop-to-silence |
| pipecat `audio_out_end_silence_secs` | 2 s (only on EndFrame/shutdown) | pipecat base_transport.py:75 | Shutdown only |
| Stop ack text | "Okay, I will stop." (full TTS each time) | voice_gate.py:55 | See hidden latency |

## Cost drivers
- Text length per sentence `L` and TTS server speed: `tts_first_chunk(L)` (streaming) or `tts_full(L)` ~ `tts_rtf * audio_s(L)` (batch). Batch first-audio scales with the **length of sentence 1** (~15 chars/s of speech: 80-char sentence ~ 5 s audio).
- LLM token rate: first audio cannot start before the first sentence terminator arrives, so `first_sentence_tokens / llm_tps` is on the path.
- #sentences: batch path fires all synths concurrently (GPU contention can slow sentence 1); streaming path serializes them.
- Audio length: per-WAV decode + int16/float32 conversions (~ms per second of audio, guess), 25 IPC messages/s/pid at 40 ms chunks (msgpack + zmq).
- Per-token runtime overhead: `runtime.publish` per chunk (pydantic validate, deep copy, uuid, task per subscriber, gather) + `_ResponseStream` 4 task creations per chunk. Rough guess 0.2-0.5 ms/chunk; negligible at <=100 tok/s but visible on a busy loop.
- #participants: independent state/sender per pid, but one event loop and one TTS server.

## Estimate formula
`t_first_audio = t_first_text + (n_tok_sentence1 / llm_tps) + tts_first(L1) + 40 ms (chunk) + pace(0) + ipc_hub_livekit + jitter(~100 ms)`

- Streaming TTS: `tts_first(L1)` = time to first PCM chunk (needs measurement).
- Batch TTS: `tts_first(L1)` = `tts_net_rtt + tts_rtf * audio_s(L1)` (needs measurement), plus WAV decode ~ ms.
- Fixed parts derivable from code: 40 ms chunk + <=120 ms pre-send reserve (not a delay: it is sent immediately) + ~100 ms client jitter ~ **~140-150 ms** between "TTS has audio" and "user hears it", plus network/LiveKit (needs measurement, typically 20-80 ms).
- Typical sentence-1 wait with LLM at 50 tok/s and a 12-token first sentence: ~240 ms of generation on top of LLM TTFT.

Stop path: `t_stop_silence = (detect STOP: vad_stt probe path) + cancel/teardown (<10 ms) + flush IPC (~1-3 ms) + client jitter (~100 ms)`. Stop *acknowledgement* additionally = `tts("Okay, I will stop.")` (voice_gate.py:403 -> streaming path) = `tts_first/tts_full` of a 3-word string (needs measurement; likely 100-500 ms).

Gap between sentences: batch path hides it (parallel synth) as long as `tts_rtf < 1`; streaming path hides it only if sentence N+1's `tts_first` is shorter than the audio still buffered ahead (<=120 ms in transport + whatever the hub holds; pipecat queue is unbounded and not rate-limited, so previously generated audio is already queued ahead, normally several seconds).

## Hidden latency / improvement candidates
1. **No sub-sentence flush** (streaming_tts.py:68-73): the first sentence is the biggest controllable term. Flushing the first chunk at a comma/clause or after N words (and sentences thereafter normal) saves `(tokens_to_first_sentence_end - tokens_to_first_clause) / llm_tps`, typically 100-500 ms. Alternatively prompt the LLM to open with a short sentence.
2. **Sentence flush waits for the next token** when the token stream emits "." and " " separately and the regex needs the whitespace; the trailing-regex path (279) covers `.` at buffer end, so effectively no wait, but it also means abbreviations/decimals ("3.5", "Dr.") flush early as separate sentences, which can fragment prosody (quality, not latency).
3. **Streaming TTS requests are serialized** (streaming_tts.py:336-343): sentence N+1 synth does not start until sentence N's stream finished. Starting N+1's synthesis speculatively (bounded to 1-2 ahead) would hide `tts_first` gaps; saving: audible inter-sentence gaps of `tts_first` (100-400 ms) when audio buffered ahead is thin.
4. **Batch TTS fires all sentence tasks at once with an unbounded queue** (streaming_tts.py:296-301): a long reply floods the TTS server; sentence 1 can slow down. Bound concurrency to 2-3 to protect `first_audio`.
5. **Stop acknowledgement re-synthesizes a fixed phrase every time** (voice_gate.py:55,403): cache the PCM at startup after the first TTS (like the chime) -> saves a full TTS call, 100-500 ms, on the STOP path.
6. **Chime is a no-op until the first TTS WAV has been observed** (gate.py:328-330, 375-391): the very first wake phrase after startup gets no chime. Prewarm TTS at startup.
7. **Per-chunk 4-task churn in `_ResponseStream`** (runtime.py:85-130) and `publish` overhead per token (runtime.py:184-238): coalesce tokens upstream (e.g. 2-4 tokens) or use a `Queue` with `get` and a sentinel instead of racing two tasks; saves ~0.1-0.3 ms/token of loop time (guess), which matters under multi-participant load.
8. `_observe_pcm` builds a WAV in memory for the first chunk of **every** streamed sentence even after the chime exists (streaming_tts.py:376, 392-399; the check is after the build): trivial CPU (~50 us), skip when `voice_gate._chime_wav` is set.
9. `wav_to_output_frames` decodes the whole WAV before pushing the first frame (_audio.py:77-94, streaming_tts.py:409): for a 10 s sentence ~ 10 ms; push incrementally to remove.
10. The 3 s hub buffer plus the pipecat queue being unbounded means a barge-in flush must drop up to the 120 ms transport reserve + hub pipe; keep `_RETURN_AUDIO_TARGET_BUFFER_S` small for snappy stops (trade-off: underruns).

## What to measure live
- nemo_relay scope `voice.tts` (streaming_tts.py:312, 367) with `input.text`: start-to-end per sentence = `tts_full`; for streaming, add a timestamp at first chunk by wrapping `_StreamingTTSService.stream` in the harness (the scope itself spans the whole stream).
- Loguru: `tts sentence dispatch pid= len=` (streaming_tts.py:286) = sentence-ready time; `data text echo` (247); `tts sender drained pid= queue_len=` (475); `hub return-audio flushed` (456).
- `_CaptureTtsCaptionFrame` -> data topic `_capture.voice.tts` (CAPTURE_TTS_TOPIC) is published with `pts_us = time of the first paced chunk send` for each sentence (_transport.py:583-596, 609-621): subscribe to it to get first-audio-send time per sentence without touching SDK code.
- Compare `AssistantResponseEndFrame` / first `TextFrame` time (log at `voice input`/first `tts sentence dispatch`) with the `_capture.voice.tts` timestamp for `t_first_audio_send`; add client-side first-audible timestamp from the client app to get network+jitter.
- Probe the pacing queue depth: `_return_audio_deadline_s` (private) - or log `send_return_audio` inter-arrival at the hub.
- Interrupt latency: time from `InterruptionFrame` log (`voice cancel pid= reason=interruption`, io.py:120) to hub `flush` handling.

## Confidence
Ordering, constants, and serialization properties: **code-derived** (high). `tts_*` and network/LiveKit/jitter terms: **needs-measurement** (jitter ~100 ms is the SDK's own documented figure). Per-token overhead estimates: **guess**. Whether streaming TTS adapters emit chunks fast enough to hide gaps depends on the adapter in xr_ai_models (not read).
