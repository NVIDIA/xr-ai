<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# STT / TTS adapters (agent-sdk/xr-ai-models/xr_ai_models/_openai_compat.py:832-1038, _riva_grpc.py, _speaker_stream.py, _factory.py:144-215, presets/parakeet_stt.py, pocket_tts.py, magpie_tts.py)

## Role in a turn
STT is the first term of the critical path (end-of-speech to transcript) and TTS the last (text to first audible audio). Both are on the user-perceived path; TTS streaming decides time-to-first-audio.

## Critical-path calls
### STT (OpenAI-compat `transcribe`, :858-885)
1. If `sample_rate` given: `_pcm_to_wav` (wave module, sync, in-memory copy) :873, :256-263.
2. `await client.post(/v1/audio/transcriptions, files=multipart, data={"response_format":"json"})` :881. Whole utterance uploaded and transcribed in ONE request (offline/batch, not streaming). multipart encoding by httpx is sync on loop.
3. `resp.json()["text"]` :885.
### STT (Riva gRPC `RivaSTT.transcribe`, _riva_grpc.py:139-180)
1. WAV parse if no sample_rate (:156-157).
2. `offline_recognize(..., future=True)`, `to_thread(call.result)` under `asyncio.timeout(timeout or 30)` :166-170 (offline, whole utterance).
### STT speaker stream (`_SpeakerStream`, _speaker_stream.py)
Websocket; `_open` connect + config send + identity recv (:509-517) bounded by `timeout_s`; `_feed` is lockstep: `send(audio)` then await the server events reply (:526-530), one round trip per audio chunk. No retry/backoff.
### TTS
* `OpenAICompatTTS.synthesize` (:938-957): `post /v1/audio/speech {input, response_format:"wav"}`; returns after full audio; whole `resp.content`.
* `_PocketTTS.stream` (:984-1038): streaming POST (`stream:true`, `response_format:"pcm"`), reads headers for sample rate (:1008), `aiter_bytes` (:1017), yields `_TTSChunk` aligned to whole samples (:1019-1022). Fallback when `x-audio-sample-rate` header absent: the streamed request is abandoned and `synthesize()` is called as a SECOND request (:1026-1030), then the WAV is parsed and yielded as one chunk.
* `RivaTTS._stream` (_riva_grpc.py:245-300): per chunk `create_task(to_thread(next, responses, None))` (:276) -> one thread hop + task per gRPC response; `synthesize()` (:216-243) is the same streaming RPC concatenated (batch RPC avoided because it segfaults some NIM builds, :232-235), so buffered synth has no latency advantage.

## Fixed waits & timers
| Constant | Default | Where |
|---|---|---|
| STT httpx timeout | 30.0 s | `OpenAICompatSTT.__init__` :844; preset parakeet_stt.py:9 |
| TTS httpx timeout | 30.0 s (Pocket preset 30.0; Magpie preset 60.0) | :924; presets/pocket_tts.py:10, magpie_tts.py:9 |
| Riva STT/TTS timeout | 30.0 s (STT: per `asyncio.timeout`; TTS: whole-stream deadline checked per read) | _riva_grpc.py:129, :206, :166, :266-279 |
| Riva channel ready probe | 3.0 s | _riva_grpc.py:92 (health only) |
| Riva TTS sample rate | 44100 Hz default | _riva_grpc.py:205 |
| Speaker stream health probe | min(2.0, timeout_s) | _speaker_stream.py:488 |
| Speaker stream open/feed/close | `timeout_s` each | _speaker_stream.py:510-531 |
| Health probe | 3.0 s | _openai_compat.py:106 |
| Retries | none | - |
| Pooling | httpx default (keepalive 5 s) | :855, :935 |
| TTS audio format | WAV default for synthesize; PCM s16 for stream | :941, :995 |

## Cost drivers
* STT: audio duration (upload size 32 KB/s at 16 kHz mono s16), server real-time factor; endpointing delay is NOT here (it lives in voice package).
* TTS: characters/tokens of text per call (how text is chunked by the voice aggregator decides time-to-first-audio), server RTF, whether streaming; Riva adds thread-hop per chunk.
* Speaker stream: number of chunks x RTT.

## Estimate formula
STT: `t_stt = t_wav_wrap(<1 ms) + t_upload(audio_bytes) + RTF_stt * audio_s + t_json`. With Parakeet-class models RTF is small (order 0.01-0.05, guess) so ~20-150 ms for a 3 s utterance: needs measurement.
TTS buffered: `t_tts_full = t_rtt + RTF_tts * audio_s + t_download`. TTS streaming first audio: `t_first_audio = t_rtt + t_model_first_chunk (+ c_hop per Riva chunk ~0.1-0.3 ms guess)`.
Plug in: server RTF and first-chunk time (needs measurement), text length per synthesize call.

## Hidden latency / improvement candidates
* Using `synthesize` (WAV) instead of `stream` makes first audio = full synthesis time: for a 3 s sentence at RTF 0.2 that is ~600 ms vs first-chunk ~100-200 ms (guess). Prefer the `_StreamingTTSService` protocol (`_protocols.py:553`) where available.
* `_PocketTTS` silent fallback to non-streaming doubles cost when the server lacks streaming (:1026-1030); detect at startup instead.
* STT offline: no partial transcripts; the speaker stream is the only incremental path.
* Riva one-thread-hop per chunk (:276): negligible (<1 ms/chunk) unless chunks are tiny.
* Speaker stream feed is lockstep (one RTT per chunk): chunk size sets min latency floor = RTT; larger chunks add buffering delay equal to chunk duration.

## What to measure live
* `make_stt(...).transcribe(pcm, sample_rate=16000)` for 1/3/10 s clips: fit `RTF`.
* `make_tts(...).synthesize(text)` vs `.stream(text)` (first chunk time, total, chunk sizes) for 20/80/200 chars.
* Pocket server: check header `x-audio-sample-rate` actually present (otherwise fallback doubles cost).
* Idle >5 s then first call (keepalive expiry).

## Confidence
Call structure/timeouts: code-derived. RTF numbers: needs measurement; my 0.01-0.2 RTF figures are guesses.
