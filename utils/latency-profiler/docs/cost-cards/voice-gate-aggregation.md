<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Voice gate + aggregation (agent-sdk/xr-ai-voice/xr_ai_voice/_processors/voice_gate.py, utils/xr-ai-voicegate/xr_ai_voicegate/{gate,config,_conversation,_chime,_phrases}.py, agent-sdk/xr-ai-voice/xr_ai_voice/_aggregation.py)

## Role in a turn
Two unrelated pieces with the same theme (policy between STT and the agent / between agents and TTS).
- **Voice gate** sits between STT and `_VoiceIOProcessor`. It decides query vs STOP vs drop (wake phrase, follow-up window, conversation start/stop phrases). Pure in-process string logic; adds ~no latency to an accepted query, but gates *whether* a turn starts and adds spoken side effects (chime, greeting, stop-ack TTS).
- **`VoiceAggregationAgent`** is optional, between concurrent producer agents (`voice.contribution` topic) and `voice.output`. It serializes speech per participant and coalesces simultaneous updates. It is on the critical path for the *producer-to-speech* leg only when used.

## Critical-path calls
Gate (per final transcript, all in-process, all awaited serially on the pipecat task):
1. `TranscriptionFrame` -> `VoiceGateProcessor._handle_transcription` (voice_gate.py:253). Modes: always_on (STOP regex then `_on_gate_query`), conversation controls (`_ControlMatcher` word match, voice_gate.py:277-318), or `VoiceGate.feed` ladder (gate.py:206-314). Regex/string ops only: microseconds.
2. If fresh wake match and the early probe did not already chime: `await gate.play_chime(pid)` -> `play_wav` decodes a 250 ms WAV to 12 x 20 ms frames and pushes them (voice_gate.py:145-168, gate.py:318-336). The chime is **only available after a real TTS WAV has been observed** (gate.py:328-330, 375-391).
3. `push_frame(GatedQueryFrame)` (voice_gate.py:383) -> `_VoiceIOProcessor._spawn_query` (io.py:166).
4. STOP: `InterruptionFrame` then `_emit_text_response("Okay, I will stop.")` -> goes through the **full TTS path** (voice_gate.py:398-403).
5. Participant join greeting is spoken through TTS (voice_gate.py:421-440).

Aggregation (per participant, a single sequential task, aggregation.py:197-240):
1. `contribute()` is a runtime subscriber: enqueue only, returns immediately (aggregation.py:128-166).
2. Finite contribution: `asyncio.sleep(coalesce_window_s)` then batch pending (aggregation.py:367-401). Interrupt contributions skip the sleep.
3. Single finite item: publish directly to `voice.output` (aggregation.py:557-558, 624-628). Multiple items: `_rewrite` LLM call (`llm.chat`, `temperature=0`, `max_tokens=192`, `enable_thinking=False`) racing a `_wait_for_interrupt` task (aggregation.py:562-600, 714-760), falling back to joined text on timeout/error.
4. After publishing, `_hold_output` blocks the participant's loop for the *estimated spoken duration* (`words*60/speech_rate_wpm`, clamped 0.4-30 s) or until an interrupt arrives (aggregation.py:630-640, 642-657, 702-708). Streams (`response_id` set, non-final) are forwarded chunk by chunk without the coalesce window (aggregation.py:403-480) and hold after the final chunk.

## Fixed waits & timers
| Constant | Default | Where | Effect |
|---|---|---|---|
| `followup_grace_s` | 5.0 s | utils/xr-ai-voicegate/config.py:50 | After a bare wake phrase, next utterance is accepted for 5 s (gate.py:283) |
| `phrase_window_s` | 6.0 s | _conversation.py:23 | Max time to assemble a split start/stop control phrase |
| Chime length | 250 ms (two tones, exp decay) | _chime.py:30 | Audio, not a wait; plays in front of/with TTS |
| Chime sample-rate source | first observed TTS WAV | gate.py:375-391 | No chime before first TTS |
| `_STOP_ACK_TEXT` | "Okay, I will stop." | voice_gate.py:55 | Costs one TTS call per stop |
| Early wake chime | on partial probe; needs probe at 0.25 s cadence, <=3 probes | vad_stt.py:80, 52; voice_gate.py:122-141 | Wake ack ~0.25-0.75 s + probe STT after speech start |
| `coalesce_window_s` | 0.15 s | _aggregation.py:76 | Added to **every** lone finite contribution |
| `queue_capacity` | 64 per pid | :77 | Oldest non-interrupt dropped on overflow |
| `max_batch_size` | 8 | :78 | |
| `max_tokens` | 192 | :79 | Bounds rewrite decode time |
| `rewrite_timeout_s` | 5.0 s | :85 | Upper bound of rewrite; fallback = concatenation |
| `stream_idle_timeout_s` | 15 s | :80 | Stream considered dead after 15 s silence |
| `participant_idle_timeout_s` | 60 s | :81 | Worker task exit after idle |
| `speech_rate_wpm` | 150 | :82 | Playback reservation = words/2.5 per s |
| `minimum_playback_s` / `maximum_playback_s` | 0.4 / 30 | :83-84 | Clamp on reservation |
| Pipeline STOP policy | regex `STOP_RE` closed grammar | _phrases.py (not a wait) | |

## Cost drivers
- Gate: transcript length (regexes), #participants (dict per pid). Negligible.
- Aggregation: #concurrent producers per participant (batch size 2-8 triggers the LLM rewrite), length of contributions (rewrite prompt/decode up to 192 tokens), estimated speech duration of the previous item (serialization wait), queue depth (<=64).
- Spoken side effects: each greeting / stop-ack / listening announcement is a full TTS request.

## Estimate formula
Gate: `t_gate = t_regex (~0.05 ms) + [chime: wav_decode ~1 ms]` ~ **< 2 ms** (code-derived; wake chime audible after `t_probe_path` or after final transcript if no early probe).

Aggregation, lone finite response: `t_agg = coalesce_window (150 ms) + publish (~0.1-0.5 ms)`.
Coalesced batch: `t_agg = 150 ms + llm_chat(prompt_tokens ~100-400, <=192 out) <= 5 s`; typical `llm_ttft + out_tokens/tps` = **needs measurement** (say 0.3-1.5 s on a small local LLM; guess). On timeout: 150 ms + 5 s then fallback text.
Stream: `t_agg ~ 0` per chunk (just two publishes: input subscriber hop + output hop).
Serialization delay for the *next* contribution from the same participant: `max(0, playback_estimate(prev_text) - elapsed)` where `playback_estimate = clamp(words*0.4 s, 0.4, 30)`. E.g. a 40-word reply reserves 16 s; a new non-urgent update waits up to that long unless `interrupt=True`.

## Hidden latency / improvement candidates
1. **Unconditional 150 ms coalesce sleep on every lone finite contribution** (aggregation.py:374-375). Saves up to 150 ms per update if skipped when no other producer is active (e.g. only sleep if another contribution arrived within the last N ms, or sleep with early wake on arrival).
2. **Playback reservation is a heuristic clock, not feedback from actual playback** (aggregation.py:702-708): words/150 wpm can over- or under-estimate real TTS duration (and ignores `tts` speed and the queue). Over-estimation delays later non-urgent updates by seconds; under-estimation leaves the voice IO queue to serialize (io.py:236-241). Use `TTSStoppedFrame`/response-end feedback. Potential savings: seconds of idle voice time in multi-agent samples.
3. **Rewrite LLM call is serial, blocking the participant's speech queue** for up to 5 s (aggregation.py:562-600). Overlap by speaking the first item immediately and only coalescing the rest, or a faster/smaller model. Saving ~0.3-2 s on batched turns.
4. **Stop acknowledgement/greeting are TTS round-trips** (voice_gate.py:398-403, 421-440): pre-synthesize and cache; saves 100-500 ms on STOP (needs `tts_first` measurement).
5. **Early wake chime depends on VAD speech-start + probe STT** (voice_gate.py:122-141): a streaming keyword spotter would cut wake ack from ~0.5-1 s to ~0.2 s; today the chime can arrive after the user has already started the command.
6. **Chime unavailable before first TTS** (gate.py:328-330): prewarm TTS.
7. Gate has no per-pid serialization (gate.py:11-14, 209-212): overlapping `feed` calls can race the follow-up window; correctness more than latency.
8. Aggregation `_wait_for_interrupt`/`state.changed` loops use `Event.clear()/wait()` (cheap); unbounded growth not present (queue_capacity 64, discarded_streams pruned).

## What to measure live
- Gate decisions: Loguru `gate decision pid= kind=` (gate.py:219-311), `chime fire pid= early=` (voice_gate.py:412), `stop ack emit` (voice_gate.py:399), `greeting emit` (439). Time deltas between `utterance finalize` (vad_stt), `gate decision`, and `voice input` (io.py:168) give gate cost (expect ~0 ms).
- Speech side effects: `tts sentence dispatch` immediately after `stop ack emit` -> then `_capture.voice.tts` data message timestamp gives stop-ack first-audio time.
- Aggregation: subscribe to both `voice.contribution` and `voice.output` topics (both `telemetry="none"`, so use `MessageMetadata.timestamp_us` and `parent_message_id`/`correlation_id` on delivery, runtime.py:306-326) and diff timestamps: contribution->output delay = coalesce + rewrite; `VoiceAggregationAgent._rewrite` can be wrapped in the harness to time `llm.chat`. Warning logs: `dropped ... voice contribution`, `voice contribution stream timed out`, `rewrite failed` (aggregation.py:272, 425, 748).
- `LLMService.chat` timing via whatever nemo_relay scope the LLM adapter emits (not in this scope).

## Confidence
Gate logic and constants: **code-derived** (high); latency ~0. Aggregation constants: **code-derived**; rewrite latency, playback-estimate error vs real TTS duration: **needs-measurement**. Early-wake-chime timing relies on the vad_stt probe cards' assumptions.
