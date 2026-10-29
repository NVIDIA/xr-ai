<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Semantic-turn prototype

The conversation sample supports opt-in semantic turn admission using its existing
Clef client, router, destinations, hub, and voice pipeline. Set
`semantic_turns.enabled: true` in
`agent-samples/conversation-example/yaml/conversation_example_worker.yaml`,
then launch `conversation_example` normally. Omitting the option or leaving it
false preserves the existing voice path. The controller and policy adapter live
in private modules of `conversation_example_worker`; there is no separate live
worker to start.

This is an experimental implementation. Scripted tests validate orchestration;
real Clef accuracy, acoustic behavior, tail latency, and target-GPU memory remain
qualification requirements. The prototype does not establish a new default
model setup or a measured GPU fit.

## Admission and sample policy

The controller never uses an acoustic VAD signal or silence duration to release
a query. After text is unchanged for the configured stability delay, one Clef
request jointly classifies readiness and relation to the preceding request.
ASR segment finality is independent of semantic completion. Speaker activity
still supports attribution and ASR segmentation; silence does not force a query.

- A filler-only candidate is ignored. Meaningful text is preserved; the prototype
  does not remove individual filler words from a meaningful sentence.
- Incomplete or uncertain input stays pending through arbitrary pauses, within
  text/session bounds. Unchanged silence causes no repeated model polling.
- A ready candidate with sufficient confidence is submitted. Corrections replace
  a segment snapshot instead of appending duplicate text. Empty replacements
  retract that segment’s pending words and invalidate an awaiting decision; a
  new empty segment preserves earlier incomplete input.
- An unspoken, side-effect-free request can be superseded by a combined query.
  After TTS commitment or started effects, only the new words become new intent;
  the previous request is supplied as explicitly labeled context. External
  actions are never automatically rolled back or replayed by the controller.
- An independent query queues by default; `independent: interrupt` selects
  interruption. A continuation of a queued request replaces only that queued
  request. A matching output-done notification releases the next query. The
  completed-response follow-up window starts at output completion, not at the
  original query time.
- A low-confidence relation holds input rather than guessing merge versus split.
  The prototype logs this state; it does not synthesize a clarification prompt.

A complete-sounding prefix followed by a long pause and qualifier is inherently
ambiguous. Stable-text delay and confidence trade response speed against early
commitment; they cannot guarantee instantaneous completion and zero premature
replies. The checked-in 150 ms delay and confidence threshold are experimental,
not calibrated defaults. ASR update cadence and shared Clef queueing also affect
latency.

## SDK boundary

Inject a sample-owned {py:class}`~xr_ai_voice.VoiceTurnPolicy` into
{py:class}`~xr_ai_voice.VoiceAgent`. The initial implementation requires required
speaker ASR, conversation mode, and `require_wake_phrase: false`. Unsupported
configurations fail before media allocation. Supply an interruption topic and
connect it to the owner of application work so cancellation is drained before
replacement admission. The normal speaker-service protocol is reused.

The SDK supplies {py:class}`~xr_ai_voice.TranscriptUpdate` after selected-speaker
and conversation authorization. Closed-state speech, diagnostic text, held
conversation-control prefixes, and acoustic speech-start/stop frames cannot
admit semantic queries. Deterministic authorized STOP and conversation-release
handling remain in the existing gate. A bare filler does not interrupt audio.
Meaningful but incomplete speech currently holds admission while existing output
continues; immediate audio-yield/resume on such input is not implemented.

Synchronous `update`, `output`, and `reset` notifications must be fast. The sample
uses revisions to invalidate an awaiting classification. The SDK serializes
`decide` calls across participants and coalesces changed sources with bounded
state. It rejects results from retired source lifecycles and applies ordered
{py:class}`~xr_ai_voice.VoiceTurnAction` values. A failed optional policy cannot
fall back to ordinary silence-based query dispatch.

The sample owns semantic state, confidence, combination, queueing, and effects.
The SDK owns source identity, admission, request retirement, output ordering,
worker cancellation and cleanup. Typed text bypasses and resets the pending
speech policy; its query receives its own identity on this opt-in path. No
Pipecat frame types are exposed to the application.

`UserQuery.request_id`, when present, is the identity a response must answer;
otherwise consumers use the runtime message ID. The router honors this field.
The conversation sample routes policy-owned identities through the same router
and response path used by ordinary input.

{py:class}`~xr_ai_voice.VoiceOutputState` reports `committed`, `done`, or
`interrupted` for the matching request. `committed` means the first text was
committed to TTS, not that the headset heard it. `done` means transport output
was drained, not a measured client playback acknowledgment. Treat commitment
conservatively as non-replayable.

The conversation sample allows unspoken general-conversation generation to be
combined with a continuation. Sample-owned callbacks mark notes and alert
operations before their answer path starts and mark camera acquisition before
requesting a frame. The synchronous pre-effect guard rejects replaced or reset speech requests even
if SDK interruption delivery is still pending. Marks invalidate in-flight merge
decisions. Notes use a separate bounded raw-input lookup so routing context and
earlier intent cannot be saved as participant-authored note text. Once an
effect starts, follow-ups carry only new intent plus labeled context, avoiding
automatic replay of the original action. Cancellation cannot undo a saved note,
camera request, or alert that has already started. Typed input bypasses and
resets the speech policy; marking a typed turn does not create speech state.

## Cumulative ASR and recovery

Use the speaker service from the same checkout for this experiment: earlier
service revisions suppress changed empty partials, so the policy cannot observe
those retractions. Held conversation-control words are delivered as empty
replacements for their segment, preserving invalidation without admitting the
control words.

The live adapter maps cumulative ASR snapshots to logical candidate segments.
After admission it suppresses an identical snapshot or punctuation-only suffix;
new appended words become a follow-up candidate. Pending incomplete segments
can span multiple ASR boundaries, including long pauses.

If ASR changes words already admitted, the adapter holds the correction and
logs that clarification is needed. It does not silently rerun a changed command.
A later source segment can continue interaction; automatic spoken repair for
this case is not implemented. The replay and classification corpus should be
extended with observed ASR corrections before acoustic qualification.

Enrollment release, service reset, track replacement and participant departure
revoke pending authority. Late query frames and late output markers are checked
against request/source identity. Shutdown closes admission, resets policy state,
and cancels/awaits the owned decision worker before closing model clients.

## Running the conversation sample

From `agent-samples/conversation-example/`, start the normal model stack for your
GPU profile, enable `semantic_turns.enabled` in the worker YAML, then start the
sample:

```bash
uv --config-file ../../uv.toml run --project model-servers conversation_example_model_servers --gpu-profile 96G_blackwell
uv --config-file ../../uv.toml run conversation_example
```

Choose the existing profile for your hardware. The normal hub, Clef router,
general conversation, observation, notes, and alerts all remain available. Open
the hub's client URL and say the configured conversation start phrase. The same
sample command supports ordinary typed queries and optional session recording.
Disable the option and restart to return to acoustic admission.

The worker YAML also holds stability, confidence, follow-up-window, timeout and
independent-query policy settings. These are experimental values, not measured
latency guarantees. Semantic admission shares Clef with routing, notes and alert
checks; contention must be measured under representative load.

The selected model profile is unchanged. This integration reuses Clef and adds
no draft or turn-detection checkpoint. All profile allocations, including
Cosmos for observation and alerts, remain part of the GPU budget. Other samples
without Clef need an additional service and a separate memory assessment.

## Replay and model evaluation

JSONL replay events are `open`, `text`, `poll`, `response`, and `reset`. Each
participant has a unique source epoch. Text events supply monotonically
increasing segment IDs, revisions and observation times, with the complete
current segment snapshot. Response events identify the request and its stage.
The replay prints admission actions rather than executing application work.

From the conversation sample directory, replay the checked-in scenarios using
the same policy settings as the worker:

```bash
uv --config-file ../../uv.toml run --project worker python eval/semantic_turns.py eval/semantic-turn-flows.jsonl --fixture
uv --config-file ../../uv.toml run --project worker python eval/semantic_turns.py eval/semantic-turn-classification.jsonl --classify
```

The second command requires real Clef; it does not start or download services.
The worker YAML supplies policy values even when its live opt-in is disabled.
Controller, adapter, effect-boundary and sample-composition tests are under
`tests/` and run in the normal CPU CI matrix.

The fixture backend is labeled `scripted-plumbing-only` and validates
orchestration. Removing `--fixture` calls real Clef through the existing typed
factory. The `--classify` mode scores independent labeled cases, reporting
choices, probability maps, confidence, latency and failures. It rejects fixture
mode because scripted labels cannot establish model accuracy.

Text, pending segments, sources and queued requests are bounded. Overflow holds
the entire candidate and requires reset rather than executing a truncated
prefix. The character bound does not replace Clef's token admission limit.
Timeouts, malformed responses and uncertain choices retain input without
repeatedly retrying unchanged silence. HTTP cancellation does not prove remote
GPU inference stopped; measure contention with other Clef consumers.

Before rollout, evaluate false filler suppression, incomplete speech, short
answers, quoted controls, negation, topic changes, corrections, adversarial
transcripts, accents and ASR errors. Measure premature releases, merge/split
errors, interruption-to-silence and end-of-speech-to-audio p50/p95. Then qualify
full mixed speech/vision load and the agreed per-GPU memory cap. Neither a
successful replay nor the existence of a live worker proves these properties.
