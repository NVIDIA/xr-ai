<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Early speech endpoints and strict filler filtering

The conversation sample has an optional voice mode with two responsibilities:
release clearly complete speech earlier, and suppress unmistakable filler-only
input. The existing speaker-ASR/VAD endpoint remains authoritative. Clef can
accelerate admission; uncertainty, low scores, errors and timeouts cannot veto
ordinary finalization.

Enable `semantic_turns.enabled` in
`agent-samples/conversation-example/yaml/conversation_example_worker.yaml` and
launch the sample normally from `agent-samples/conversation-example/`:

```bash
uv --config-file ../../uv.toml run conversation_example
```

Start the existing model stack first, following the sample README. The same hub,
router, conversation, observation, notes and alerts remain available. The option
is disabled in the checked-in configuration. Existing clients and typed input
continue to use the normal sample paths.

## Admission behavior

After partial transcript text stabilizes, Clef answers one binary question:
has the speaker already expressed a complete routable intent? A sufficiently
confident yes can release input before the normal endpoint. Any other result
waits for that endpoint. Early advice does not rewrite speech.

A native final snapshot releases its unsubmitted words without asking or waiting
for Clef. The SDK cancels an awaiting optional decision, prioritizes final input,
and skips the stable-text delay. Final submissions are retained independently
of newer partial snapshots, so a new segment cannot overwrite a completed one.
Already released prefixes are deduplicated when the matching final arrives.
Appended words become ordinary new input.

Filler filtering is deliberately narrow. Only input composed entirely of the
vocalizations `um`, `umm`, `uh`, `uhh` and `erm`, separated by whitespace or
punctuation, is suppressed. `yes`, `no`, `okay`, `hmm`, quoted words, and meaningful
sentences containing hesitations pass through. A model's broad interpretation
of filler cannot discard ordinary final input. Filler alone does not admit a
query or interrupt the response.

Routing, interruptions and quick follow-ups use the normal conversation sample
behavior. This mode does not perform semantic merge/split or follow-up queue
classification. The ordinary endpoint continues to determine how long pauses
are tolerated.

## Tuning and feedback

The worker YAML owns the experimental early-advice settings:

- `stable_s`: stable-text delay before asking Clef for early completion.
- `min_confidence`: threshold for a positive early release only.
- `decision_timeout_s`: deadline for optional advice; final input preempts it.
- `max_chars`: optional early-model input bound; larger final speech still passes.
- `max_sessions`: bounded selected-speaker state.

The checked-in delay is 150 ms, early threshold 0.85, and model deadline 300 ms.
These are experimental operating settings, not calibrated confidence or audio
latency guarantees. Shared Clef contention and ASR partial cadence affect the
possible benefit. Tune premature early releases and latency on real speech;
missed early opportunities fall back to the existing endpoint.

The worker startup log identifies `early-advice-and-strict-fillers` mode.
Admission logs distinguish `action=early_admit`, `action=vad_admit` and
`action=filler_drop`. Debug logs include the binary choice, score and early
threshold. Enable terminal debug output with:

```bash
XR_AI_VERBOSE=1 uv --config-file ../../uv.toml run conversation_example
```

Accepted queries also emit `voice.user-query`; `semantic:` request IDs correlate
them with routed responses in Relay. The sample sends an additional message on
that existing client topic when the query reaches the runtime:

- `User [Clef]: …` identifies a positive early-completion trigger.
- `User [VAD]: …` identifies ordinary ASR/VAD finalization.

The original query message remains, so the source-labelled entry appears beside
it in the web or XR client's message list. Provenance is correlated by request
identity, not inferred from the text or arrival time. It is display-only: the
query sent to the router and models is unchanged. Typed queries and suppressed
fillers produce no source-labelled entry. Diagnostic delivery is best-effort;
a failure cannot fail an admitted query. No public SDK API or client protocol
change is needed. Raw transcript captions alone do not prove query admission.

## SDK and resource ownership

The sample owns its private advisory policy and reuses the existing Clef client.
The opt-in {py:class}`~xr_ai_voice.VoiceTurnPolicy` hook receives authorized
{py:class}`~xr_ai_voice.TranscriptUpdate` values. Initial support requires required
speaker ASR, conversation mode, and no per-query wake requirement. Normal acoustic activity frames, deterministic
STOP and conversation controls remain in the gate. Held controls retract pending
text through empty snapshots without exposing control words as queries.

The SDK owns source authority, request correlation, scheduling, cancellation and
cleanup. An ASR final preempts an awaited `decide` call; policies must propagate
cancellation. The sample's final path performs no model work. Participant or
speaker reset revokes pending input and retained finals. Typed input resets and
bypasses speech advice. No public framework frames are exposed.

Each source retains only its latest partial plus at most 64 undelivered native
final actions. Exceeding that delivery capacity raises an explicit overload
error and resets that source; input is never silently truncated. HTTP
cancellation does not prove remote GPU inference stopped. The existing model
profile is unchanged and no extra model checkpoint is loaded. Extra inference
traffic, latency and peak memory still need measurement under mixed load.

## Replay and real-model checks

From the conversation sample directory:

```bash
uv --config-file ../../uv.toml run --project worker python eval/semantic_turns.py eval/semantic-turn-flows.jsonl --fixture
uv --config-file ../../uv.toml run --project worker python eval/semantic_turns.py eval/semantic-turn-classification.jsonl --classify
```

Replay uses authorized snapshot events and finalization, including low early
scores followed by ordinary admission. Fixture mode is labeled
`scripted-plumbing-only`. The real-model command uses the configured router
endpoint and does not start or download models. It reports actual early and
final submissions, choices, scores, model errors, elapsed times and per-case
behavior. It checks that meaningful input passes exactly once and incomplete
prefixes do not release early. Scores alone do not establish admission.

Controller/adapter, real framework queue, sample composition and failure-path
tests run under the normal `tests/` CPU CI discovery. Live model tests remain
separate from hardware and acoustic qualification. If ASR rewrites words after
an early release, the sample logs the correction instead of replaying the
already admitted command. Measure such corrections and premature releases
before rollout. Use a matching speaker-service checkout to receive changed
empty partial snapshots.
