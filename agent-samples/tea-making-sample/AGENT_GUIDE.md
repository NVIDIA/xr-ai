<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Agentic development guide

This file is the continuity contract for future human-test iterations. Update
it whenever the execution model, YAML schema, prompt budget, or event vocabulary
changes.

## Invariants

1. Task behavior lives in `yaml/workflow.yaml`; Python contains no tea-step
   switch or tool-name routing branch.
2. Every external observation is a NAT function selected by `trigger.function`.
3. Every state mutation goes through `workflow__commit`.
4. Every LLM loop is a built-in NAT tool-calling agent. Do not add a hand-rolled
   `while tool_calls` loop.
5. Observation agents can write only the active step's `writes`. The
   deterministic dispatcher invokes exactly one foreground agent per voice
   turn. Every root function declares inline, foreground, or background effect.
   Quick tools are read-only; only lifecycle NAT tools may change application
   ownership or tea state.
6. No conversation transcript is included. Add memory only for a named case
   that cannot be solved with the current question and step projection.
7. Recover only errors that a later frame or corrected model call can repair.
   Let configuration, programming, and service failures retain their traceback.
8. Package brewing values outrank RAG. Retrieval is step-scoped through the
   exact `rag_lookup` NAT function, and its source is committed with the values.
   Never retrieve without a specific visible tea name; return at most two
   passages to the small model. RAG supplies brewing values, never tea identity.
9. Step completion never transitions. Observation continues homogeneously;
   only the active tea foreground agent may advance after an explicit user voice request.
   Imperative next, continue, advance, and skip select the advance function;
   questions containing those words remain step questions.
10. VLM triggers return plain-text captions from focus-only questions. A
    YAML evidence gate, not the observation agent's interpretation alone,
    authorizes visual completion after repeated matching frames.
11. Sessions are ephemeral: first join and app-start roster replay reset to
    root, stop background state, and reset tea. Participant leave flushes and
    discards state; duplicate roster events are idempotent.
12. Foreground management is literal. Root input goes directly to root; tea
    input goes directly to the current tea-step variant. No LLM delegates to
    another LLM. Root tools are unavailable while tea owns foreground.
13. Model-visible vision requests contain only a question. `current_view`
    injects participant identity from invocation scope.
14. User-facing values are natural speech. The shared rule forbids Markdown,
    lists, formatting marks, code syntax, and internal names while spelling out
    shorthand and units. Keep state numeric and use message render filters
    instead of step-specific display fields or preformatted state.
15. Treat prompt failures as prompt-tuning work first. Confirm before changing
    evidence thresholds, state semantics, transitions, or other runtime policy.
16. Automatic heating observation records only `heating_started`; it never
    tracks readiness or compares temperatures. On a user hot-enough question,
    the voice agent passes the fresh number and unit to `temperature__verify`.
    The tool reads the target from session state, performs normalization and
    comparison, and never changes workflow state.
17. Observation agents receive `already_complete` as prior status plus a
    generated contract containing writable field meanings and the YAML
    completion condition. Prior status is never copied into writable state.
18. Every observation turn calls `workflow__commit` exactly once. Completed or
    unsupported observations commit empty updates and message; they never emit
    free-form state or force a readiness field.
19. Prompts state both positive evidence for each readiness write and its
    negative stop condition. Identification commits all resolved fields and
    `tea_ready` atomically; unrelated retrieval text never supplies a value.
20. Observation `message` briefly announces a real, non-completing state
    change. The store deterministically discards messages on no-change,
    repeated, and completion commits; this rule stays generic, not step YAML.
21. Tea identification is literal OCR, not visual classification. The vision
    focus excludes artwork and supplied words; RAG may provide brewing values
    only when its passage contains the variety already visible in the caption.
22. Launching requires explicit model, voice, and TTS modes. The orchestrator
    writes temporary model, worker, application, and RAG references from those
    selections so their profiles cannot diverge; never restore an implicit
    launch default.
23. Continuous Omni captions disable reasoning and have a small output cap.
    Omni agent calls also disable reasoning so their token budget produces a
    tool call. A bounded caption timeout is recoverable and must leave the
    observation loop active; other model failures retain their traceback.
24. Identification has no draft state. Its agent ignores prior incomplete
    identity, derives the name from the fresh front-label caption, and either
    commits every final field with `tea_ready: true` or commits no updates.
25. Starting steeping requires two separately visible facts in one vessel:
    water and tea-water contact. A tea item alone or an obscured water surface
    never starts the clock.
26. Wake-word and always-on behavior are separate voice-gate profiles selected
    by `--voice-mode`; do not make operators edit a shared gate file to switch.
27. Evidence negatives apply to the required fact, not unrelated caption text.
    Missing hazards or heater indicators do not veto a clear temperature with
    its unit.
28. Agent reasoning always uses the shared Omni service. `--model-mode` selects
    only whether continuous vision uses Cosmos or that same Omni service; it
    never selects a second text model or requires a model-server restart.
29. `--tts-mode` selects the managed Piper or Magpie process and rewrites only
    the temporary TTS preset and endpoint. The launched process and model
    profile must always agree. Keep Magpie at its native speaking rate; waveform
    time stretching makes generated speech sound hollow.
30. A model-generated tool argument validation error gets one immediate retry.
    If it repeats, skip that observation frame and continue; service,
    configuration, and application errors still propagate.
31. A temperature-readout question repeats the fresh visible value without
    verification. A hot-enough question verifies only a fresh numeric reading
    with an explicit Celsius or Fahrenheit unit. Neither path mutates state.
32. Heating detection announces that heating is underway and tells the user to
    wait for the target. It never describes the following step as ready.
33. Foreground tool surfaces separate application modes. Root is composed from
    `RoutedFunction` metadata: vision and RAG are inline, tea launch captures
    foreground, and watcher/transcript/video-log controls remain background.
    Each tea variant exposes tea lifecycle tools plus only its step's quick
    tools.
34. Direct foreground answers are valid and do not change state. Lifecycle
    intent must call its constrained workflow tool; a model-generated invalid
    tool schema receives one immediate retry.
35. Starting is valid only from idle and is idempotent while active. Reset
    returns to root; restart alone clears state and enters the first step.
    Lifecycle mutation tools name the `tea_guide` scope, while timer and
    appliance requests remain step-agent work regardless of lifecycle words.
36. Voice follows foreground-process semantics. A stack selects root or the
    current application before the LLM call. Tea launch pushes tea, reset or
    completion pops it, and internal steps select prebuilt tea variants. The
    inactive agent is never invoked; nested foregrounds return to their caller.
37. Background applications never capture foreground. They receive final raw
    speech transcripts before command gating or ticks only while active;
    synthetic notices bypass both root and transcript input. Start, stop, and
    status are generated NAT functions.
38. Visual change watching keeps only two prior captions by default. The VLM
    receives the per-session start instruction as its focus; one NAT agent sees
    only that instruction plus recent captions, decides importance, and commits
    once. Its labeled data-channel output bypasses TTS. Unavailable frames and
    malformed repeated tool calls defer to a later tick.
39. Transcript recording persists JSON Lines incrementally and does not require
    a wake phrase once explicitly started. At the configured monotonic interval,
    the summary agent receives only unsummarized utterances and emits labeled
    UI text without TTS or root routing.
40. Video logging runs a broad VLM caption on its configured interval. Its delta
    agent receives only the current caption and rolling caption window, commits
    exactly once, and persists both caption and delta. It has no spoken output.
41. The sample-local activity viewer is a read-only process, not an agent
    function. It baselines existing JSON Lines files at startup, tails only new
    complete records, and extracts only configured structured events from the
    current worker log. Its simultaneous transcript, video-log, change-watcher,
    and agent panes do not enter foreground routing, model calls, or TTS.
42. Transcript, video-log, and change-watch applications each own a separate
    per-participant JSON Lines session file and use the shared `jsonl.py`
    lifecycle helpers. The activity viewer treats all three as directory
    sources; only agent diagnostics are decoded from `worker.log`.

## Desktop, foreground, and observation machines

`desktop/runtime.py` owns application scheduling:

```text
foreground stack: root -> app -> nested app
background set:   {change_watch, transcript, video_log, ...}
```

The independent `tea_making_activity_viewer` process tails the change-watch,
transcript, and video-log artifact directories plus the current run's worker log
as declared in `yaml/activity_viewer.json`. It is started before the worker so a
demo recording includes the first structured agent event; existing artifact
files are offsets, not page history. Source formats and event prefix filters are
configuration, and the bounded browser feed never becomes a persistence layer.

An inline function returns to the same foreground. A foreground launch pushes
an application, and exit pops back to the caller. Starting or stopping a
background only changes the set. `desktop/registry.py` reads the stack before
the model call and invokes exactly one registered handler.

`RoutedFunction` is the reusable composition contract. `name` resolves a NAT
function, `route` tells root which requests match, `effect` exposes ownership,
and `return_direct` controls the NAT agent loop. `yaml/applications.yaml` loads
these code-level contracts for this sample; it is not required by the runtime.

`SessionStore` owns the outer deterministic machine:

```text
idle --voice:start--> step --voice:advance--> ... --voice:advance--> idle
```

The active step owns a repeated agentic machine:

```text
due -> trigger -> observation agent -> commit -> evaluate complete -> due
```

When completion already holds, the observation agent still runs and should
commit no updates or message. `SessionStore` records every such call as
`step.commit_noop` and suppresses attempted mutations or notices.

The foreground voice machine is independent of all background monitoring:

```text
root request -> root agent -> {answer | inline tool | foreground launch | background control}
tea request -> current tea-step agent -> {answer | step tool | advance/restart/status/exit}
start tea -> push tea; exit tea -> pop tea
start watcher/transcript/video log -> root remains foreground; background loop starts
```

The generated background start schema accepts one optional bounded
`instruction`. Change watching stores it per participant and uses the configured
default only when it is omitted. Application text output uses the human title
as its data-channel topic, so clients render `[Visual change watcher] ...`
without creating a synthetic voice turn.

Step voice prompts include a compact procedure so references to the current
action can be answered without storing the previous conversation turn.

`runtime/scope.py` carries participant and trace identity into workflow NAT
functions without putting those repeated values in every tool schema. It also
records the lifecycle operation selected during the one locked participant
turn plus small application-private context needed by commit functions. The
root receives only the request and generated route catalog. A tea variant
receives the exact request plus its step's sparse state projection. Change
detection receives only its start instruction and recent captions; transcript
summaries receive only utterances accumulated since the prior summary. The
transcript writer receives final STT output before the wake gate; command
routing still receives only accepted queries.

Video logging receives only its broad current caption and configured rolling
window; neither voice history nor tea state enters that prompt.

## YAML review checklist

For a new state field:

- Use the smallest stable type and description.
- Add `initial` only when an explicit initial value is meaningful.
- Expose it only through steps that read or write it.

For a new step:

- Choose a registered NAT trigger; add a sample-local function only when none
  expresses the external fact.
- Write the VLM question as a short focus guide. Do not prescribe JSON,
  booleans, keys, or another output grammar.
- Add `evidence.pattern` and `evidence.consecutive` when a visual completion
  needs protection from one-frame hallucination or ambiguity.
- Calibrate thresholds to consequence: identification can accept one readable
  OCR caption, while physical milestones such as visible water need repeated
  evidence.
- Keep the step prompt focused on mapping the current observation to state.
- State both the evidence required for a write and when to commit empty updates.
- For derived readiness, require every comparison or supporting fact explicitly;
  repetition, a target value, and a user report are not evidence.
- List only tools the step may call.
- Make completion deterministic through `complete_when`.
- Use `next` only as the destination of explicit voice advancement.
- Render numeric temperatures with `temperature_c` and second counts with
  `duration` in user-facing message templates.
- Add a non-memorization eval case under `eval/cases.yaml`.

Different trigger functions, intervals, prompts, tool sets, projections, and
transition modes do not require a new engine mechanism.

## Prompt and context budgets

Budgets are guardrails for the small local models:

| Input | Budget |
|---|---:|
| Root prompt + generated function catalog + speech rule | 950 characters |
| Tea management + shared voice rules | 550 characters |
| Full per-step foreground prompt | 850 characters |
| Change-watch caption focus | 180 characters |
| Change-watch event policy | 420 characters |
| Transcript summary policy | 240 characters |
| Shared observation prompt | 350 characters |
| Shared voice prompt | 300 characters |
| Generated state contract | 500 characters |
| Step VLM question | 240 characters |
| Step observation policy | 420 characters |
| Step voice policy | 300 characters |

The shared prompts own a generic final-output rule across steps: plain spoken
prose only, without Markdown, lists, formatting marks, code syntax, or internal
names, with shorthand and units spelled out. Do not put domain-specific examples
or repeat this rule in step prompts.

The eval check enforces the budgets. Requests use compact JSON with no
indentation. The root receives only `request`; tool ownership and routing are a
generated catalog, not per-turn context. A tea variant also receives only its
sparse state projection. The voice eval expands each active intent across every
workflow step and root application launch.
An observation or voice agent receives only its step projection. The generated
contract includes only writable fields and their completion values. Do not add
a whole-workflow dump, previous frames, or assistant history.

## Human-test loop

Record these four facts with feedback:

```text
participant_id:
wall-clock window:
spoken request or physical action:
expected versus observed behavior:
```

Then inspect in this order:

1. `desktop.route` and `desktop.foreground.*` for selected ownership.
2. `agent.foreground.request/response` for the selected root or tea agent;
   inspect `agent.foreground.retry` for corrected tool arguments.
3. `desktop.background.*`, `change_watch.*`, `transcript.*`, or `video_log.*`
   for continuous work.
4. `trigger.request/response` for the exact fresh tea caption and latency.
5. `step.evidence` for the deterministic match and consecutive count.
6. `rag.lookup.request/response` when identification needs missing brew facts.
7. `agent.observe.request/response` for the compact context seen by the model.
8. `step.commit`, `step.commit_noop`, or `step.commit_rejected` for the result.
9. `step.ready`, `step.enter`, and `notice.queued` for readiness, explicit
   transition, and user notification.

Prefer the narrowest fix:

- Wrong visual fact: adjust only the focus guide; tune the YAML evidence gate
  when captions are accurate but the acceptance threshold is wrong.
- Missing package fact: inspect the `rag_lookup` query, score, and source before
  changing retrieval text or thresholds.
- Correct fact, wrong state: adjust only the step observation policy.
- Correct identification, delayed readiness: inspect the retrieval passage and
  verify that the first supported field commit also writes `tea_ready` true.
- Wrong tool: reduce or clarify only that step's tool list/description.
- Wrong voice action: tighten only the active foreground prompt or function
  route description and add a state-matrix eval. Do not add another router.
- Wrong watcher output: inspect the stored instruction and caption first, then
  tune only the caption focus or event policy depending on which stage failed.
- Missing transcript summary: inspect its interval deadline, unsummarized
  utterances, summary request, and labeled output before changing the interval.
- Wrong video delta: inspect `video_log.caption` first, then compare the rolling
  history in `video_log.delta`; tune only the corresponding YAML prompt.
- Wrong live answer with a fabricated participant ID: the model-visible vision
  schema has regressed; participant identity must remain invocation-scoped.
- Runtime invariant failure: fix Python and add a unit test.

Never add conversation history to repair a missing state field. Never add a
task-specific Python branch to repair a prompt or YAML policy.

## Event contract

Every event includes `event`. Model and function calls also include `trace_id`,
`participant_id`, and step when applicable. Request events record exact compact
inputs; response events record outputs. `step.evidence` records the plain
caption, match result, consecutive count, and threshold. `step.commit` records
the state delta, revision, completion result, and notification;
`step.commit_noop` records continued observation after completion. Preserve
these fields so test feedback remains searchable across iterations.

Desktop events name `application`, foreground depth, background membership,
and revision. Change-watch events retain the exact caption and importance
summary. `application.text_output` records the label and UI-only message;
watcher output never creates a notice or TTS request. Its JSON Lines artifact
persists the session focus, baseline, every evaluated caption, importance, and
summary. Transcript events log file paths only at lifecycle boundaries and
record input size rather than duplicating user text in the standard log; its
JSON Lines file is the transcript source of truth. `transcription.observed`
proves pre-gate delivery without copying the transcript into the standard log.
Video-log events retain each broad caption and rolling delta; its JSON Lines
artifact is the durable source of truth.

## Reuse candidates

These pieces now have multiple concrete applications but stay sample-local for
this iteration. Move them together in a dedicated library PR after the API has
survived human tests:

- `desktop/types.py`: routed NAT function metadata.
- `desktop/runtime.py`: foreground stack and background set.
- `desktop/registry.py`: deterministic one-foreground dispatch.
- `applications/controls.py` and `applications/background.py`: generated
  background controls and lifecycle fan-out.

Guidance-engine pieces remain sample-only until a second guided workflow uses
them:

- Generic trigger argument resolution and `TriggerRegistry`.
- Participant invocation scope for stateful NAT tools.
- Typed sparse `SessionStore` with YAML write boundaries and completion rules.
- Workflow management NAT functions and compact agent factory.
- Background spoken-notice and labeled UI-text bridges into `VoiceSession`.
- Deterministic temperature normalization and threshold verification.

The retrieval service and `xr_rag` group are already shared library
capabilities. Only the exact-tool alias and tea corpus belong to this sample.

When a second use exists, move the smallest proven contract into `xr-ai-nat`
or `xr-ai-voice`; do not move tea YAML, messages, or state names.

## Non-goals

- Durable sessions across participant disconnects.
- Unbounded conversational memory.
- Arbitrary expressions or code in YAML.
- Silent fallbacks for unavailable model services or invalid configuration.
- Publishing the sample-local desktop API as shared SDK before human validation.
