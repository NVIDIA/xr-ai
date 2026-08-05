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
5. Observation agents can write only the active step's `writes`. Voice agents
   are read-only. The top router alone chooses management versus delegation.
6. No conversation transcript is included. Add memory only for a named case
   that cannot be solved with the current question and step projection.
7. Recover only errors that a later frame or corrected model call can repair.
   Let configuration, programming, and service failures retain their traceback.
8. Package brewing values outrank RAG. Retrieval is step-scoped through the
   exact `rag_lookup` NAT function, and its source is committed with the values.
   Never retrieve without a specific visible tea name; return at most two
   passages to the small model. RAG supplies brewing values, never tea identity.
9. Step completion never transitions. Observation continues homogeneously;
   only the top router may advance after an explicit user voice request. Next,
   continue, advance, and skip always select the advance function, never the
   step question function.
10. VLM triggers return plain-text captions from focus-only questions. A
    YAML evidence gate, not the observation agent's interpretation alone,
    authorizes visual completion after repeated matching frames.
11. Sessions are ephemeral: first join and app-start roster replay reset to
    idle, participant leave discards state, and duplicate roster events are
    idempotent.
12. Router management is literal. Reports of physical actions and all questions
    delegate to the step agent; only explicit management language selects a
    management function.
13. Model-visible vision requests contain only a question. `current_view`
    injects participant identity from invocation scope.
14. User-facing values are natural speech. Keep state numeric; use the shared
    voice rule and message render filters instead of step-specific display
    fields or preformatted state.
15. Treat prompt failures as prompt-tuning work first. Confirm before changing
    evidence thresholds, state semantics, transitions, or other runtime policy.
16. When observations and state can use different units, delegate normalization
    and comparison to a typed NAT tool. The heat agent transcribes the fresh
    number and unit, passes the state target to `temperature__verify`, and
    follows its boolean result; it never performs arithmetic or substitutes the
    target for the reading.
17. Observation agents receive `already_complete` as prior status plus a
    generated contract containing writable field meanings and the YAML
    completion condition. Prior status is never copied into writable state.
18. Every observation turn calls `workflow__commit` exactly once. Completed or
    unsupported observations commit empty updates and message; they never emit
    free-form state or force a readiness field.
19. Prompts state both positive evidence for each readiness write and its
    negative stop condition. Identification commits all resolved fields and
    `tea_ready` atomically; unrelated retrieval text never supplies a value.
20. Observation `message` briefly announces a real, non-completing state change
    or immediate danger. No-change, repeated, and completion commits leave it
    empty; this rule stays in the shared agent prompt, not step YAML.
21. Tea identification is literal OCR, not visual classification. The vision
    focus excludes artwork and supplied words; RAG may provide brewing values
    only when its passage contains the variety already visible in the caption.
22. Launching requires explicit model, voice, and TTS modes. The orchestrator
    writes temporary model, worker, and RAG configs from those selections so
    their profiles cannot diverge; never restore an implicit launch default.
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
    profile must always agree.
30. A model-generated tool argument validation error gets one immediate retry.
    If it repeats, skip that observation frame and continue; service,
    configuration, and application errors still propagate.

## Nested machines

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

The voice machine is independent of monitoring:

```text
request -> router -> {start | advance | reset | status | ask_step}
ask_step -> current read-only voice agent -> optional read-only tools -> answer
```

Step voice prompts include a compact procedure so references to the current
action can be answered without storing the previous conversation turn.

`runtime/scope.py` carries participant and trace identity into workflow NAT
functions without putting those repeated values in every tool schema. The
scope exists only during one locked participant turn or observation.

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
| Shared router prompt | 300 characters |
| Shared observation prompt | 350 characters |
| Shared voice prompt | 300 characters |
| Generated state contract | 500 characters |
| Step VLM question | 240 characters |
| Step observation policy | 420 characters |
| Step voice policy | 300 characters |

The shared prompts own a generic final-output rule across steps: rewrite
abbreviations, symbols, unit notation, machine formats, and compact tool or
state text as complete spoken words and familiar quantities while preserving
meaning. Do not put domain-specific examples or repeat this rule in step
prompts.

The eval check enforces the budgets. Requests use compact JSON with no
indentation. Router context is only `request`, `active`, and current step title.
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

1. `agent.router.request/response` for the selected outer function.
2. `trigger.request/response` for the exact fresh caption and latency.
3. `step.evidence` for the deterministic match and consecutive count.
4. `rag.lookup.request/response` when identification needs missing brew facts.
5. `agent.observe.request/response` for the compact context seen by the model.
6. `step.commit`, `step.commit_noop`, or `step.commit_rejected` for the result.
7. `step.ready`, `step.enter`, and `notice.queued` for readiness, explicit
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
- Wrong route: tighten the shared router or management function description and
  add a routing eval; keep the task router prompt limited to domain identity.
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

## Reuse candidates

These pieces stay sample-local until a second concrete guided application uses
them:

- Generic trigger argument resolution and `TriggerRegistry`.
- Participant invocation scope for stateful NAT tools.
- Typed sparse `SessionStore` with YAML write boundaries and completion rules.
- Workflow management NAT functions and compact agent factory.
- Background notice bridge into `VoiceSession`.
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
- A generic library abstraction before another sample validates it.
