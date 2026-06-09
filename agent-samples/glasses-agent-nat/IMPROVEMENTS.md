<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# glasses-agent-nat — teacher/student workflow review & improvements

A focused review of the teacher→student demonstration workflow against four
axes plus the depth of NeMo Agent Toolkit (NAT) usage. Each section states
what the code does today, where it can improve, and which items are
**implemented on this branch** vs. **recommended** (with concrete designs).

> Constraint: these changes were authored without running the full
> GPU/model/NAT stack. Implemented items are conservative and
> correct-by-construction; larger refactors are written up as ready-to-apply
> designs rather than blind edits.

## Update — structured key-info monitoring (implemented)

The biggest source of over-strict monitoring was the agent checking the
student against the teacher's **long free-text caption + raw frame**, which
made it sensitive to irrelevant differences (background, lighting, camera
angle). This is now fixed end-to-end:

- **New `StepKeyInfo` structure** (`memory.py`): each step distills its long
  description into `objects` / `action` / `position` / `target_state` plus an
  `ignore` list of details that must not affect the verdict. Stored on
  `DemoStep.key_info`, populated at finalize time by a new NAT worker task
  **`derive_step_key_info`** (`glasses_nat_tasks.py` + schema + register +
  workflow YAML include).
- **Key-info-guided, two-tier check** (`check_guidance_step_complete_impl`):
  1. **image → image** — compare the student frame to the teacher frame, but
     judge ONLY the key objects/action/position; differences in the `ignore`
     list (background, lighting, camera angle, clothing, hand pose,
     color-name) are explicitly told not to change the verdict. Pass ⇒
     advance.
  2. **image → text** — if the image-to-image check fails, check whether the
     student frame satisfies the key info as text (objects in target
     state/placement, action). Pass ⇒ advance.
  3. **correction** — if both fail, tell the student what is wrong, grounded
     in key info (`_key_info_correction`: e.g. "headset should be on your
     head") rather than a strict pixel mismatch or a parser reason.

This directly implements the requested flow and stops the agent from failing
a correct step over an irrelevant background change.

---

## 1. Capturing key frames

**Today.** During recording, `_background_vlm_loop` densely captures every
new live frame at `vlm_interval_s` cadence — each frame gets a full VLM
`ask_image` description, is copied to a stable per-demo path, and is appended
to a JSONL log (no dedup; analysis happens after stop). Key frames are then
*selected* per step after recording by `_select_reference_frames_for_step`,
which scores candidate frames by token overlap between the step instruction
and the frame's VLM caption, anchored to the teacher's **voice-note
timestamps** (step boundaries), with a VLM tie-break for close scores.

**Gaps.**
- Key-frame selection is anchored to *when the teacher narrated*, not to
  *what visually changed*. A step the teacher narrates poorly gets a weak
  reference frame. There is no motion / scene-change signal to nominate
  "this is the moment the state changed."
- Dense per-frame VLM captioning is expensive and largely redundant when the
  scene is static (teacher pausing, repositioning).

**Implemented on this branch.**
- `recording_warmup_s` is now configurable (was hard-coded 2 s) so the
  positioning skip can be tuned per device/task.

**Recommended (designs).**
- **Motion-gated capture.** Before the per-frame VLM call, compute a cheap
  perceptual delta (downscaled gray mean-abs-diff or dHash) against the last
  captured frame; only run the VLM caption when the delta exceeds a
  threshold. Tag each `RecordedFrame` with a `change_score`. This both cuts
  VLM load and yields a natural "key moment" signal.
- **Change-anchored key frames.** In `_select_reference_frames_for_step`, add
  the post-change settle frame (local maximum of `change_score` within the
  step window) as a candidate alongside the voice-note-anchored frames, so
  the end-state frame is preferred even when narration timing is off.

---

## 2. Dictating instructions

**Today.** Steps come from the teacher's `VoiceNote`s; `analyze_recording`
turns voice (primary) + frame captions (secondary) into ordered step text.
During guidance, `_speak_current_guidance_step` speaks
`"Step N of M: <instruction>"` verbatim.

**Gaps.**
- The spoken instruction is the raw analyzed text. The per-step
  `teacher_caption` (a VLM description of the teacher's end-state frame) and
  `expected_requirements` are already stored but are **not** woven into what
  the student hears, so dictation can be vaguer than the captured demo
  supports ("hold the controller" vs. "hold the blue controller in your right
  hand, grip facing down").

**Recommended (designs).**
- **Grounded dictation.** When speaking a step that has a reliable
  `teacher_caption`, generate the spoken line from `instruction +
  teacher_caption` via the existing `worker_llm` (one bounded NAT task,
  e.g. `phrase_step_instruction`) constrained to ≤ 1 sentence, second person,
  no aesthetics — reusing the style contract already in the workflow prompt.
  Cache the phrasing on the `DemoStep` at finalize time so guidance stays
  latency-free and deterministic.
- **Progressive detail.** Speak the concise instruction first; if a
  correction fires for the same requirement twice, escalate to the
  caption-grounded detail ("the lid should be fully closed and flush").

---

## 3. Data structure for teacher-demo information

**Today.** `Demonstration { recorded_frames[], voice_notes[], steps[],
summary, instructions[] }` with a rich `DemoStep { description, image_path,
teacher_caption, expected_requirements[], reference_image_paths[],
reference_reliable, text_video_mismatch }`. This is already well-factored.

**Gaps.**
- Demos do **not** persist across sessions — and this is *intentional*
  because reference `image_path`s point at run-scoped `XR_RUN_DIR` files that
  die on restart (see `memory.py`). So "just persist demos" is a known trap.
- No explicit per-step timing/object metadata beyond free text; the
  candidate-selection scores are traced but not retained for later
  re-evaluation.

**Recommended (designs).**
- **Durable demos (done right).** Add `save_demonstration(demo)` /
  `load_demonstrations()` that copy each step's reference frame into a
  persistent `demos/<name>/` directory (outside `XR_RUN_DIR`), serialize the
  `Demonstration` dataclass to JSON there, and rewrite `image_path`s to the
  durable copies on load. This makes the existing structure session-stable
  without the dead-path hazard, and can ride the existing `transcript_mcp`
  group for the JSON index.
- **Retain selection provenance.** Persist the per-step candidate scores +
  chosen `frame_idx` on `DemoStep` (additive field) so reference quality can
  be audited/replayed offline against the eval harness.

---

## 4. Monitoring student behavior

**Today.** `_guidance_monitor_loop` runs every `guidance_check_interval_s`:
it calls the NAT `check_guidance_step_complete` task (VLM compares the live
student frame to the teacher reference frame), advances after **2**
consecutive grounded "complete" verdicts, and emits rate-limited spoken
corrections on grounded mismatches. Solid grounding discipline (evidence must
come from the student frame, not the teacher reference).

**Gaps.**
- The loop logged an observation `delta` but **never used it** to gate the
  expensive check — and during guidance the background observation loop is
  paused, so that delta is always 0. The grounded VLM compare therefore ran
  every cycle even when the camera frame was unchanged, wasting VLM bandwidth
  and exposing the advance decision to VLM nondeterminism on identical
  frames.
- The "2 consecutive" threshold was hard-coded.
- Monitoring is per-step only: there is no detection of the student doing the
  *wrong* / a *later* step, or skipping ahead.

**Implemented on this branch.**
- **Static-frame skip** (`guidance_skip_static_frames`, default on): the
  monitor does a cheap metadata-only `get_latest_frame` and skips the grounded
  VLM check when the live frame timestamp hasn't advanced since the last
  check — identical frame ⇒ identical verdict. Leaves the yes-streak and
  correction state untouched, so it is strictly behavior-preserving while
  cutting redundant VLM calls (new `_latest_live_frame_ts`, tracked via
  `_guidance_last_checked_live_ts`, reset per step).
- **Configurable confirmations** (`guidance_advance_confirmations`,
  default 2): the consecutive-grounded-YES count required to auto-advance is
  now tunable.

**Recommended (designs).**
- **Wrong-step / look-ahead detection.** On a grounded mismatch, run a cheap
  check of the *next* step's requirements; if those are satisfied, the
  student skipped ahead — advance and acknowledge rather than correct. (The
  reference frames for all steps already exist.)
- **Idle nudge.** Track time since the step was spoken with no progress and,
  after a configurable interval, offer the grounded detail / a "need a hand?"
  prompt instead of silently re-checking.

---

## 5. Is it NAT-native? Making more use of NAT

**Today — already genuinely NAT-native.** The sample registers two custom
function groups (`glasses_agent_tools`, `glasses_worker_tasks`) via
`register_function_group`, wires `vlm_mcp` / `video_mcp` / `transcript_mcp` as
NAT `mcp_client` groups, runs request-time work through a NAT
`tool_calling_agent` workflow, and drives everything through
`WorkflowBuilder.from_config` + `SessionManager`. Worker-internal bounded
tasks (`analyze_recording`, `derive_step_requirements`,
`condense_observations`, `check_guidance_step_complete`) are real NAT
functions with typed pydantic I/O.

**Implemented on this branch.**
- **Worker-task LLM calls now go through NAT's LLM layer.** The four bounded
  tasks (`analyze_recording`, `condense_observations`,
  `derive_step_requirements`, `derive_step_key_info`) no longer hand-roll
  `httpx` POSTs to `/v1/chat/completions`. `glasses_worker_tasks` resolves the
  models declared in the workflow `llms:` block via
  `builder.get_llm(name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)` and the
  tasks invoke through a single `_chat(llm, messages, …)` helper
  (`.bind(...)` for per-call `max_tokens`/`temperature`/`extra_body`). Endpoint,
  retry, and timeout config now live once in the YAML `llms:` block instead of
  the function-group's `agent_llm_server`/`llm_server` URLs (replaced by
  `agent_llm_name`/`worker_llm_name`). The carefully tuned prompt dicts are
  unchanged — LangChain normalizes the OpenAI-style message dicts — so this is
  a transport swap, not a prompt change. This removes the only hand-rolled HTTP
  in the NAT task layer. (The worker's own latency-critical quick-ack/intent
  path stays direct by design; it is not a NAT function.)

**Where NAT is still under-used.**
- **No NAT observability.** NAT ships tracing/telemetry and an `eval`
  harness; neither is configured. Enabling `general.telemetry` tracing would
  surface every function/LLM call (latency, tokens, errors) — exactly the
  kind of multi-tool agentic pipeline NAT's observability is meant to show
  off.
- **Hand-rolled orchestration.** The guidance monitor and recording analysis
  are bespoke Python loops. A NAT-native showcase could express the
  per-step "decide: advance / correct / wait" decision as a small NAT
  reasoning function, and run the recording-analysis as a NAT pipeline.

**Recommended (designs, in priority order).**
1. ~~Route `glasses_worker_tasks` LLM calls through `builder.get_llm(...)`
   instead of `_post_chat`.~~ **Done on this branch** (see "Implemented" above).
2. Add a `general.telemetry` tracing block to
   `glasses_agent_nat_workflow.yaml` and document `nat eval` for the guidance
   completion check, so the sample demonstrates NAT observability + eval, not
   just function registration.
3. (Larger) Express the guidance step decision as a NAT function/agent so the
   advance/correct policy is declarative and reusable.

---

## Summary of changes on this branch (`wenxind/glasses-improve`)

| File | Change |
|---|---|
| `worker/memory.py` | New `StepKeyInfo` (objects/action/position/target_state/ignore); `DemoStep.key_info`. |
| `worker/glasses_nat_schemas.py` | `DeriveStepKeyInfo{Input,Output}`; key-info fields on `GuidanceStepInput`. |
| `worker/glasses_nat_tasks.py` | New `derive_step_key_info_impl`; key-info-guided + background-tolerant prompts in the two-tier completion check; `_key_info_correction`. **NAT-native LLM:** replaced hand-rolled `_post_chat` httpx with a `_chat(llm, …)` helper over NAT's LangChain LLM. |
| `worker/glasses_nat_register.py` | Register `derive_step_key_info`; thread key-info into the completion check. **NAT-native LLM:** resolve `agent_llm`/`worker_llm` via `builder.get_llm(..., LLMFrameworkEnum.LANGCHAIN)`; config now takes `agent_llm_name`/`worker_llm_name` instead of base URLs. |
| `worker/processors.py` | Extract+store key info at finalize; pass it into the guidance check. Auto-advance monitor: static-frame skip (cheap `_latest_live_frame_ts`, behavior-preserving) + configurable confirmation count. |
| `worker/config.py` | New tunables: `recording_warmup_s`, `guidance_advance_confirmations`, `guidance_skip_static_frames`. |
| `worker/agent.py` | Recording warmup window now uses `recording_warmup_s`. |
| `yaml/glasses_agent_nat_workflow.yaml` | `derive_step_key_info` added to the worker-task group. **NAT-native LLM:** `glasses_worker_tasks` now takes `agent_llm_name`/`worker_llm_name` (resolved from the `llms:` block) instead of `agent_llm_server`/`llm_server` URLs. |
| `yaml/glasses_agent_nat_worker.yaml` | Documents the three new knobs. |
| `README.md` | Notes the worker tasks run through NAT's LLM layer. |
| `IMPROVEMENTS.md` | This review. |

All other items above are left as concrete, prioritized recommendations for
follow-up (they need the live stack to validate). No PR — pushed to the branch
for iteration.
