<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# glasses-agent-nat

NAT-native variant of `agent-samples/glasses-agent`. It keeps the XR process
stack, VAD, STT/TTS, background loop scheduling, memory ownership,
demonstration recording, and guidance lifecycle aligned with the baseline
sample, while moving bounded LLM/tool work into NeMo Agent Toolkit functions.

## Run

```bash
cd agent-samples/glasses-agent-nat
uv sync
uv run glasses_agent_nat
```

The orchestrator starts hub, STT, TTS, VLM, two LLM servers, VLM/video/transcript
MCP servers, and the worker. The worker loads
`yaml/glasses_agent_nat_workflow.yaml` and invokes NAT in-process.

Voice interruption follows the same timing as `simple-vlm-example`: raw VAD
speech-start does not flush spoken responses because it can fire on background
noise or speaker echo. Once an utterance is finalized, STT and the NAT noise
gates decide whether it is a real request; accepted transcripts interrupt the
current response and flush queued TTS. Saying `stop`, `cancel`, or `be quiet`
still stops speech after the transcript is accepted.

To record a demonstration, say `start recording <task name>` or
`record demo <task name>`, then say `stop recording` when finished. The worker
owns recording state and sends bounded analysis tasks through NAT. During
guidance, the worker checks the student's current frame on a fixed cadence,
guided by each step's structured **key info** (objects/action/position/
target_state). The check is two-tier: it first compares the live frame to the
teacher reference frame via `vlm_mcp.ask_frames` (judging only the key info,
ignoring background/lighting/camera angle); if that fails it checks whether the
live frame satisfies the key info as text via `vlm_mcp.ask_image`; if both
fail it speaks a key-info-grounded correction (e.g. "headset should be on your
head"). Live student frames must be newer than the current spoken step before
they can advance guidance, and the monitor skips redundant checks when the
frame has not changed (`guidance_skip_static_frames`). A configurable number of
grounded correct checks (`guidance_advance_confirmations`, default 2) advances
to the next step.

## NAT Workflow

`yaml/glasses_agent_nat_workflow.yaml` declares:

- `vlm_mcp` as a NAT `mcp_client` function group exposing `ask_image` and
  `ask_frames`.
- `video_mcp` as a NAT `mcp_client` function group exposing
  `list_live_participants` and `get_latest_frame` to custom NAT functions.
- `transcript_mcp` as a NAT `mcp_client` function group for internal memory
  persistence and restore; it is not listed in the agent workflow `tool_names`.
- `glasses_agent_tools` as the LLM-facing custom function group exposing
  `describe_current_view`, a composite current-frame VLM tool.
- `glasses_worker_tasks` as an internal custom function group for
  `analyze_recording`, `derive_step_requirements`, `derive_step_key_info`,
  `condense_observations`, and `check_guidance_step_complete`. At finalize
  time each step is distilled into structured **key info**
  (`objects`/`action`/`position`/`target_state` + an `ignore` list) stored on
  `DemoStep.key_info`. The guidance task receives the current
  `DemoStep.image_path` (Image 1) and the step's key info, and judges ONLY the
  key objects/action/placement — differences in the `ignore` list
  (background, lighting, camera angle, clothing) must not change the verdict.
- `workflow` as `tool_calling_agent` with `handle_tool_errors: true` and
  `max_iterations: 8`.

The configured `include` lists are the LLM-facing tool contract. If you enable
recorded video tools in `video-mcp`, update the NAT workflow YAML and custom
function groups in the same change so the exposed tools stay explicit.

## Inspect

From `agent-samples/glasses-agent-nat/worker`:

```bash
uv run nat validate --config_file ../yaml/glasses_agent_nat_workflow.yaml
uv run nat info components -t function_group
uv run nat mcp client tool list --url http://localhost:8240/mcp
uv run nat mcp client tool list --url http://localhost:8210/mcp
```

The same workflow can be served through NAT for standalone inspection:

```bash
uv run nat serve --config_file ../yaml/glasses_agent_nat_workflow.yaml
uv run nat mcp serve --config_file ../yaml/glasses_agent_nat_workflow.yaml
```
