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

To record a demonstration, say `start recording <task name>` or
`record demo <task name>`, then say `stop recording` when finished. The worker
owns recording state and sends bounded analysis tasks through NAT. Guidance
follows the baseline worker loop: scene changes advance steps, while NAT's
`check_guidance_step_complete` task is only the idle visual fallback.

## NAT Workflow

`yaml/glasses_agent_nat_workflow.yaml` declares:

- `vlm_mcp` as a NAT `mcp_client` function group exposing `ask_image`.
- `video_mcp` as a NAT `mcp_client` function group exposing
  `list_live_participants` and `get_latest_frame` to custom NAT functions.
- `transcript_mcp` as a NAT `mcp_client` function group for internal memory
  persistence and restore; it is not listed in the agent workflow `tool_names`.
- `glasses_agent_tools` as the LLM-facing custom function group exposing
  `describe_current_view`, a composite current-frame VLM tool.
- `glasses_worker_tasks` as an internal custom function group for
  `analyze_recording`, `condense_observations`, and
  `check_guidance_step_complete`.
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
