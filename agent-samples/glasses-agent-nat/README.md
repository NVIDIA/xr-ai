<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# glasses-agent-nat

NAT-native variant of `agent-samples/glasses-agent`. It keeps the XR process
stack, VAD, STT/TTS, background loop scheduling, memory ownership,
demonstration recording, and guidance lifecycle aligned with the baseline
sample, while moving bounded LLM/tool work into NeMo Agent Toolkit functions.

## Code layout

`worker/` is split by responsibility so each file has a single thing to
read. AI-service HTTP (LLM / STT / TTS) is fully routed through
`agent-sdk/xr-ai-models` per the repo-wide AGENTS.md rule; no
hand-rolled `httpx.post("/v1/...")` calls remain.

| File | What it owns |
|---|---|
| `glasses_agent_nat_worker.py` | Entry point. Probes services, restores memory from transcript-mcp, wires up `GlassesAgent` + `QueryProcessor` + `NatRuntime`, handles shutdown. |
| `agent.py` | Hub IO (audio / data / participant / frame callbacks), per-participant VAD, background VLM observation + memory condenser loops, STT / TTS via `xr_ai_models.make_stt` / `make_tts`. |
| `processors.py` | `QueryProcessor` — entry point for one transcribed utterance. Routes intent (demo? guidance? agentic?) to the right controller. Owns the shared `worker_llm` client used by quick-ack + guidance-question. |
| `demo_phrases.py` | Pure-function phrase matchers (`is_demo_end`, `extract_demo_name`, `match_guidance_request`, etc.). No state. |
| `demo_lifecycle.py` | `DemoController` — start / end / NAT-driven analyze_recording. |
| `guidance.py` | `GuidanceController` — request / advance / finish / monitor / question. Shares `worker_llm` with `QueryProcessor`. |
| `mcp_shims.py` | Thin `call_vlm` / `call_video` / `get_latest_frame_path` wrappers over `NatRuntime.call_tool`. |
| `nat_runtime.py` | `NatRuntime` — one workflow instance + `run_agent` / `call_tool` + shared `normalize_nat_result` for MCP / NAT return shapes. |
| `nat_agent.py` | `NatAgentRunner` — formats one glasses turn into a NAT prompt and invokes the tool-calling workflow. |
| `glasses_nat_register.py` | `@register_function_group` for `glasses_agent_tools` (LLM-facing) and `glasses_worker_tasks` (worker-internal). Builds `agent_llm` / `worker_llm` clients from this sample's `yaml/models.yaml` once at workflow load. |
| `glasses_nat_schemas.py` | Pydantic input / output schemas for NAT functions. |
| `glasses_nat_tasks.py` | Pure task impls (`analyze_recording`, `condense_observations`, `check_guidance_step_complete`, `describe_current_view`). All LLM calls take an injected `LLMService`. |
| `memory.py` | `AgentMemory` + `TranscriptClient` (talks to transcript-mcp via NAT MCP function group, not raw FastMCP). |
| `vad.py` | Per-participant silero-vad detector. |
| `config.py` | `WorkerConfig` loader (resolves `models_yaml` relative to the config dir, same as `nat_workflow_config`). |

The split mirrors the prose in the rest of this README — read the section
about a feature, open the file with the same name.

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
owns recording state and sends bounded analysis tasks through NAT.

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
