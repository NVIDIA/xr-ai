<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# nat-agent

Voice + vision conversational agent built on **NeMo Agent Toolkit (NAT)**'s
`tool_calling_agent`. The LLM is the brain — it decides per turn whether to
fetch a camera frame, ask the vision-language model about it, and how to phrase
that question. Two MCP servers (`vlm-mcp`, `video-mcp`) are exposed verbatim;
the LLM bridges them.

STT (parakeet) and TTS (Piper) run a Pipecat full-duplex pipeline outside the
agentic loop. Echo cancellation is handled at the client (audio input device);
the worker carries no `STTMuteFrame` / playback-tail-wait plumbing.

## Architecture

```
Client (web/iOS/visionOS)
        │ audio + data + camera
        ▼
   xr-media-hub  ───── ZMQ IPC ───────────────────────────┐
   (LiveKit + NVENC                                       │
    video recording)                                      │
        │                                                 │
        │  audio + data                                   │
        ▼                                                 │
   nat-agent worker (Pipecat)                             │
   ┌────────────────────────────────────────────────────┐ │
   │  STT (8103)  →  NAT tool_calling_agent  →  TTS    │ │
   │  (parakeet)     │                          (piper) │ │
   │  (data path also feeds NAT directly)              │ │
   └─────────────────┬───────────────────────────────────┘ │
                     │  infer(user, pid)                   │
                     ▼                                     │
              nemotron3_nano LLM (8107)                    │
              ┌────────────────────────┐                   │
              │ vLLM, qwen3_coder      │                   │
              │ tool-call parser,      │                   │
              │ nano_v3 reasoning      │                   │
              └─────┬──────────────┬───┘                   │
                    │              │                       │
        get_*_frame │              │ ask_image(            │
        get_*_stats │              │   question,           │
        list_*      │              │   image_path)         │
        query_video │              │                       │
                    ▼              ▼                       │
            video-mcp (8210)   vlm-mcp (8220)              │
            ──IPC──────────►   reads PNG from disk         │
                                  │                        │
                                  │ HTTP POST              │
                                  ▼                        │
                          vlm-server (8100)                │
                          Cosmos-Reason1-7B                │
                          (transformers, in-process)       │
                                                           │
                                                           │
                                ◄──────────────────────────┘
                          recordings on /dev/shm
                          (NVENC H.264 chunks)
```

## Agentic flow

A typical visual turn — "describe what you see":

1. STT → text → `NatAgent.infer("describe what you see", pid="alice-xyz")`.
2. Worker prepends `[Live participant_id: alice-xyz]\nUser: describe what you see` and runs the NAT `tool_calling_agent` workflow.
3. LLM emits a tool call: `get_latest_frame(participant_id="alice-xyz")` → returns `{"path": "/tmp/xr_video_queries/alice-xyz_latest_…png", ...}`.
4. LLM emits a second tool call: `ask_image(question="Describe the scene in detail.", image_path="/tmp/...")` — note the LLM is free to rephrase, expand, or invent its own follow-up question.
5. vlm-mcp reads the PNG, POSTs to vlm-server, returns the answer.
6. LLM composes the final reply.
7. Reply goes out as `agent.response` data + sentence-batched Piper TTS.

For non-visual turns ("what's 2+2"), the LLM answers directly without any tool call.

## Quickstart

```bash
cd agent-samples/nat-agent

# Install orchestrator + worker
uv sync
(cd worker && uv sync)

# Boot everything
uv run nat_agent
```

First run downloads ~35 GB of model weights (Cosmos-Reason1-7B + Nemotron-3-Nano NVFP4 + Parakeet + Piper voice) into the repo-root `models/` cache. Subsequent runs start offline.

Connect a browser at `http://localhost:8080` (or `https://localhost:8443` if `web_server_tls: true`).

## Requirements

- **Blackwell GPU** for the NVFP4 LLM (B200, RTX PRO 6000, Jetson Thor, DGX Spark). FlashInfer FP4 MoE kernels target Blackwell. Swap to `nvidia/Nemotron-Nano-3-30B-A3B` (BF16, ~60 GB VRAM) on Hopper/Ampere.
- **NVENC + NVDEC** for hub video recording and `get_frame_at_time` decoding.
- **VRAM budget**: vLLM (~57 GB at default `gpu_memory_utilization: 0.6` on a 96 GB card), vlm-server (~14 GB), stt-server (~1.5 GB).

## Ports

| Service     | Port | Purpose                                                       |
|-------------|------|---------------------------------------------------------------|
| LiveKit WS  | 7880 | WebSocket signaling                                           |
| LiveKit TCP | 7881 | TURN/ICE over TCP                                             |
| LiveKit UDP | 7882 | TURN/ICE over UDP                                             |
| Web client  | 8080 | HTTP (or 8443 with TLS)                                       |
| vlm-server  | 8100 | Cosmos-Reason1-7B (called only by vlm-mcp)                    |
| stt-server  | 8103 | Parakeet-TDT-0.6B                                             |
| tts-server  | 8105 | Piper (en_US-lessac-medium)                                   |
| llm-server  | 8107 | Nemotron-3-Nano-30B-A3B-NVFP4 via vLLM                        |
| video-mcp   | 8210 | FastMCP at /mcp; live + recorded frame access                 |
| vlm-mcp     | 8220 | FastMCP at /mcp; single tool `ask_image(question, image_path)` |

## Tool surface visible to the LLM

Every MCP tool from both servers is exposed verbatim in `nat_agent_workflow.yaml`:

| Tool                           | From      | Purpose                                            |
|--------------------------------|-----------|----------------------------------------------------|
| `ask_image`                    | vlm-mcp   | Question + PNG path → answer text                  |
| `get_latest_frame`             | video-mcp | Most recent live frame for a participant           |
| `get_frame_at_time`            | video-mcp | Past frame near a given Unix-µs timestamp          |
| `get_video_stats`              | video-mcp | Chunk count/byte total/earliest/latest µs          |
| `query_video`                  | video-mcp | H.264 video clip (LLM rarely uses; VLM can't read) |
| `list_live_participants`       | video-mcp | Hub IPC roster                                     |
| `list_recorded_participants`   | video-mcp | Disk roster (recorded chunks)                      |

The LLM picks. There are no NAT-side wrappers hiding tool arguments — the
`participant_id` that video-mcp tools require is supplied via a per-turn
preamble (`[Live participant_id: <pid>]`) that the worker prepends to the
user message; the LLM reads it and passes it through.

## Configuration

| File                              | Purpose                                          |
|-----------------------------------|--------------------------------------------------|
| `xr_media_hub.yaml`               | Hub + LiveKit + NVENC recording                  |
| `stt_server.yaml`                 | Parakeet model + device                          |
| `piper_tts_server.yaml`           | Piper voice                                      |
| `vlm_server.yaml`                 | VLM model (called via vlm-mcp)                   |
| `nemotron3_nano_llm_server.yaml`  | LLM model + vLLM knobs                           |
| `vlm_mcp_server.yaml`             | vlm-mcp host/port + vlm-server URL               |
| `video_mcp_server.yaml`           | video-mcp host/port + recordings_dir             |
| `nat_agent_worker.yaml`           | Worker URLs, prompts, VAD knobs                  |
| `nat_agent_workflow.yaml`         | NAT workflow (LLM + tools + agent config)        |

`nat_agent_worker.yaml` ↔ `nat_agent_workflow.yaml` placeholders: the worker
loads the workflow YAML at startup and substitutes `${...}` placeholders
(LLM URL, MCP URLs, prompt, sampling) from the worker config. Edit the
worker YAML to change ports/timeouts/prompts; edit the workflow YAML to
restructure the agent itself.

## How this differs from `pipecat-nat-nemotron3nano`

The reference sample at `Desktop/xr-ai/agent-samples/pipecat-nat-nemotron3nano/`
ships the same LLM but a fundamentally different agentic design:

| | `pipecat-nat-nemotron3nano` | `nat-agent` (this sample) |
|---|---|---|
| MCP tool surface | One tool: `ask_camera(question)` | All seven tools verbatim |
| Frame source | vlm-mcp owns hub IPC, pulls frames itself | video-mcp owns hub IPC; LLM bridges via PNG path |
| `participant_id` | Hidden via NAT-side `_PidProvider` wrapper | Exposed in user-message preamble; LLM passes it through |
| LLM agency | Routes to vlm or doesn't | Picks the frame source, crafts its own VLM question |
| Custom NAT functions | 1 (`mcp_ask_camera`) | 0 |
| vlm-mcp deps | `xr-ai-agent`, numpy, FastAPI | None of those — pure FastMCP + httpx + Pillow |
| Echo handling | `STTMuteFrame` + playback-tail wait in worker | Dropped; client AEC handles it |
| Past-frame access | Not available | `get_frame_at_time` + recorded chunks |

## Swap models / TTS engine

| Layer | File                              | Key                          |
|-------|-----------------------------------|------------------------------|
| LLM   | `nemotron3_nano_llm_server.yaml`  | `model`                      |
| VLM   | `vlm_server.yaml`                 | `model`                      |
| STT   | `stt_server.yaml`                 | `model`                      |
| TTS   | `piper_tts_server.yaml`           | `voice`                      |

To switch to Magpie TTS (multilingual, GPU, ~2 s/sentence): change the
`Process("tts", ...)` row in `nat_agent.py` to point at
`../../ai-services/tts/magpie` and copy `magpie_tts_server.yaml`. Update
`nat_agent_worker.yaml`'s `tts_server` URL to `http://localhost:8104`.
