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

1. STT → text → `NatAgent.infer("describe what you see", pid="alice-xyz", reference_time_us=1777601631000000)`.
   The `reference_time_us` is captured at end-of-speech (voice path) or at the data-message timestamp (data path); without it, every visual frame would be 5-15 seconds out of date because of LLM thinking latency.
2. Worker prepends `[Live participant_id: alice-xyz; user_asked_at_us: 1777601631000000]\nUser: describe what you see` and runs the NAT `tool_calling_agent` workflow.
3. LLM emits a tool call: `get_frame_from_time(participant_id="alice-xyz", second_ago=0, reference_time_us=1777601631000000)` → video-mcp looks up the recorded chunk covering that instant and returns `{"path": "/tmp/xr_video_queries/alice-xyz_ago0_…png", ...}`. The frame matches when the user spoke, not when the tool fires.
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
- **NVENC + NVDEC** for hub video recording and `get_frame_from_time` past-frame decoding.
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
| `get_frame_from_time`          | video-mcp | Frame at `reference_time_us − N s`. Pass `reference_time_us` from the turn's preamble so the frame matches when the user spoke. |
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


## Latency knobs

The default config is tuned for **low-latency, single-user voice**:

| Knob | Where | What it does |
|---|---|---|
| `extra_body.chat_template_kwargs.enable_thinking: false` | `nat_agent_workflow.yaml` (LLM block) | Suppresses Nemotron-3-Nano's `<think>...</think>` reasoning preamble. Each LLM call drops from 8-28 s to 2-5 s on this NVFP4 + eager-mode deployment. Set to `true` if you observe the LLM picking the wrong tool — that is the known failure mode of reasoning-off in this model family. NVIDIA's published recommendation is reasoning OFF for tool calling specifically. |
| `temperature: 0.6, top_p: 0.95` | same | NVIDIA model card's recommended sampling preset for tool-calling. Reasoning tasks use `1.0/1.0`. |
| `enforce_eager: true` | `nemotron3_nano_llm_server.yaml` | Disables CUDA graph capture + FlashInfer FP4 MoE autotune, trading 10-20 % per-token decode for a 5 s startup instead of 3-8 min. Flip to `false` if you want maximum decode throughput and can tolerate the cold-start. |
| `max_iterations: 6` | `nat_agent_workflow.yaml` (workflow block) | Cap on tool-call rounds per turn. Lower it if you want tighter worst-case latency; raise it if you want the LLM to chain more sub-queries. |

### Queue behavior under voice spam

Both the voice and data paths use a **latest-only-replace queue**: at most
one NAT inference runs at a time, with at most one pending behind it. When
a new utterance arrives while the previous one is still processing, it
replaces the pending one (and discards the in-flight result if it lands
after the newer utterance). Logged as `NAT pending REPLACED — older
transcript dropped`. This bounds the queue depth so phantom transcripts
(echo from TTS, background conversation) and rapid user repeats can't pile
up indefinitely.

The currently running LLM call is *not* hard-cancelled — its result is
silently dropped if superseded. So the worst case after a "spam" burst is
one wasted in-flight call, not an unbounded backlog.

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
