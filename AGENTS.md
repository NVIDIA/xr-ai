<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai — Working Conventions

The contract every change must satisfy. Topic deep-dives live in `docs/`;
historical decisions in `docs/changelog.md`.

## Architecture (sketch)

```
client-samples/     # Platform clients (Android, iOS/visionOS, Web)
agent-sdk/          # Five packages:
                    #   xr-ai-hub-client   — IPC client library (pyzmq + msgpack only)
                    #   xr-ai-models       — LLM/VLM/STT/TTS service protocols + OpenAI-compat clients
                    #   xr-ai-pipecat      — optional Pipecat transport bridge (heavier deps)
                    #   xr-ai-voice        — voice runtime (VoiceSession); introduced alongside xr-ai-pipecat
                    #   xr-ai-nat          — native Relay-managed tools; legacy NAT compatibility during migration
utils/              # Shared infra: launcher, logging, vad, vllm, voicegate
services/           # XR hub, CloudXR, model-serving, and typed capability services
agent-mcp-servers/  # MCP adapters: oxr, render, transcript, vec, video, vlm
agent-samples/      # End-to-end agent demos
tests/              # Multi-client / multi-agent integration tests
docs/               # Topic deep-dives + changelog
models/             # Gitignored model-weight cache (per-YAML model_cache target)
deps/               # Gitignored downloaded binaries (e.g. LOVR AppImage)
```

## Hard rules

- **One hub, many clients, many agents.** Hub fans inbound to every
  `ProcessorEndpoint`; return traffic goes only to the originating client.
- **Agents talk to the hub via IPC only.** LiveKit is an internal transport
  detail — never surface it to agents.
- **Client readiness is owned by the hub, not by any one agent.** An agent
  reports only its own `_agent.status`; the hub aggregates across the agents
  responsible for that participant and gates the result on a confirmed
  subscription. Readiness participation is opt-in
  (`announces_readiness=True`) and an endpoint answers only for participants
  it subscribes to — a passive `ProcessorEndpoint` must not gate any client.
  Don't publish availability straight to clients from a worker.
- **`agent-sdk/xr-ai-hub-client` depends only on `pyzmq` + `msgpack`.** No
  LiveKit, FastAPI, or uvicorn. `agent-sdk/xr-ai-pipecat` is a separate
  optional package with heavier deps (pipecat-ai, scipy, numpy, httpx,
  fastmcp); it bridges `ProcessorEndpoint` to Pipecat pipelines.
- **All HTTP calls to AI services go through `agent-sdk/xr-ai-models`.**
  Workers and services depend on its typed protocols
  (`LLMService`, `VLMService`, `STTService`, `TTSService`, `EmbeddingService`)
  and construct clients from a per-sample model profile via `make_llm` /
  `make_vlm` / `make_stt` / `make_tts` / `make_embedding`. Profiles separate
  adapter behavior, endpoint
  connectivity/readiness, and deployment ownership. Hand-rolled `httpx` clients
  against `/v1/chat/completions`, `/v1/audio/transcriptions`, or
  `/v1/audio/speech`, or `/v1/embeddings` are forbidden — model quirks belong in this one
  package's presets, not in callers. No vendor SDKs (no `openai`, no
  `anthropic`, no `litellm`); all in-tree backends speak
  OpenAI-compatible HTTP.
- **Workers never import from `xr_media_hub` or `xr_ai_launcher`.** Use the
  public `xr_ai_hub`, `xr_ai_models`, `xr_ai_nat`, and `xr_ai_voice` SDK
  surfaces plus task-specific libraries (numpy, torch, …).
- **Agentic functions are native and in-process.** New and migrated tools live
  in `xr-ai-nat`; every tool and tool-driven agent lifecycle passes through
  NeMo Relay, and all model I/O remains in `xr-ai-models`. Its legacy extras
  retain NeMo Agent Toolkit compatibility only while existing function groups
  migrate. Existing MCP servers remain compatibility surfaces while their
  capabilities migrate.
- **RAG is a native typed capability.** `rag-service` owns document chunking,
  embedding caches, and dense retrieval behind private msgpack/ZMQ;
  `RAGFunctionsConfig` exposes it as the `xr_rag` NAT function group.
- **A process boundary does not imply MCP.** `xr-ai-nat[mcp]` may expose an
  application's explicit native-function list to MCP-only agents, but native
  applications invoke the functions directly. XR tracking calls the typed
  OpenXR service; OXR MCP only republishes the tracking and spatial functions.
  Video memory calls its typed service for recorded history; live frames stay
  with the caller's hub client and Video MCP keeps its temporary live adapter.
  Text-memory owns transcript
  JSONL storage; transcript MCP only republishes that capability.
- **Application-specific capabilities stay with their application.** XR render
  scene state, native scene functions, and the LOVR app live together under
  `agent-samples/xr-render-demo/scene`; they are not exported from `xr-ai-nat`.
  Render MCP only republishes that sample-local typed capability.
- **Native vision tools own frame acquisition; the file-path adapter stays
  separate.** The native `xr_vision_tools` group (`look_at_current_frame` /
  `look_at_past_frame`) acquires the participant's live or recorded frame itself
  and calls a `VLMService` from `xr-ai-models`; recorded lookups resolve through
  the `xr_video_memory` group. The legacy file-path `ask_image` tool is not part
  of the native surface — it lives self-contained in VLM MCP for MCP-only clients
  that already hold an image path.
- **No API keys or tokens in source files** — use env vars or
  `xr_media_hub.yaml`. See `docs/credentials.md`.

## Process model essentials

Each sample has an orchestrator and worker; application-specific capability
processes may be nested beside them when they are not reusable SDK services:

| Sub-project | Role | Dependencies |
|---|---|---|
| `<sample>/` | Orchestrator — declares `PROCESSES`, calls `run_stack` | `xr-ai-launcher` only |
| `<sample>/worker/` | Agent worker — connects to hub via IPC | `xr-ai-hub-client` + task libs |
| `<sample>/<capability>/` | Optional application-specific native function/service slice | Narrow capability deps |

- Processes start serially in declaration order; each must `Path(--ready-file).touch()`
  when ready.
- `xr_media_hub` always runs as its own process — never embedded.
- `run_stack` is fail-fast: any process exit terminates the stack.
- Process management lives in `utils/xr-ai-launcher/`, not inside any process it manages.

Full mechanics and the `Process(...)` declaration form: `docs/process-model.md`.

## Adding a sample

Pick a kebab-case name (e.g. `simple-vlm-example`); derive everything else
mechanically:

| Thing | Convention | Example |
|---|---|---|
| Sample directory | `agent-samples/<kebab>/` | `simple-vlm-example/` |
| Orchestrator entry | `<snake_name>` | `simple_vlm_example` |
| Worker entry | `<snake_name>_worker` | `simple_vlm_example_worker` |
| Agent class | `<CamelName>Agent` | `SimpleVlmAgent` |

**Worker code rules** (apply to every sample worker):

- Import IPC types from `xr_ai_hub`; native agent functions come from
  `xr_ai_nat`, model clients from `xr_ai_models`, and the native voice runtime
  from `xr_ai_voice`.
- Raw IPC workers keep `_HUB_PUB` / `_HUB_PUSH` as module-level constants,
  wire `SIGINT` and `SIGTERM` to a synchronous `shutdown()`, cancel asyncio
  tasks first, then call `ep.stop()` + `ep.close()`. Voice workers delegate
  readiness, signals, pipeline cancellation, and cleanup to `VoiceSession`.
- Callbacks are `async def` even if the work inside is sync.
- CPU-bound work goes through `loop.run_in_executor(...)` — never block the
  event loop.
- New and migrated workers are named packages with relative internal imports,
  an explicit `__init__.py`, and a `__main__.py` entry point.

**Checklist for a new sample:**

- [ ] `agent-samples/<name>/pyproject.toml` — orchestrator, deps: `xr-ai-launcher` only
- [ ] `agent-samples/<name>/worker/pyproject.toml` — worker, narrow task deps; package `<snake_name>_worker`
- [ ] `agent-samples/<name>/main.py` — exact orchestrator boilerplate
- [ ] `agent-samples/<name>/worker/<snake_name>_worker/` — package with
      `__init__.py`, `__main__.py`, and cohesive sibling modules
- [ ] `agent-samples/<name>/yaml/xr_media_hub.yaml` — hub config
- [ ] `agent-samples/<name>/yaml/<command>.yaml` — one per process that needs config
- [ ] `agent-samples/<name>/yaml/models.local.json` — structured adapter,
      endpoint, and deployment specs; worker-only legacy YAML remains supported
      (see `agent-sdk/xr-ai-models/README.md`)
- [ ] `uv sync` in both `agent-samples/<name>/` and `agent-samples/<name>/worker/`
- [ ] `agent-samples/<name>/README.md` — sample-specific setup and operation
- [ ] Root `README.md` updated — sample tour and quickstart

Boilerplate templates (orchestrator, worker, `pyproject.toml`): `docs/adding-a-sample.md`.
Reference implementation: `agent-samples/simple-vlm-example/`.

## Agent sample architecture: reusable modules

Samples must **reuse** the shared building blocks rather than re-implement
them. They split across SDK packages by what they depend on:

Typed agent functions live in `xr-ai-nat`. `SpatialMathFunctionsConfig`
registers deterministic coordinate operations that receive an explicit spatial
frame; tracking and process boundaries remain outside the math functions.
`TextMemoryFunctionsConfig` provides persistent timestamped text without a
network boundary. `VisionToolsConfig` (`xr_vision_tools`) adds
`look_at_current_frame` / `look_at_past_frame` question answering over an
injected `xr-ai-models` VLM, acquiring the participant's live or recorded frame
itself. `XRTrackingFunctionsConfig` exposes the current user frame through
the typed OpenXR service without routing native agents through MCP.
`VideoMemoryFunctionsConfig` exposes recorded-video discovery, queries, and
frame extraction through a typed service while keeping MCP optional; callers
obtain current frames through the hub client. `StreamingVisionConfig` composes
raw frame acquisition with VLM streaming behind one native function for voice
workflows. `ModelsLLMConfig` adapts the `xr-ai-models` service boundary to
NAT's built-in LangChain-backed agent types; applications install
`xr-ai-nat[agents]` rather than calling LangChain model clients directly.

The public **native voice runtime** lives in `xr-ai-voice` (it depends on
pipecat internally):

- **Voice session** — `VoiceSession.run(handler)` privately assembles
  `input → VadStt → VoiceGate → handler → StreamingTts → output`, owns model
  readiness and ready-file semantics, installs signal handlers, and closes the
  transport and model clients. It touches the ready file only after the input
  transport has entered its hub IPC receive loop.
- **Native handler** — `xr_ai_nat.adapters.as_voice_handler` maps a typed NAT
  function onto `VoiceSession`; `TextMessageInput` routes participant text
  through the same turn path as speech.
- **Wake word / speech gate** — `xr-ai-voicegate` (the `VoiceGate` state
  machine) wired in as `VoiceGateProcessor`; per-sample config in
  `yaml/voice_gate.yaml` (`magic_phrases: ["hey agent"]`, or `[]` for
  always-on). No sample code — config only.

`xr-ai-pipecat` remains available for samples that still subclass its
`BrainProcessor`; those workers run the assembled pipeline with
`run_voice_pipeline(worker, transport, on_ready=ready_file.touch)` so they use
the same IPC-start readiness boundary.

A native voice sample adapts its NAT function to `VoiceSession`; wake-word
behavior comes from config alone.

### Scope decision and named follow-ups

The function/pipeline boundary is explicit: `xr-ai-pipecat` stays "voice
pipeline plumbing", while reusable typed agent functions live in `xr-ai-nat`.
Application-specific capabilities stay with their application;
`xr-ai-pipecat` must not become a catch-all.
Planned structural follow-ups (own PRs):

1. **`MCPToolset`.** `RenderSceneProcessor` still takes one `McpClient` per MCP
   server and routes via a hardcoded `_execute_tool` switch — adding a server
   means a new arg + frozenset + branch, reusable by nothing. Replace it with a
   toolset that pairs a client with the tool names it owns, so a brain accepts a
   list and auto-routes:

   ```python
   brain = AgentBrain(
       transport=transport, llm=llm,
       toolsets=[MCPToolset(oxr, _OXR_TOOLS), MCPToolset(render)],  # render = catch-all
   )
   ```

2. **Hardware-capability gating.** A declared `HardwareProfile` → derived
   features → a pipeline that auto-includes only the stages the device supports
   (no mic → text input; no camera → vision off). Prototyped and deferred.

## Documentation rule

Update `README.md` (and relevant sub-repo docs) **in the same task** as the
code change. A change is not done until the docs reflect it. This applies to
new packages, changed entry points, new quickstart flows, renamed commands,
and new config files.

## Prompt-driven samples: write eval cases

When a sample's behaviour is driven by an LLM prompt (e.g.
`agent-samples/xr-render-demo/`):

- When you add or change a rule in `system.txt`, add or update a case
  in the sample's `eval/` harness in the same edit. A rule without a
  case is unverified.
- **Don't train on the test set.** Don't reuse a prompt's worked-
  example specifics (coordinates, colors, shapes, trigger phrases) in
  a case fixture, or vice versa — that makes the eval a memorization
  check. The harness audits this at startup and warns; clear the
  warning by changing the prompt example, not the case.

## Dependency discipline

`DEPENDENCIES.md` at the repo root is the authoritative dependency map.
Any change to a `pyproject.toml` must update `DEPENDENCIES.md` in the same
commit. A change is not complete until `DEPENDENCIES.md` reflects it.

Hard rules (also in `DEPENDENCIES.md`):

- `utils/xr-ai-launcher/` has zero runtime dependencies — stdlib only. Keep it that way.
- `utils/xr-ai-logging/` depends only on `loguru>=0.7`. Used by every process via `setup_logging()`.
- `agent-sdk/xr-ai-hub-client` depends only on `pyzmq` + `msgpack`.
- `agent-sdk/xr-ai-models` depends only on `xr-ai-logging` + `httpx` + `pyyaml`. No vendor SDKs.
- Agent workers import only from `xr_ai_hub` + `xr_ai_models` (and task-specific libs).
- Agent workers must never import from `xr_media_hub` or `xr_ai_launcher`.
- Don't add abstractions until needed by two concrete use-cases.

## License headers

Every new source file gets the SPDX header at the top:

```
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

Comment-style table, file-type exceptions, and enforcement: `docs/spdx-headers.md`.

## Comments

Write comments for the next person reading the code, not as a record of how
the code came to exist. The two questions a comment must answer are
"what non-obvious thing does this do?" or "why isn't the obvious version
correct?". If a comment doesn't answer one of those, delete it.

Concrete rules:

- **No play-by-play.** Don't narrate the debugging journey, the things you
  tried first, or the alternatives you ruled out. The current code is the
  decision; the comment exists to make it readable, not to argue for it.
- **No "we discussed" / "decided not to" / "for now" / "originally"**.
  Future readers don't have your context and don't need it. If the rationale
  is genuinely load-bearing, put one sentence stating the invariant ("must
  be 2D — NVENC reads strides"), not a paragraph reconstructing how you
  found out.
- **No restating the code.** `// loop over participants` above a
  `for pid in participants:` is noise.
- **One sentence is usually enough.** Two sentences if the "why" needs a
  concrete failure mode. A multi-paragraph block comment almost always
  means the comment is doing the wrong job — either the code needs
  refactoring or the content belongs in `docs/changelog.md`.
- **Architectural rationale and historical context belong in
  `docs/changelog.md`**, not in source comments. Source comments are read
  every time someone touches the line; the changelog is read when someone
  needs the history.
- **Same rules apply to docstrings and README sections** added by an
  agent. Lead with the contract; don't recap the design conversation.

When in doubt, prefer the shorter comment. A future reader can read the
git log; they cannot un-read a wall of text wrapping a one-liner.

**Scope**: apply this only to comments you are writing or to comments on
lines you are already changing. Don't open existing files just to trim
comments — that's out of scope for any task other than an explicit
"clean up comments in <file>" request, and creates churn that obscures
the real change in review.

## docs/ index

Read these on demand when the topic comes up:

| File | When to read |
|---|---|
| `docs/architecture.md` | Working across module boundaries; understanding hub ↔ transport ↔ agent boundaries; the same-origin wss:// signaling proxy in front of LiveKit |
| `docs/process-model.md` | Touching `utils/xr-ai-launcher/`, orchestrators, ready-files, or adding a managed process type |
| `docs/credentials.md` | Code that needs `HF_TOKEN` / `NGC_API_KEY` |
| `docs/ai-services.md` | Adding, calling, or operating a VLM / STT / TTS / LLM / embedding server (incl. vLLM persistence) |
| `docs/xr-render-demo.md` | Working inside `agent-samples/xr-render-demo/` — process stack, LLM roles, agentic loop, XR lifecycle |
| `docs/adding-a-sample.md` | Scaffolding a new sample — full boilerplate templates |
| `docs/adding-cloudxr.md` | Wiring CloudXR into a sample |
| `docs/spdx-headers.md` | SPDX comment styles, exceptions, enforcement |
| `docs/networking.md` | Firewall ports, TLS for the web client |
| `docs/troubleshooting.md` | Known frictions, first-time setup gotchas, runtime symptoms |
| `docs/changelog.md` | Why something is the way it is — significant decisions in reverse chronological order |

Record significant new decisions in `docs/changelog.md` (reverse chronological).
