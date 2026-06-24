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
server-runtime/     # XR-Media-Hub core + LiveKit transport
agent-sdk/          # Four packages:
                    #   xr-ai-agent        — IPC client library (pyzmq + msgpack only)
                    #   xr-ai-models       — LLM/VLM/STT/TTS service protocols + OpenAI-compat clients
                    #   xr-ai-capabilities — framework-agnostic reusable agent features (VisionModule)
                    #   xr-ai-pipecat      — optional Pipecat transport bridge (heavier deps)
utils/              # Shared infra: launcher, logging, vad, vllm, voicegate
cloudxr-runtime/    # Shared CloudXR OpenXR runtime + WSS proxy (opt-in)
ai-services/        # OpenAI-compatible inference servers (VLM, STT, TTS, LLM)
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
- **`agent-sdk/xr-ai-agent` depends only on `pyzmq` + `msgpack`.** No
  LiveKit, FastAPI, or uvicorn. `agent-sdk/xr-ai-pipecat` is a separate
  optional package with heavier deps (pipecat-ai, scipy, numpy, httpx,
  fastmcp); it bridges `ProcessorEndpoint` to Pipecat pipelines.
- **All HTTP calls to AI services go through `agent-sdk/xr-ai-models`.**
  Workers and MCP servers depend on its four protocols
  (`LLMService`, `VLMService`, `STTService`, `TTSService`) and construct
  clients from a per-sample `yaml/models.yaml` via `make_llm` /
  `make_vlm` / `make_stt` / `make_tts`.  Hand-rolled `httpx` clients
  against `/v1/chat/completions`, `/v1/audio/transcriptions`, or
  `/v1/audio/speech` are forbidden — model quirks belong in this one
  package's presets, not in callers. No vendor SDKs (no `openai`, no
  `anthropic`, no `litellm`); all in-tree backends speak
  OpenAI-compatible HTTP.
- **Workers never import from `server-runtime` or `xr_ai_launcher`.** Only
  `xr_ai_agent` + `xr_ai_models` + task-specific libs (numpy, torch, …).
- **MCP servers are the agent's only interface to XR data and rendering.**
- **No API keys or tokens in source files** — use env vars or
  `xr_media_hub.yaml`. See `docs/credentials.md`.

## Process model essentials

Each sample has two sub-projects:

| Sub-project | Role | Dependencies |
|---|---|---|
| `<sample>/` | Orchestrator — declares `PROCESSES`, calls `run_stack` | `xr-ai-launcher` only |
| `<sample>/worker/` | Agent worker — connects to hub via IPC | `xr-ai-agent` + task libs |

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

- Only import from `xr_ai_agent` for IPC types.
- `_HUB_PUB` / `_HUB_PUSH` are module-level constants, not magic strings.
- Wire `SIGINT` and `SIGTERM` to `agent.shutdown()`; wrap `await agent.run()`
  in `try/finally` calling `shutdown()`.
- `shutdown()` is synchronous (signal-handler safe). Cancel asyncio tasks
  first, then `ep.stop()` + `ep.close()`.
- Callbacks are `async def` even if the work inside is sync.
- CPU-bound work goes through `loop.run_in_executor(...)` — never block the
  event loop.
- Imports are absolute (flat module layout). No `__init__.py` or `__main__.py`.

**Checklist for a new sample:**

- [ ] `agent-samples/<name>/pyproject.toml` — orchestrator, deps: `xr-ai-launcher` only
- [ ] `agent-samples/<name>/worker/pyproject.toml` — worker, deps: `xr-ai-agent` + task libs (list every `.py` in `only-include`)
- [ ] `agent-samples/<name>/main.py` — exact orchestrator boilerplate
- [ ] `agent-samples/<name>/worker/<snake_name>_worker.py` — entry point + (optional) split helpers
- [ ] `agent-samples/<name>/yaml/xr_media_hub.yaml` — hub config
- [ ] `agent-samples/<name>/yaml/<command>.yaml` — one per process that needs config
- [ ] `agent-samples/<name>/yaml/models.yaml` — logical model names + preset references (see `agent-sdk/xr-ai-models/README.md`)
- [ ] `uv sync` in both `agent-samples/<name>/` and `agent-samples/<name>/worker/`
- [ ] `README.md` updated — sample tour and quickstart

Boilerplate templates (orchestrator, worker, `pyproject.toml`): `docs/adding-a-sample.md`.
Reference implementation: `agent-samples/simple-vlm-example/`.

## Agent sample architecture: reusable modules

Samples must **reuse** the shared building blocks rather than re-implement
them. They split across two SDK packages by what they depend on:

The **voice pipeline** lives in `xr-ai-pipecat` (it depends on pipecat):

- **Voice pipeline** — `make_voice_pipeline` assembles
  `input → VadStt → VoiceGate → brain → StreamingTts → output`. A sample
  subclasses `BrainProcessor` and hands it to the factory.
- **Wake word / speech gate** — `xr-ai-voicegate` (the `VoiceGate` state
  machine) wired in as `VoiceGateProcessor`; per-sample config in
  `yaml/voice_gate.yaml` (`magic_phrases: ["hey agent"]`, or `[]` for
  always-on). No sample code — config only.

**Reusable capabilities** live in `xr-ai-capabilities` — framework-agnostic
features that talk to the hub through a `ProcessorEndpoint` and depend only on
the core SDK (no pipecat), so both pipecat and non-pipecat agents can compose
them:

- **Live-camera vision Q&A** — `VisionModule` (frame tracking,
  camera-on-demand, the VLM call). Two call styles over one
  frame-acquisition path: `ask(pid, q)` **streams** tokens (for TTS) and
  `perceive(pid, q)` returns a **string** (for agentic tool loops, raising
  `VisionUnavailable`). `pixels` is its frame → JPEG codec. A pipecat brain
  constructs it with `VisionModule(transport.endpoint, vlm)`.

A new vision sample's brain therefore reduces to thin glue over `VisionModule`;
a voice sample gets the wake word from config alone.

### Scope decision and named follow-ups

The capabilities/pipeline boundary is now explicit: `xr-ai-pipecat` stays "voice
pipeline plumbing" and reusable capabilities live in `xr-ai-capabilities`, which
has no pipeline-framework dependency. Keep new reusable agent building blocks in
`xr-ai-capabilities` — `xr-ai-pipecat` must not become a catch-all.
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
       capabilities=[VisionModule(transport.endpoint, vlm)],
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
- `agent-sdk/xr-ai-agent` depends only on `pyzmq` + `msgpack`.
- `agent-sdk/xr-ai-models` depends only on `xr-ai-logging` + `httpx` + `pyyaml`. No vendor SDKs.
- Agent workers import only from `xr_ai_agent` + `xr_ai_models` (and task-specific libs).
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
| `docs/ai-services.md` | Adding, calling, or operating a VLM / STT / TTS / LLM server (incl. vLLM persistence) |
| `docs/xr-render-demo.md` | Working inside `agent-samples/xr-render-demo/` — process stack, two-LLM split, agentic loop, XR lifecycle |
| `docs/adding-a-sample.md` | Scaffolding a new sample — full boilerplate templates |
| `docs/adding-cloudxr.md` | Wiring CloudXR into a sample |
| `docs/spdx-headers.md` | SPDX comment styles, exceptions, enforcement |
| `docs/networking.md` | Firewall ports, TLS for the web client |
| `docs/troubleshooting.md` | Known frictions, first-time setup gotchas, runtime symptoms |
| `docs/changelog.md` | Why something is the way it is — significant decisions in reverse chronological order |

Record significant new decisions in `docs/changelog.md` (reverse chronological).
