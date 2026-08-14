<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai working contract

This file contains the repository-wide constraints for humans and agents. Read
the nearest package or sample README for local details. User-facing docs live
only under `docs/source/`.

## Repository map

```text
client-samples/  Platform clients
agent-sdk/       Agent runtime, hub IPC, model, tool, and voice libraries
agent-samples/   Runnable agent stacks
services/        Hub, model servers, and typed capability services
utils/           Launcher, logging, VAD, vLLM, and voice-gate utilities
tests/           Cross-package and integration tests
docs/source/     User and contributor documentation
skills/          Skill bank that sets coding agents up
```

## Architecture boundaries

- One hub serves many clients and agents. It fans inbound messages to subscribed
  `ProcessorEndpoint`s and routes return traffic only to the originating client.
- Agents communicate with the hub through `xr_ai_hub` IPC. LiveKit is internal
  to the hub and must not appear in agent APIs.
- The hub owns client readiness. Agents report only their own `_agent.status`.
  Readiness participation is opt-in with `announces_readiness=True`, applies
  only to subscribed participants, and must never be published directly by a
  worker.
- `agent-sdk/xr-ai-hub/` depends only on `pyzmq` and `msgpack`.
- All model HTTP goes through the typed services and factories in
  `xr_ai_models`. Do not add vendor SDKs or hand-written model HTTP clients to
  workers or services. One scoped exception: Riva speech NIMs expose only
  gRPC, so the optional `riva` extra adds `nvidia-riva-client` behind
  `kind: riva_grpc`, with the import deferred into `make_stt`/`make_tts`;
  the base install stays httpx-only.
- Workers never import `device_io_hub` or `xr_ai_launcher`. Use public SDK
  packages and task-specific libraries.
- Tools are native, in-process `Tool` or `AsyncTool` instances from
  `xr_ai_tools`. Every execution passes through NeMo Relay. The repository does
  not ship MCP compatibility servers.
- Agents call another agent's tools directly with `execute()` or `stream()`.
  `xr_ai_runtime` provides typed publish/fan-out and owns only its delivery
  tasks. Agents own their state, resources, lifecycle, background tasks, and
  concurrency policy.
- Application-specific capabilities stay with their application. Shared
  process boundaries use typed msgpack/ZMQ services, not MCP.
- Image selection and visual inference are separate tools. Selection returns
  lightweight image references; single-image, multi-image, and timestamped
  video-frame query tools resolve them through `VLMService`. Raw media remains
  on the hub path and out of tool results. Latest recorded windows need only a
  duration; historical frame and video selection share an absolute `start_us`.

The authoritative dependency graph and enforced package limits are in
[`DEPENDENCIES.md`](DEPENDENCIES.md).

## Process and sample layout

Each sample contains a stdlib-style orchestrator and a separately packaged
worker. Optional application-specific capability processes sit beside them.

<<<<<<< HEAD
```text
agent-samples/<kebab-name>/
  main.py
  pyproject.toml
  README.md
  yaml/
  worker/
    pyproject.toml
    <snake_name>_worker/
      __init__.py
      __main__.py
=======
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

- Import IPC types from `xr_ai_hub`; new and migrated native agent tools come
  from `xr_ai_tools`, model clients from `xr_ai_models`, and the native voice
  runtime from `xr_ai_voice`.
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

Typed agent tools live in `xr-ai-tools`. Its pure spatial functions receive an
explicit frame; tracking and process boundaries remain outside the math.
`TextMemoryTools` provides persistent timestamped text and conversation recall
without a network boundary. `LiveVisionTool` and `HistoricalVisionTool` acquire
the participant's live or recorded frame and call an injected `xr-ai-models`
VLM. `TrackingTools`, `VideoMemoryTools`, and `RAGTools` call their private
typed services over the shared msgpack/ZMQ RPC boundary. `StreamingVisionTool` in `xr-ai-tools` composes raw frame acquisition with VLM streaming and stays independent of voice.

The public **native voice runtime** lives in `xr-ai-voice` (it depends on
pipecat internally):

- **Voice agent** — `VoiceAgent` owns `VoiceSession`, publishes accepted speech
  and typed text as its `UserQuery` schema on a sample-named topic, publishes
  participant and interruption events on sample-named topics, and subscribes
  to `voice.output`. Application agents subscribe to lifecycle events and own
  their cleanup; the application entry point only composes them. Runtime
  publication provides acknowledged delivery through the same output path. The
  private session owns model readiness, ready-file semantics, signals, the
  media pipeline, and cleanup.
- **Wake word / speech gate** — `xr-ai-voicegate` (the `VoiceGate` state
  machine) wired in as `VoiceGateProcessor`; per-sample config in
  `yaml/voice_gate.yaml` (`magic_phrases: ["hey agent"]`, or `[]` for
  always-on). No sample code — config only.

`xr-ai-pipecat` remains a compatibility package, but current samples use
`VoiceSession` and do not import its processors or pipeline assembly APIs.
Pipecat stays private inside `xr-ai-voice`.

A native voice sample registers `VoiceAgent` and application agents on the same
runtime; wake-word behavior comes from config alone.

### Scope decision and named follow-ups

The function/pipeline boundary is explicit: `xr-ai-pipecat` stays "voice
pipeline plumbing", while new and migrated reusable typed agent tools live in
`xr-ai-tools`.
Application-specific capabilities stay with their application;
`xr-ai-pipecat` must not become a catch-all.
Planned structural follow-up:

1. **Hardware-capability gating.** A declared `HardwareProfile` → derived
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

- When you add or change a rule in any agent prompt (xr-render-demo:
  `supervisor_prompt.txt` and `agents/*/prompt.txt`), add or update a
  case in the sample's `eval/` harness in the same edit. A rule without
  a case is unverified.
- **Run the cheap gate after every prompt or ops edit** (xr-render-demo:
  `uv run xr_render_demo_eval utterances`). Prompt edits contaminate
  neighboring behaviors, and full-suite variance hides single-case
  damage; debug at the lowest tier that reproduces a failure and rerun
  the tiers above it before calling a fix done.
- **Don't train on the test set.** Don't reuse a prompt's worked-
  example specifics (coordinates, colors, shapes, ids, trigger phrases)
  in a case fixture, or vice versa — that makes the eval a memorization
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
>>>>>>> 06ad511 (Migrate XR render sample to NAT and improve eval)
```

- Orchestrators depend on `xr-ai-launcher`, declare `PROCESSES`, and call
  `run_stack`.
- Stack items start in declaration order; members of a `Parallel` item start
  concurrently. The launcher waits for each item to signal ready, and fails the
  stack if any process exits.
- `device_io_hub` always runs as its own process.
- Raw IPC workers keep hub addresses as module constants, use async callbacks,
  cancel tasks before `ProcessorEndpoint.stop()` and `.close()`, and move
  CPU-bound work to an executor.
- Voice workers delegate readiness, signals, pipeline cancellation, and cleanup
  to `VoiceAgent`; its media session is private.
- New worker code is a named package with relative internal imports and an
  explicit `__main__.py`.

See [Adding a sample](docs/source/guides/adding-a-sample.md) and the
[`simple-vlm-example`](agent-samples/simple-vlm-example/README.md) reference.

## Change contract

- Public Python API names, signatures, types, defaults, and field behavior live
  in `__all__`, declarations, and co-located docstrings; Sphinx generates their
  reference pages. Update narrative README or `docs/source/` content only when
  concepts, workflows, operations, or architecture change. Breaking changes
  also require a migration entry.
- Top-level sample commands and options live in their `[project.scripts]` and
  `argparse` declarations; Sphinx generates the user-facing CLI reference.
- Sample configuration values and field guidance live in checked-in YAML/JSON
  and adjacent YAML comments. Files under a top-level sample's `yaml/` tree or
  beside a direct capability subproject are generated into the config reference.
- After any `pyproject.toml` change, run
  `uv run --script .github/scripts/generate_dependency_map.py`; the pre-commit hook
  normally regenerates the Python inventory automatically and CI rejects drift.
  Do not hand-edit the generated section in `DEPENDENCIES.md`. Regenerate the
  affected project's gitignored `uv.lock` locally to verify resolution.
- Never put API keys or tokens in source files. Use environment variables or
  the credential store documented in `docs/source/getting_started/credentials.md`.
- Do not add an abstraction until two concrete use cases need it.
- When a sample's behavior is driven by an LLM prompt, changing a rule in its
  `system.txt` requires a corresponding eval case. Do not reuse worked-example
  specifics in the eval fixture.
- New source files need the repository SPDX header. File-type rules and
  exceptions are in [SPDX headers](docs/source/guides/spdx-headers.md).
- Preserve unrelated work in a dirty tree. Never use destructive Git commands
  to discard user changes.

## Comments and documentation

Comments explain a non-obvious invariant or failure mode. Do not narrate the
code, debugging history, rejected alternatives, or plans. Keep architectural
docs about the current system; issue trackers and Git history hold proposals
and past decisions.

Use these canonical references when working in their area:

| Topic | Document |
|---|---|
| Architecture | [`docs/source/overview/architecture.md`](docs/source/overview/architecture.md) |
| Processes | [`docs/source/components/launcher-and-process-model.md`](docs/source/components/launcher-and-process-model.md) |
| Agent SDK | [`docs/source/components/agent-sdk.md`](docs/source/components/agent-sdk.md) |
| AI services | [`docs/source/components/ai-services.md`](docs/source/components/ai-services.md) |
| Credentials | [`docs/source/getting_started/credentials.md`](docs/source/getting_started/credentials.md) |
| Networking | [`docs/source/getting_started/networking.md`](docs/source/getting_started/networking.md) |
| Troubleshooting | [`docs/source/guides/troubleshooting.md`](docs/source/guides/troubleshooting.md) |
| xr-render-demo | [`docs/source/reference/xr-render-demo.md`](docs/source/reference/xr-render-demo.md) |
