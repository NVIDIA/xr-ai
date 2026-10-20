<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai working contract

This file describes the XR AI implementation, shared engineering guidance, and
upstream contribution policies. Read the nearest package or sample README for
its entry point and canonical-doc links. XR AI's user-facing docs live under
`docs/source/`.

Building an application? Read
[`docs/source/guides/building-your-app.md`](docs/source/guides/building-your-app.md).

<a id="change-contract"></a>

Before opening an XR AI pull request, read
[`docs/source/guides/contributing-and-conventions.md`](docs/source/guides/contributing-and-conventions.md).

Refer to [Skills](docs/source/getting_started/skills.md) for available
coding-agent skills and installation.
For requests to try xr-ai or run its Simple VLM browser demo, follow
[`skills/run-demo/SKILL.md`](skills/run-demo/SKILL.md).

## Repository map

```text
client-samples/  Platform clients
agent-sdk/       Agent runtime, hub IPC, model, tool, and voice libraries
agent-samples/   Runnable agent stacks
deployment/      External model endpoint profiles and validation
apps/            Application workspaces outside repository sample tooling
services/        Hub, model servers, and typed capability services
utils/           Launcher, logging, VAD, vLLM, and voice-gate utilities
tests/           Cross-package and integration tests
docs/source/     User and contributor documentation
third_party_licenses/  Unmodified third-party license and attribution texts
skills/          Coding-agent skills for setup, demos, development, and reviews
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
  workers or services. One scoped exception: The supported Riva speech NIM integration uses
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
  process boundaries use typed msgpack over ZMQ services, not MCP.
- Image selection and visual inference are separate tools. Selection returns
  lightweight image references; single-image, multi-image, and timestamped
  video-frame query tools resolve them through `VLMService`. Raw media remains
  on the hub path and out of tool results. Latest recorded windows need only a
  duration; historical frame and video selection share an absolute `start_us`.

The authoritative dependency graph and enforced package limits are in
[`DEPENDENCIES.md`](DEPENDENCIES.md).

## Process and sample layout

Each agent sample contains a stdlib-style orchestrator and a separately packaged
worker. Optional application-specific capability processes sit beside them.

```text
agent-samples/<kebab-name>/
  main.py
  pyproject.toml
  README.md
  yaml/
  model-servers/  Convenience launcher and sample-owned GPU-profile YAML
  worker/
    pyproject.toml
    <snake_name>_worker/
      __init__.py
      __main__.py
```

The reference application workflow uses the sample layout under
`apps/<kebab-name>/`. Repository dependency, lint, SPDX, sample catalog, and
test discovery exclude that top-level directory.

- Model-server convenience stacks start from each service's own YAML under
  `services/` and apply sample-owned overrides under `model-servers/yaml/`. Stop the current stack before changing its settings,
  switching samples, or updating the checkout; these launchers need no launch
  history or cross-version reconciliation. Refer to
  [Model-server workflow](docs/source/guides/customizing-model-servers.md).
- Developer samples import public XR AI package APIs. Keep private module and
  symbol imports inside their owning packages or implementation tests.
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

When preparing a repository sample, refer to
[Adding a sample](docs/source/guides/adding-a-sample.md) and the
[`conversation-example`](agent-samples/conversation-example/README.md) reference.

## Engineering guidance

Apply these practices to repository and downstream work. Work under `apps/`
uses this engineering guidance. For changes elsewhere in this checkout, also
follow [Contributing and conventions](docs/source/guides/contributing-and-conventions.md)
unless the user identifies the work as downstream-only.

- Reuse an existing mechanism before adding one, and follow the nearest
  sibling's lifecycle, configuration, and ownership patterns. Fix behavior at
  its owning boundary; avoid duplicating checks already owned by a dependency.
  Do not add an abstraction until two concrete use cases need it.
- After changing names or behavior, update affected callers, configuration,
  tests, and documentation within the project. Check equivalent sibling paths
  for the same defect.
- Tests must prove the changed production behavior. Follow
  [Test design](docs/source/guides/testing.md#test-design).
- Verify that declared dependency minimums provide the APIs the code uses.
  Refer to [Dependency API compatibility](docs/source/guides/contributing-and-conventions.md#dependency-api-compatibility).
- Validate configuration and requests at their owning boundary. Reject unknown
  keys unless the schema explicitly permits extensions. Reject conflicting
  settings, malformed payloads, and explicitly configured missing files with
  actionable errors. Defaults apply to omitted optional input, not invalid
  explicit input.
- Remote control handlers validate the sender's authority as well as the topic
  and payload; a topic name alone grants no authority. Keep camera, microphone,
  and similar sensitive capabilities off until the user explicitly enables
  them. Bound path and URL inputs to the intended resource access policy.
- HTTP and WebSocket clients connecting to repository-launched services on
  loopback must opt out of ambient proxy routing with `trust_env=False`,
  `proxy=None`, or the equivalent for their client library.
- Resource owners release only resources they still own, including after
  cancellation or replacement. Apply the
  [lifecycle rules](docs/source/guides/contributing-and-conventions.md#resource-lifecycle).
- Never put API keys or tokens in source files. Use environment variables or
  the project's credential store. XR AI's supported sources are documented in
  `docs/source/getting_started/credentials.md`.
- When behavior is driven by an LLM prompt, changing a rule requires a
  corresponding eval case. Do not reuse worked-example specifics in the eval
  fixture.
- Preserve unrelated work in a dirty tree. Never use destructive Git commands
  to discard user changes.
- Check factual claims in docs, comments, and docstrings against the code.
  Refer to [Factual accuracy](docs/source/guides/documentation-style.md#factual-accuracy)
  for conditions and failure paths to cover.

## Comments and documentation

Comments explain a non-obvious invariant or failure mode. Do not narrate the
code, debugging history, rejected alternatives, or plans. Keep architectural
docs about the current system; issue trackers and Git history hold proposals
and past decisions. Document state and ownership invariants where the state is
declared.

Use these canonical references when working in their area:

| Topic | Document |
|---|---|
| Architecture | [`docs/source/overview/architecture.md`](docs/source/overview/architecture.md) |
| Processes | [`docs/source/components/launcher-and-process-model.md`](docs/source/components/launcher-and-process-model.md) |
| Agent SDK | [`docs/source/components/agent-sdk.md`](docs/source/components/agent-sdk.md) |
| AI services | [`docs/source/components/ai-services.md`](docs/source/components/ai-services.md) |
| Credentials | [`docs/source/getting_started/credentials.md`](docs/source/getting_started/credentials.md) |
| Networking | [`docs/source/getting_started/networking.md`](docs/source/getting_started/networking.md) |
| Testing | [`docs/source/guides/testing.md`](docs/source/guides/testing.md) |
| Documentation style | [`docs/source/guides/documentation-style.md`](docs/source/guides/documentation-style.md) |
| Python code style | [`docs/source/guides/python-code-style.md`](docs/source/guides/python-code-style.md) |
| Troubleshooting | [`docs/source/guides/troubleshooting.md`](docs/source/guides/troubleshooting.md) |
| xr-render-demo | [`docs/source/reference/xr-render-demo.md`](docs/source/reference/xr-render-demo.md) |
