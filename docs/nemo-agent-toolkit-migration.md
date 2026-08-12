<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# NeMo Agent Toolkit migration

XR AI is retiring its NeMo Agent Toolkit dependency without changing the
application boundary: agents remain in-process XR workers, use `xr-ai-models`
for all model HTTP, and reach clients only through the Hub IPC SDK.

## Target architecture

NeMo Relay is the local execution boundary. The XR-owned `xr-ai-tools` package
is the toolkit-independent tools layer: Pydantic tool schemas, trigger
dispatch, and small helpers that adapt those tools to `xr-ai-models` tool-call
types. `xr-ai-agent-runtime` separately owns agent resource lifetimes,
runtime-owned background tasks, and fan-out `publish()`. Agents expose the
existing `Tool` and `AsyncTool` objects directly and own their synchronization.
Applications own their model calls, history, and loop policy. Relay owns tool
lifecycles, middleware, guardrails, and telemetry. Existing NeMo Agent Toolkit
function groups remain compatibility extras until their concrete tools migrate.

NeMo Platform and NeMo Fabric are deployment and evaluation integrations, not
worker dependencies. Platform currently requires Python 3.12 or 3.13 and owns
local services, evaluation, tuning, and deployments; Fabric runs a selected
agent harness in a configured environment. XR AI supports Python 3.11 and 3.12
and needs direct local Hub, camera, and voice ownership, so both integrations
remain optional launch targets after the local runtime has migrated.

```text
XR worker
  -> xr-ai-agent-runtime: agent lifetimes, background tasks, and publish
  -> xr-ai-tools: typed tools and trigger dispatch
  -> NeMo Relay: managed tool execution, guardrails, telemetry
  -> xr-ai-models: private model boundary used by model-backed tools
  -> Hub IPC: media and client data

Optional operations path
  -> NeMo Platform: evaluation, tuning, deployment, monitoring
  -> NeMo Fabric: configured harness execution
```

## Tea-making acceptance reference

`origin/devdeepr/tea-making-sample` is the behavioral reference. Its eventual
port must retain deterministic foreground ownership, independently lifecycled
background applications, typed state commits, evidence-gated workflow progress,
RAG only for missing known-product guidance, participant isolation, and the
read-only activity viewer. The branch currently uses NAT, so it defines
acceptance behavior rather than an implementation dependency.

## Focused PR sequence

1. **Native tools foundation** — add `xr-ai-tools` with Relay-managed
   tools plus model tool-call dispatch helpers, and retain existing function
   groups behind legacy extras.
2. **Simple VLM tool** — add a finite current-frame tool for agentic flows,
   retain streaming in a separate transport-independent async tool,
   and prove the lightweight sample selects no legacy extra.
3. **Native event dispatcher** — port tea-making's typed participant-scoped
   subscriptions and periodic background sources so voice and autonomous work
   invoke the same registered tools.
4. **Agent runtime** — add runtime-owned agent resource lifetimes, background
   tasks, typed `publish()`, and agents that expose existing unary and streaming
   tools directly while owning their concurrency policy.
5. **Deterministic and service capabilities** — port spatial math, text memory,
   RAG, vision, XR tracking, and video memory to the native tool surface; keep
   MCP adapters as explicit compatibility publishers.
6. **Existing agent workflows** — port render-demo's tool catalog and evaluation
   harness to the tools layer, then replace its NAT builder and LangChain bridge.
7. **Tea-making sample** — land the sample in slices: workflow/state core,
   foreground and resident agents, observation applications, then activity
   viewer and end-to-end evaluations.
8. **Retirement and operations** — remove all `nat.*`, `nvidia-nat-*`, and
   `nemo_toolkit` legacy references; remove lockfile and notice entries;
   add opt-in Platform evaluation/deployment and Fabric harness profiles.

Every migration PR must preserve participant scoping, add or update the direct
tests for the surface it changes, update `DEPENDENCIES.md`, and leave no
hand-rolled model HTTP client behind.

Fabric integration should adapt the runtime lifetime at the hosting boundary
or provide an `Agent` implementation. It must use the existing tool APIs rather
than introduce a second application invocation path alongside `publish()`.

## Exit criteria

The retirement is complete only when repository-wide search finds no runtime
imports from `nat` and no `nvidia-nat-*` dependency, worker requirements select
`xr-ai-tools` for toolkit-independent tools, all migrated samples pass their
unit and evaluation suites, and `THIRD_PARTY_NOTICES.md` no longer lists NeMo
Agent Toolkit.
