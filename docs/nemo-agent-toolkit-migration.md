<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# NeMo Agent Toolkit migration

XR AI is retiring its NeMo Agent Toolkit dependency without changing the
application boundary: agents remain in-process XR workers, use `xr-ai-models`
for all model HTTP, and reach clients only through the Hub IPC SDK.

## Target architecture

NeMo Relay is the local execution boundary. The XR-owned `xr-ai-nat` package
is becoming a toolkit-independent tools layer: Pydantic tool schemas, trigger
dispatch, and the generic `AgentRunner` protocol. Its bundled agent is a
bounded tool loop, while `as_agent_tool(...)` lets custom or future
Fabric-backed runners use the same trigger path. Relay owns LLM and tool
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
  -> xr-ai-nat: typed tools and trigger dispatch
  -> NeMo Relay: managed tool and agent execution, guardrails, telemetry
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

1. **Native tools foundation** — make `xr-ai-nat` independent of NeMo Agent
   Toolkit by default, add Relay-managed tools, the generic `AgentRunner`
   seam, and a bounded default tool loop, and retain existing function groups
   behind legacy extras.
2. **Simple VLM tool** — add a finite current-frame tool for agentic flows,
   retain streaming only for its direct-voice responder, and prove the
   lightweight sample selects no legacy extra.
3. **Native event dispatcher** — port tea-making's typed participant-scoped
   subscriptions and periodic background sources so voice and autonomous work
   invoke the same registered tools.
4. **Deterministic and service capabilities** — port spatial math, text memory,
   RAG, vision, XR tracking, and video memory to the native tool surface; keep
   MCP adapters as explicit compatibility publishers.
5. **Existing agent workflows** — port render-demo's tool catalog and evaluation
   harness to the tools layer, then replace its NAT builder and LangChain bridge.
6. **Tea-making sample** — land the sample in slices: workflow/state core,
   foreground/background application manager, observation applications, then
   activity viewer and end-to-end evaluations.
7. **Retirement and operations** — remove all `nat.*`, `nvidia-nat-*`, and
   `nemo_toolkit` legacy references; remove lockfile and notice entries;
   add opt-in Platform evaluation/deployment and Fabric harness profiles.

Every migration PR must preserve participant scoping, add or update the direct
tests for the surface it changes, update `DEPENDENCIES.md`, and leave no
hand-rolled model HTTP client behind.

## Exit criteria

The retirement is complete only when repository-wide search finds no runtime
imports from `nat` and no `nvidia-nat-*` dependency, worker requirements select
only toolkit-independent `xr-ai-nat` extras, all migrated samples pass their
unit and evaluation suites, and `THIRD_PARTY_NOTICES.md` no longer lists NeMo
Agent Toolkit.
