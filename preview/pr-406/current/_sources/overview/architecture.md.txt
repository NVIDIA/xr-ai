<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Architecture

XR AI connects interactive clients, including XR glasses and headsets, phones,
and web browsers, to agent workers and AI services without making transport,
media, or model implementation details part of application logic. The system
is process-oriented: the device I/O hub, workers, model servers, and optional
capability services run independently and communicate through small, explicit
interfaces.

This page describes the system shape, runtime paths, and ownership boundaries.
The component pages linked under [Where details live](#where-details-live)
contain configuration, protocol, and operational details.

## System context

```text
                                     control plane
                          sample orchestrator + launcher
                         starts, orders, and monitors processes
                                         |
                                         v
+---------------+   media and data   +----------------+   IPC events   +----------------+
| XR clients    | <----------------> | DeviceIOHub    | <------------> | agent workers  |
| web/mobile/XR |                    | + transport    |                | + agent SDK     |
+---------------+                    +----------------+                +-------+--------+
                                                                            |
                                                   typed model/tool calls   |
                                                +---------------------------+--------+
                                                |                                    |
                                                v                                    v
                                      +-------------------+               +--------------------+
                                      | AI model services |               | capability services |
                                      | local or hosted   |               | or local tools     |
                                      +-------------------+               +--------------------+
```

The architecture has four cooperating planes:

- **Media plane:** DeviceIOHub receives client audio, video, and data through
  a transport connector, then fans participant-tagged events out to workers.
- **Application plane:** agent workers own application behavior, participant
  state, model and tool orchestration, concurrency, and cancellation.
- **Service plane:** typed model clients and tools isolate workers from model
  hosting and capability-process protocols.
- **Control plane:** each sample's orchestrator uses `xr-ai-launcher` to start
  processes in dependency order and monitor their readiness and lifetime.

These planes are boundaries of responsibility, not required deployment hosts.
They can run on one machine, while model endpoints can also be persistent local
services or externally hosted APIs.

## Components and ownership

| Component | Owns | Does not own |
|---|---|---|
| XR clients | Device capture, presentation, and user interaction | Agent execution or service orchestration |
| Transport connector | Transport-specific sessions and conversion to hub events | Agent-facing APIs or application policy |
| DeviceIOHub | Media fan-out, participant identity, return routing, and shared-media access | Agent state, model calls, or tools |
| Agent SDK | Lightweight hub IPC, typed runtime events, model protocols, voice composition, and tool primitives | Application lifecycle and decision policy |
| Agent worker | Application state, tasks, prompts, model/tool loops, concurrency, and cleanup | Transport internals or model-server lifecycle |
| AI model services | Inference and model-specific serving behavior | Participant routing or application policy |
| Tools and capability services | Bounded application capabilities | General agent orchestration |
| Sample orchestrator | Process declarations, startup ordering, readiness, and shutdown | Runtime business logic |

The separation lets a worker change model deployment or client transport
without rewriting its application logic. It also keeps heavy transport and
model dependencies out of the minimal agent-to-hub IPC package.

## Runtime data paths

### Client media and participant events

1. A client joins through the configured transport and publishes audio, video,
   and data.
2. The transport connector converts that input into participant-tagged hub
   events. Audio, data, and participant events travel inline; video pixels stay
   in shared memory and frame notifications remain lightweight.
3. DeviceIOHub fans subscribed events out to `ProcessorEndpoint` consumers.
   Multiple agents or passive processors can subscribe to the same input.
4. A worker requests video pixels only when its application needs a frame.
5. Return audio and data name the originating participant. The hub validates
   the target and the connector delivers the response only to that participant.

The resulting portable contract is:

```text
participant -> hub -> subscribed worker -> hub -> same participant
```

LiveKit currently provides the client transport, but it remains behind the hub
boundary. Workers communicate only through XR AI's msgpack/ZMQ IPC and do not
import or address LiveKit directly. See {doc}`Server runtime
</components/server-runtime>` for shared-memory behavior, topics, and transport
implementation details.

### Agent, model, and tool execution

A voice-capable application commonly composes this path:

```text
audio -> voice input -> STT -> application agent -> LLM/VLM -> voice output -> TTS
                                  |                 ^                         |
video -- on-demand frame access --+                 |                         |
                                  +---- tools ------+                         v
                                                               return audio and data
```

This is composition rather than a fixed pipeline. An application may consume
text instead of speech, omit models, execute deterministic tools locally, call
typed capability services, or publish events for another in-process agent
registered with the same `AgentRuntime` to consume.

Workers obtain LLM, VLM, STT, TTS, and embedding clients from
`xr_ai_models`. Model profiles keep three concerns separate:

- adapter behavior and model-specific wire details;
- endpoint location, credentials, and health behavior;
- deployment ownership by the current stack, a reused service, or an external
  provider.

Tools are ordinary in-process `Tool` or `AsyncTool` objects. A tool can perform
local work or call a typed service, but application agents retain ownership of
tool selection, task lifetime, retries, and participant context. See
{doc}`Agent SDK </components/agent-sdk>` and {doc}`AI services
</components/ai-services>` for the concrete interfaces and profiles.

## Process and deployment model

Each sample is an executable process graph declared by a small orchestrator.
DeviceIOHub always runs as its own process; workers and capability services
run separately so their dependencies, failures, and cleanup remain isolated.

Stack items start in declaration order; members of a `Parallel` item start
concurrently, and the next item waits for every member to signal ready. Each
ready-file reports only that process's initialization. A premature process
exit fails the stack and triggers coordinated shutdown. Ready-files order
process startup; they do not determine whether a client may connect.

Model services use the same ownership model without forcing every sample to
reload large weights:

- **Managed:** the current orchestrator starts and owns the service.
- **Reused:** the service is expected to be running already, commonly from the
  shared `model-servers` stack.
- **External:** XR AI connects to an endpoint it neither starts nor stops.

Heavy model servers can use a persistent launch mode and remain hot across
sample restarts. The model profile co-locates process ownership and endpoint
choices used by the orchestrator and worker, reducing configuration drift. The
launcher does not validate an endpoint's `base_url` against the launched
service's separate configuration. Detailed startup, shutdown, and persistence
behavior belongs in {doc}`Launcher and process model
</components/launcher-and-process-model>`.

## Architectural invariants

The following constraints define the supported system boundary:

- **One hub, many clients, many agents.** The hub may fan one participant's
  input out to several consumers, but it never uses the return path to route
  one participant's data to another.
- **Transport details stop at the hub.** Workers use `xr_ai_hub`; transport
  SDKs and server packages do not enter agent APIs.
- **Raw media stays on the media path.** Video pixels remain in shared memory
  until explicitly requested, and raw media is not embedded in runtime events
  or tool results.
- **Applications own behavior and resources.** An agent owns its state,
  background tasks, lifecycle, queues, cancellation, and concurrency policy.
  Shared runtime packages provide delivery and composition rather than taking
  over that ownership.
- **Models are reached through typed factories.** Vendor protocols and
  model-specific quirks remain behind `xr_ai_models`, rather than spreading
  through workers.
- **Process dependencies are explicit.** Orchestrators declare startup order,
  readiness, and ownership instead of relying on import-time side effects or
  an implicit global runtime.
- **Process and client readiness are distinct.** Ready-files order launcher
  items and report only process readiness. DeviceIOHub owns client readiness.
  Agents opt in with `announces_readiness=True`, report only their own
  `_agent.status`, and gate only subscribed participants; the hub aggregates
  their status per participant. Passive processors do not gate clients.

These constraints are the stable architecture. Individual transports, model
backends, tools, and sample workflows are replaceable implementations within
it.

## Extension points

| To add or replace | Extend at this boundary |
|---|---|
| XR client | Join through the client transport and consume participant-scoped return media/data |
| Transport | Implement the connector side of the hub IPC contract |
| Agent application | Add a worker that uses `xr_ai_hub` and the relevant SDK packages |
| Model or provider | Add model configuration and an adapter behind the typed model protocols |
| Local capability | Implement an in-process tool |
| Shared or isolated capability | Implement a typed service and a service-backed tool |
| Managed process | Add a launcher `Process` entry and signal readiness from the process |

For the repository conventions and concrete file layout used by new samples,
see {doc}`Adding a sample </guides/adding-a-sample>`.

## Where details live

This page intentionally stops at system structure and contracts. Use these
pages for implementation and operational detail:

| Topic | Authoritative page |
|---|---|
| Hub IPC, shared memory, participant isolation, and LiveKit integration | {doc}`Server runtime </components/server-runtime>` |
| SDK package boundaries, agents, voice, runtime events, and tools | {doc}`Agent SDK </components/agent-sdk>` |
| Model protocols, deployment profiles, NIM, vLLM, and persistent servers | {doc}`AI services </components/ai-services>` |
| Process ordering, readiness, launch modes, and shutdown | {doc}`Launcher and process model </components/launcher-and-process-model>` |
| External ports, TLS, proxies, and firewall requirements | {doc}`Networking and firewall </getting_started/networking>` |
| API keys and credential storage | {doc}`Credentials </getting_started/credentials>` |
