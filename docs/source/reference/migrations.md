<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Release migration

## Overview

Required changes when upgrading from `v0.4.0-beta` to the checked-out version.
`main` is unreleased. Apply the entries relevant to your application's SDK
usage, configuration, clients, and services.

## Changes since v0.4.0-beta

If upgrading from an older release, first apply
{doc}`migration-history/v0.3.0-beta-to-v0.4.0-beta`. Refer to
{doc}`migration-history` for earlier guides and apply them in release order.

### NVIDIA vLLM container

Docker-mode model services now default to `nvcr.io/nvidia/vllm:26.11-py3`
(CUDA 13.5, NVIDIA vLLM 0.33.0). Stop the running model stack before updating
and restart it to use the new image. Verify host driver compatibility and
qualify your GPU and model workload. Explicit `vllm_image` overrides remain
in effect. Pip-mode environments retain their existing qualified versions.
Refer to {doc}`/components/ai-services` for image metadata and configuration.

### LLM request defaults

`OpenAICompatLLM` now applies `adapter.default_extras` only when a call omits
an optional value. Explicit `max_tokens`, `temperature`, tools and output
constraints override the corresponding defaults. The selected model, caller
messages and method's streaming mode cannot be replaced by extras. If a
profile previously relied on extras overriding those values, move that choice
into the call instead. Constraints supplied through extras are validated
against endpoint capabilities before transport. `OpenAICompatVLM` wraps the
same client, so VLM profile extras follow the same rules.

### Restart Pocket TTS before upgrading

The Pocket TTS launcher now reuses a listener only when `/health` identifies
`service: pocket_tts_server`. A running `v0.4.0-beta` server returns only
`status: ok`, so the updated launcher rejects it and reports that the port is
already in use.

Before updating, stop the model stack with the old checkout's sample command
and the GPU profile used to start it. For example, from
`agent-samples/conversation-example/`:

```bash
uv run --project model-servers conversation_example_model_servers --stop
```

For a standalone deployment, stop Pocket TTS through the supervisor that
started it. Update the checkout, refresh the service environment, and restart
the model stack before workers. Refer to
{doc}`/guides/customizing-model-servers` for model startup and shutdown.

## Earlier releases

(changes-since-v0-3-0-beta)=
Refer to {doc}`migration-history/v0.3.0-beta-to-v0.4.0-beta` for the
completed `v0.3.0-beta` to `v0.4.0-beta` transition.

Refer to {doc}`migration-history` for completed guides.

(swift-microphone-cleanup-errors)=
Refer to [Swift microphone cleanup errors](
migration-history/through-v0.3.0-beta.md#swift-microphone-cleanup-errors).

(streamkit-image-capture-dependencies)=
Refer to [StreamKit image-capture dependencies](
migration-history/through-v0.3.0-beta.md#streamkit-image-capture-dependencies).

(deviceiohub-rename)=
Refer to [DeviceIOHub rename](
migration-history/through-v0.3.0-beta.md#deviceiohub-rename).

(shared-model-server-sample-location)=
Refer to [Shared model-server sample location](
migration-history/through-v0.3.0-beta.md#shared-model-server-sample-location).

(local-speech-service)=
Refer to [Local speech service](
migration-history/through-v0.3.0-beta.md#local-speech-service).

(removed-sdk-compatibility-surfaces)=
Refer to [Removed SDK compatibility surfaces](
migration-history/through-v0.3.0-beta.md#removed-sdk-compatibility-surfaces).

<a id="consumer-model-readiness"></a>
For current worker startup and model availability requirements, refer to
{ref}`consumer-model-readiness`.

(clef-image-input-migration)=
(clef-image-decision-inputs)=
(clef-model-cache-configuration)=
For Clef updates between development checkouts, refer to
{ref}`clef-service-updates`. For image inputs and model cache configuration,
refer to {doc}`clef-flash`.

(routed-agents-publish-their-own-output)=
For routing and conversation output contracts, refer to
[the routing contract](agent-sdk-routing.md#response-progress-and-routing-context)
and {doc}`agent-sdk-conversation`.

(simple-vlm-example-deprecation)=
Refer to [Simple VLM example deprecation](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#simple-vlm-example-deprecation).

(vllm-0-31-security-update)=
Refer to [vLLM 0.31 security update](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#vllm-031-security-update).

(interaction-messages-move-to-xr-ai-interaction)=
Refer to [Interaction messages move to xr-ai-interaction](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#interaction-messages-move-to-xr-ai-interaction).

(coding-agent-setup-skill-rename)=
Refer to [Coding-agent setup skill rename](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#coding-agent-setup-skill-rename).

(dependency-refresh-and-native-tool-callback-exceptions)=
Refer to [Dependency refresh and native tool callback exceptions](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#dependency-refresh-and-native-tool-callback-exceptions).

(voice-output-declares-reply-or-update)=
Refer to [Voice output declares reply or update](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#voice-output-declares-reply-or-update).

(persistent-services-after-startup-failure)=
Refer to [Persistent services after startup failure](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#persistent-services-after-startup-failure).

(nemotron-3-embedding-model)=
Refer to [Nemotron 3 embedding model](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#nemotron-3-embedding-model).

(embedding-service-identity-migration)=
(embedding-sdk-identity-and-rag-cache-contract)=
Refer to [Embedding SDK identity and RAG cache contract](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#embedding-sdk-identity-and-rag-cache-contract).

(shared-accepted-query-display)=
Refer to [Shared accepted-query display](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#shared-accepted-query-display).

(session-recording-flag)=
Refer to [Session recording flag](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#session-recording-flag).

(voice-sample-conversation-defaults)=
(explicit-voice-interaction-modes)=
Refer to [Explicit voice interaction modes](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#explicit-voice-interaction-modes).

(single-llm-entry-in-model-profiles)=
Refer to [Single LLM entry in model profiles](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#single-llm-entry-in-model-profiles).

(selected-model-server-cleanup)=
(local-model-server-cli)=
(sample-local-model-server-launchers)=
Refer to [Sample-local model-server launchers](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#sample-local-model-server-launchers).

(standalone-model-server-defaults)=
Refer to [Standalone model-server defaults](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#standalone-model-server-defaults).

(batch-stt-migration)=
(shared-model-stack-speaker-filtering)=
Refer to [Shared model-stack speaker filtering](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#shared-model-stack-speaker-filtering).

(native-nim-endpoint-profiles)=
Refer to [Native NIM endpoint profiles](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#native-nim-endpoint-profiles).

(livekit-sample-credentials)=
Refer to [LiveKit sample credentials](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#livekit-sample-credentials).

(launcher-failure-exit-statuses)=
Refer to [Launcher failure exit statuses](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#launcher-failure-exit-statuses).

(model-reasoning-control)=
Refer to [Model reasoning control](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#model-reasoning-control).

(llama-nemotron-llm-removed)=
Refer to [Llama Nemotron LLM removed](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#llama-nemotron-llm-removed).

(operator-visible-runtime-changes)=
Refer to [Operator-visible runtime changes](
migration-history/v0.3.0-beta-to-v0.4.0-beta.md#operator-visible-runtime-changes).
