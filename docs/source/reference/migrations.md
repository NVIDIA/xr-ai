<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Release migration

## Changes since v0.3.0-beta

The development version on `main` is unreleased. These instructions cover
changes from `v0.3.0-beta` to that version. If upgrading from an older release,
first apply {doc}`migration-history/through-v0.3.0-beta`, then apply the changes
below. Refer to {doc}`migration-history` for completed migration guides.

Stop running model services with the old checkout's commands and configuration
before updating the checkout. Start the updated services before restarting
workers; launchers can reuse a running service whose identity matches even when
its application code predates the update.

### Persistent services after startup failure

Stacks using `launch_mode="persist"` now retain services that have signaled
readiness when a later service fails or startup is cancelled. Owned processes
and unfinished persistent launches are still stopped. If you relied on startup
failure to clear the entire model stack, run the sample's model command with
`--stop` explicitly. Refer to {doc}`/guides/customizing-model-servers`.

### Nemotron 3 embedding model

The embedding server now defaults to `nvidia/Nemotron-3-Embed-1B-BF16`; the NIM
consumer profile selects `nvidia/nemotron-3-embed-1b`. Both the tea sample and
standalone RAG now default to 2048 dimensions. The local endpoint remains on
port 8109 with the served name `embed`.

1. Stop the agent and its persistent embedding server **before updating**.
   For the tea sample, run from the repository root:

   ```bash
   uv run --project agent-samples/tea-making-sample/model-servers tea_making_sample_model_servers --stop
   ```

   For a standalone deployment, stop the embedding process or container through
   the supervisor that started it. Check that its listener on port 8109 has
   stopped. Stopping only RAG or the embedding launcher does not stop persistent
   weights; the pip launcher reuses a healthy listener without comparing models.
2. Update the checkout and change `embedding_dim: 768` to `embedding_dim: 2048`
   in copied tea **and standalone** `rag_service.yaml` files. Restart the
   embedding server before RAG. Verify the `embed` entry's `root` in
   `/v1/models` identifies Nemotron 3 Embed, not the previous model.
3. Restart the agent. Changing the default dimension from 768 to 2048 changes
   the existing cache digest, so RAG rebuilds the index automatically. If a
   copied configuration already used 2048 dimensions, set a new `cache_key` or
   delete its generated cache before
   restarting RAG; a local served alias such as `embed` does not identify weights.
   If RAG started while the previous pip server was still listening, stop RAG,
   verify the new model is serving, and delete the generated cache
   (`models/tea-making-rag-cache` for tea, or the configured `.rag-cache` for
   standalone RAG) or set a new `cache_key` before restarting RAG. Otherwise,
   old-model vectors saved at 2048 dimensions remain reusable after the server
   switches models. The Docker launcher recreates its container on model changes.

Old 768-dimensional `.npz` caches are no longer used by the default
configuration and can be deleted to reclaim space without losing documents.

Independently deployed NIM endpoints must serve the new model before using the
updated NIM profile. For later model changes behind an unchanged profile, set
an explicit model revision in `cache_key` and restart RAG. Do not change models
while RAG is running; its document vectors must match query embeddings.

### Shared accepted-query display

Every `VoiceAgent` now returns accepted query text to its originating participant
on the new `voice.user-query` client data topic. This generic display supports
checking what the voice layer understood without a sample-specific echo agent.
Payloads are UTF-8 `Speaker N: <text>` for speaker ASR or `User: <text>` for batch
STT and typed input. Text is gate-normalized accepted input, not all raw STT;
control phrases do not become queries. The separate internal `voice.transcript`
event retains its existing mode/backend publication policy. Optional ignored-speaker display remains
on `voice.speaker-diagnostic`, controlled by `speaker.diagnostics`.

Internal application query topics, including `tea-making.user-query`,
`lab-instrument-monitoring.user-query`, `xr-render.user-query`, `task.capture.input`
and `simple-vlm.user-query`, are unchanged and are not used for client display.
Clients with topic-specific handlers can handle or ignore `voice.user-query`;
catch-all message lists will show the additional messages, including typed echoes.
There is no new constructor switch. Routing remains participant-targeted;
topic filtering is not an authorization boundary.

### Session recording flag

Replace `--capture` with `--session-recording` in saved launch commands for
`simple_vlm_example`, `xr_render_demo`, `tea_making_sample`, and
`lab_instrument_monitoring`. The old flag is no longer accepted. For example,
`uv run simple_vlm_example --capture` becomes
`uv run simple_vlm_example --session-recording`.

### Explicit voice interaction modes

Set an explicit `mode: conversation`, `hybrid`, `wake_only`, or `always_on` at
the top of the selected voice-gate YAML. For the default speaker ASR stack,
add this mapping to private voice-gate files retained from `v0.3.0-beta`:

```yaml
speaker:
  enabled: true
  backend: required
  base_url: http://127.0.0.1:8102
```

A retained YAML without a `speaker` mapping selects batch STT. The new sample
profiles omit the `stt` role, so this configuration raises `KeyError` during
worker construction. With the mapping above, workers wait for `speaker-stt`
before reporting ready. Refer to
{ref}`the batch alternative <batch-stt-migration>` if retaining batch speech.

In tea's worker YAML, replace `voice_gate_yaml: voice_gate.always-on.yaml` with
`voice_gate_yaml: voice_gate.yaml`, then set `mode: always_on` in that file to
retain ungated speech. The alternate file is removed; keeping its path loads
the package's batch-STT defaults.

(voice-sample-conversation-defaults)=
The four voice samples select `conversation`. Say
`Hey agent, let's start talking` after connecting and
`Hey agent, let's stop talking` to close the conversation.
Select `wake_only` for addressed turns, `hybrid` for addressed turns alongside
conversations, or `always_on` for ungated dispatch. `wake_only` and `hybrid`
require non-empty `magic_phrases`. Set `conversation.require_wake_phrase: true`
to require wake phrases during an active conversation.

In conversation mode, only accepted final transcripts during an active
conversation reach `VOICE_TRANSCRIPT_TOPIC`; hybrid mode also delivers accepted
closed-state one-off questions. Applications needing ungated final transcripts
must select `always_on`. Restart workers after configuration edits. Refer to
{ref}`voice-conversation-controls` for interaction behavior.

### Single LLM entry in model profiles

XR Render and its eval commands now use `llm` instead of `agent_llm`.
For custom XR Render, NIM consumer, or shared model profiles, move any settings
from `agent_llm` you need into `llm`, then remove the old entry. The NIM smoke
check rejects the old entry.

(selected-model-server-cleanup)=
(local-model-server-cli)=
### Sample-local model-server launchers

Replace commands targeting `model-server-samples/model-servers` with each agent
sample's `model-servers/` project. Before updating, run the applicable stop
command from the old checkout's repository root:

```bash
uv run --project model-server-samples/model-servers model_servers --stop
uv run --project model-server-samples/model-servers-nim model_servers_nim --stop
```

The old commands discover their cleanup ports without `--models`. The shared
launcher rejects combining `--models` with `--stop`; the NIM launcher ignores
`--models` during shutdown. Stop former batch STT and NIM services before
updating; the new launcher's stop plan omits those services.

After updating, run from `agent-samples/simple-vlm-example/`:

```bash
uv run --project model-servers simple_vlm_example_model_servers
uv run --project model-servers simple_vlm_example_model_servers --stop
```

Other samples have corresponding `<sample>_model_servers` commands. Keep
`--gpu-profile NAME` for hardware selection. Pass the same profile to `--stop`
when it overrides service ports. Remove `--models` and `--allow-anonymous`;
local stacks require `HF_TOKEN`. Move custom server settings into shared
defaults under `utils/xr-ai-model-server-stack/yaml/` or sample GPU overrides
under `agent-samples/<sample>/model-servers/yaml/`.

The new stop plan covers only its selected services. Stop that plan before
changing its configuration or switching samples. Refer to
{doc}`/guides/customizing-model-servers` for the current workflow.

### Shared model-stack speaker filtering

(batch-stt-migration)=
The four voice samples use speaker ASR and omit batch STT on port 8103.
Replace unconditional worker `make_stt` calls with `VoiceAgent.from_models`,
passing the loaded models and voice-gate settings. To retain batch STT, select
`stt` in the sample's launcher and hardware overrides and configure the worker's
batch adapter as `models.stt` in the worker profile. For batch-only speech,
omit `speaker-stt` from the selection and set `speaker.enabled: false`. Refer to
{doc}`/guides/customizing-model-servers` for serving both paths.

### Native NIM endpoint profiles

Replace the managed `model_servers_nim` stack and
`model_servers --models vlm_llm_nim` with independently deployed native NIM
endpoints. Configure native model IDs with `openai_compat`, `riva_grpc` for
speech, and embedding `default_extras.input_type: passage` for prefix-aware
asymmetric inputs. Refer to {doc}`/guides/deploying-with-nim` for the removed
compatibility surfaces, endpoint configuration, and validation.

Native Magpie speech omits the compatibility wrapper's 300 ms pause. Validate
inference separately from Riva readiness, which checks channel connectivity.

### LiveKit sample credentials

Shipped DeviceIOHub YAML files now leave `api_key` and `api_secret` blank.
The YAML loader generates a fresh random pair when both values are missing or
blank, so samples require no credential setup. Existing private configurations
and environment overrides remain supported; partial pairs still fail startup.
Refer to {doc}`/getting_started/credentials` for fixed-pair configuration.

### Launcher failure exit statuses

Update shell scripts and supervisors that interpret sample launcher exit
statuses. An unexpected monitored process exit now fails the stack; it
previously returned success. A process that exits during startup now reports
its failure status; startup failures previously returned 130.

The launcher preserves positive child exit codes, maps signal termination to
`128 + signal number`, and maps an unexpected zero exit to 1. User cancellation
during startup returns 130. Refer to
{doc}`../components/launcher-and-process-model` for failure diagnostics and
shutdown behavior.

### Model reasoning control

Model profiles now declare reasoning with a typed `adapter.thinking` setting.
A per-call `enable_thinking=True` or `False` always applies to that call. A
legacy `default_extras.chat_template_kwargs.enable_thinking: true` default could
previously not be turned off per call. Refer to {ref}`agent-sdk-models-thinking`.

- **`llama_nemotron` profiles:** requests now start the system message with
  `detailed thinking off` unless a call passes `enable_thinking=True`; the model
  previously reasoned by default and its preamble appeared in the response. To
  keep reasoning on by default, set it in the profile:

  ```yaml
  adapter:
    preset: llama_nemotron
    thinking:
      default: true
  ```

- **Custom `LLMService` implementations:** `enable_thinking` now defaults to
  `None` instead of `False`. Treat `None` as the endpoint's configured default.
- **Profiles with the legacy key:** no change is required. With a template-driven
  or presetless adapter, the value becomes `thinking.default`, so a call can
  override it. It must now be a boolean, and a profile that also sets a
  different `thinking.default` is rejected. Prefer `thinking.default` in new
  profiles.

### Operator-visible runtime changes

Recorded-video perception is enabled when `video_history_enabled` is omitted
from the XR Render worker YAML. Set it to `false` to opt out, or configure
{ref}`xr-render-recording-prerequisites` before using recorded-video tools.

## Earlier releases

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
