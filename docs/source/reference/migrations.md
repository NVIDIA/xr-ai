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

### vLLM 0.31 security update

The four vLLM service environments pin `vllm==0.31.0`. Their standalone YAMLs
and the shared Docker default now use `nvcr.io/nvidia/vllm:26.10.pre1-py3` in
place of `nvcr.io/nvidia/vllm:26.09-py3`. The NGC preview supports Linux
x86_64 and aarch64 with CUDA 13.4 and NVIDIA vLLM `0.31.0+88641915`. Qualify
this preview on the target GPU and model workload before deployment.
Refresh pip environments with
`uv --config-file uv.toml sync --project <service-directory>`.

Per-request `mm_processor_kwargs` and `media_io_kwargs` overrides are rejected
by default. Move trusted processor settings into server configuration; do not
enable `--trust-request-mm-kwargs` for untrusted clients. Custom image overrides
must include the fixes for CVE-2026-114050 and CVE-2026-114055 independently of
the host Python environment. Refer to {doc}`/components/ai-services` for runtime
selection and container replacement.

### Procedure scenario observation checks use decisions

In procedure-guide scenario manifests, replace `observe.answers` with boolean
`observe.accept` (for example, `action: observe, accept: false`), or select a
shared visual `case`. High-confidence image decisions do not generate captions;
checks now assert the final decision independently of the inference path.
Unavailable observations fail both positive and negative expectations.

### Procedure guides separate visual evidence from decisions

Guide documents now require `schema_version: 2`; version 1 is rejected with a
migration diagnostic. Capture packets remain at schema version 1.

For each `current_view` step, replace `evidence.pattern` with an explicit prose
`evidence.condition` and remove its observation `agent` block. Rewrite the
trigger question to request descriptive visual facts, including the relevant
region, count, orientation, contact, or labeled measurement. CLEF first judges
the image and uses its selected answer only above 80% confidence. Otherwise
Cosmos supplies those facts and CLEF judges that description. Retain the
step's `consecutive`, typed `commit`, and `complete_when` requirements. Approved
visual steps require a commit; unresolved draft steps must remain drafts.
Timer steps keep their elapsed-time evidence patterns and runtime-owned clocks.
After migrating the steps, set the guide's `schema_version` to `2`.

Restart the sample's model-server launcher to include CLEF on port 8120 and use
the updated `yaml/models.json` decision entry. The CLEF endpoint must support
image input; text-only endpoints cannot serve the image-first policy.
Guide mode waits for CLEF;
capture mode is unchanged. Re-evaluate existing guides after this migration,
since regex matching of a visual answer no longer determines completion.

### Interaction messages move to xr-ai-interaction

Code that imports `UserQuery`, `VoiceOutput`, `VoiceParticipantJoined`,
`VoiceParticipantLeft`, or `VoiceInterrupted` from `xr_ai_voice` must import
them from the new `xr-ai-interaction` package under their new names, and add
`xr-ai-interaction` to its dependencies:

| Before (`xr_ai_voice`) | After (`xr_ai_interaction`) |
|---|---|
| `UserQuery` | `UserQuery` |
| `VoiceOutput` | `ResponseOutput` |
| `VoiceParticipantJoined` | `ParticipantJoined` |
| `VoiceParticipantLeft` | `ParticipantLeft` |
| `VoiceInterrupted` | `Interrupted` |

Apart from the `ResponseOutput` fields described under *Voice output declares
reply or update*, fields, validation rules, and the `voice.output` and
`voice.transcript` topics are unchanged; `VOICE_OUTPUT_TOPIC`, `VoiceTranscript`, and `VoiceAgent` stay in
`xr_ai_voice`.

Two validation error messages now say `response output` instead of `voice
output`: `non-final response output needs a response_id` and `complete response
output must contain text`. Update tests or callers that match the previous
wording.

```python
# Before
from xr_ai_voice import VOICE_OUTPUT_TOPIC, UserQuery, VoiceOutput

# After
from xr_ai_interaction import ResponseOutput, UserQuery
from xr_ai_voice import VOICE_OUTPUT_TOPIC
```

Replace `xr-ai-routing[voice]` with `xr-ai-routing[runtime]`. Remove the
`[voice]` extra from `xr-ai-conversation`, which now depends on
`xr-ai-interaction` directly.

### Coding-agent setup skill rename

The development preparation skill is now named `setup`. Replace installed
`getting-started` copies with `setup` and update saved setup prompts to use
`skills/setup/SKILL.md` at the selected repository ref. Use
`skills/run-demo/SKILL.md` to launch the browser demo. Historical release tags
retain their original files. Refer to {doc}`/getting_started/skills` for
installation and release selection.

### Dependency refresh and native tool callback exceptions

The SDK now requires the NeMo Relay 0.10 release series. Update copied worker
dependency constraints from `nemo-relay>=0.7.2,<0.8` to `nemo-relay>=0.10.0,<0.11`.
The dependency qualification date is now `2026-10-22T21:39:43Z`, including
Relay 0.10.4. Refresh existing environments from the repository root with
`uv --config-file uv.toml sync --project <project-directory>`.

Native tool and model callbacks now propagate their original Python exceptions.
Catch an expected exception such as `FrameUnavailable` directly instead of
catching `RuntimeError` and parsing its message. Unexpected exceptions and
cancellation continue to propagate. Tool requests and returned SDK result types
are unchanged; Relay's tool result envelope stays inside the adapter.

Applications that supply a Relay observability configuration must migrate it
to `version = 4`. Custom Relay tool execution intercepts now receive
`(context, next_call)`; read the tool name and arguments from `context`.
LLM execution intercepts must accept the new codec context parameter. Refer to the
[Relay 0.10.0 migration notes](https://github.com/NVIDIA/NeMo-Relay/releases/tag/0.10.0)
for the configuration and intercept changes.

### Voice output declares reply or update

`ResponseOutput` replaces `interrupt` with `kind`, `request_id`, and `urgent`.
`kind` is required: a `reply` answers `request_id`, and an `update` is unprompted
and may set `urgent=True` to cut current speech. Updates are finite and have no
`response_id`. `VoiceAgent` now decides when each message is spoken, so
`VoiceAggregationAgent` and `VOICE_CONTRIBUTION_TOPIC` are removed. Publish
updates straight to `VOICE_OUTPUT_TOPIC`, and pass
`combiner=llm_update_combiner(llm)` to `VoiceAgent` to have pending updates
rewritten into one utterance. `ResponseOutputs`, exported from
`xr_ai_interaction`, builds these messages and frames streamed replies. A
producer whose queued updates became stale can drop them with
`outputs.withdraw(participant_id)`, which publishes `kind="withdraw"`.

```python
# Before
from xr_ai_voice import VOICE_CONTRIBUTION_TOPIC, VoiceAggregationAgent, VoiceOutput

runtime.register("voice-aggregation", VoiceAggregationAgent(llm=llm))
await ctx.publish(VOICE_CONTRIBUTION_TOPIC, VoiceOutput(text="Timer done.", interrupt=True))

# After
from xr_ai_interaction import ResponseOutputs
from xr_ai_voice import VOICE_OUTPUT_TOPIC, VoiceAgent, llm_update_combiner

voice = VoiceAgent(..., combiner=llm_update_combiner(llm))
outputs = ResponseOutputs(ctx, VOICE_OUTPUT_TOPIC)
await outputs.update(participant_id, "Timer done.", urgent=True)
async with outputs.reply(participant_id, request_id=query_id) as reply:
    await reply.send("Here is the answer.")
```

The aggregation tuning parameters have no replacement: coalescing follows real
playback state, and `llm_update_combiner` takes only `prompt`, `max_tokens`,
and `timeout_s`.

Scheduling changes too. A reply for a different request than the one being
spoken now cuts that reply and flushes queued audio; a playing update is never
cut by a reply and finishes first. Non-urgent updates wait while the
participant is speaking, and speech that is cut is never replayed. A streamed reply that publishes no
fragment for 15 seconds is closed and its later fragments are dropped; set
`stream_idle_timeout_s` on `VoiceAgent` for producers that pause longer.

### Persistent services after startup failure

Stacks using `launch_mode="persist"` now retain services that had signaled
readiness when cleanup began after a startup failure or cancellation. Owned
processes and unfinished persistent wrappers are still stopped. Persistent
pip-backend vLLM can keep loading in its separate session after its wrapper
stops. If you relied on startup failure to clear the entire model stack, run
the sample's model command with
`--stop` explicitly. Refer to {doc}`/guides/customizing-model-servers`.

### Nemotron 3 embedding model

The embedding server now defaults to `nvidia/Nemotron-3-Embed-1B-BF16`; the NIM
consumer profile selects `nvidia/nemotron-3-embed-1b`. Both the tea sample and
standalone RAG now default to 2048 dimensions. The local endpoint remains on
port 8109 with the served name `embed`.

1. Stop the agent and its persistent embedding server **before updating**.
   For the `v0.3.0-beta` shared model stack, run from the old checkout's
   repository root:

   ```bash
   uv run --project model-server-samples/model-servers model_servers --stop
   ```

   For a standalone deployment, stop the embedding process or container through
   the supervisor that started it. Check that its listener on port 8109 has
   stopped. Stopping only RAG or the embedding launcher does not stop persistent
   weights; the pip launcher reuses a healthy listener without comparing models.
2. Update the checkout and change `embedding_dim: 768` to
   `embedding_dim: 2048` in copied tea **and standalone** `rag_service.yaml`
   files. While RAG is stopped, you can empty its generated cache directory
   (`models/tea-making-rag-cache` for tea, or the configured `.rag-cache` for
   standalone RAG) to reclaim space. Restart the embedding server before RAG.
   Verify the `embed` entry's `root` in `/v1/models` identifies Nemotron 3
   Embed, not the previous model. If it still reports the previous model, stop
   the old listener, restart the updated embedding server, and repeat this
   check before starting RAG.
3. Restart the agent. RAG's new cache format rebuilds the `v0.3.0-beta` index
   automatically, including caches that already used 2048 dimensions. Refer to
   {ref}`embedding-service-identity-migration` for the endpoint requirements.

Independently deployed NIM endpoints must serve the new model before using the
updated NIM profile. For later model changes behind an unchanged profile, set
an explicit model revision in `cache_key` and restart RAG. Do not change models
while RAG is running; its document vectors must match query embeddings.

(embedding-service-identity-migration)=
### Embedding SDK identity and RAG cache contract

`EmbeddingService` now requires `async model_identity() -> str | None`.
Custom implementations must add this method to satisfy the runtime-checkable
protocol and support RAG startup. Return a stable backing-model identifier when
known, or `None` only when a successful lookup finds no identity metadata. Do
not use an alias that can silently switch weights. Existing implementations
without this method no longer satisfy the protocol.

The built-in OpenAI-compatible client reads the matching model's `root` from
`/v1/models`. HTTP failures, timeouts, invalid JSON, and malformed or ambiguous
metadata raise `ModelServiceError`, exported by `xr_ai_models`, with the original
transport or JSON error retained as the cause. Custom adapters must likewise
raise this error for lookup failures instead of returning `None`.

RAG permits retrieval without persistent cache reads or writes when a successful
lookup returns `None`. It propagates lookup failures and fails startup: fix the
endpoint, credentials, or metadata response before restarting. Cancellation
continues to propagate. `cache_key` does not override missing identity or a
failed lookup.

The cache format now combines the reported identity, a hash of the resolved
embedding role, manual `cache_key`, indexing settings, and document content.
Existing caches are rebuilt once and can be deleted afterward. `cache_key`
now supplements the role hash instead of replacing it. Unrelated LLM, VLM,
or TTS edits and profile formatting changes do not rebuild the index.

The reported identity is not a weight checksum. This repository's Docker
launcher reports a resolved snapshot path; pip reports the Hugging Face
repository ID. Switching backends can rebuild the cache, while a Hub revision
change under the same pip model ID requires a new `cache_key`.

The post-build check raises `ModelServiceError` if the reported identity changes,
including appearing or disappearing, before saving or serving the vectors.
A failed lookup also aborts startup. Metadata and embeddings are separate
requests: matching before/after identities cannot detect a swap-and-restore
or mixed replicas behind one endpoint. Every replica must remain on the same
weights for the entire indexing run and serving lifetime. Health probes check
availability, not model consistency. Stop RAG before changing models, bump
`cache_key` if weights change under the same reported identity, and restart
against the stable deployment.

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
local stacks require `HF_TOKEN`. Move custom server settings into the
service's own YAML under `services/` or sample GPU overrides
under `agent-samples/<sample>/model-servers/yaml/`.

The new stop plan covers only its selected services. Stop that plan before
changing its configuration or switching samples. Refer to
{doc}`/guides/customizing-model-servers` for the current workflow.

### Standalone model-server defaults

Sample model stacks now start from each service's own YAML under `services/`,
so standalone launches use the values that stacks used. If you run these servers
directly and depend on the earlier defaults, set them in your YAML:

| Server | Now | Earlier |
|---|---|---|
| VLM and embedding | `enforce_eager: true` | `false` |
| Omni | `max_model_len: 32768`, `video_num_frames: 64` | `131072`, `256` |
| Pocket TTS | `device: cuda` | `auto` |

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

- **`thinking_budget` call argument:** now sent as vLLM's
  `thinking_token_budget` instead of `chat_template_kwargs.thinking_budget`,
  which the Nemotron chat templates ignored. The argument is omitted when
  reasoning is off. Enforcement requires server support and has been verified
  on Nemotron Omni with vLLM's `--reasoning-parser nemotron_v3`; NIM budget
  support has not been qualified. Other endpoints may reject or ignore it;
  the SDK does not detect support or retry without the budget. Refer to
  {ref}`agent-sdk-models-thinking`. Raw
  `default_extras.chat_template_kwargs.thinking_budget` profile values are
  forwarded unchanged, not migrated to the new field.
- **Custom `LLMService` implementations:** `enable_thinking` now defaults to
  `None` instead of `False`. Treat `None` as the endpoint's configured default.
- **Profiles with the legacy key:** no change is required. With a template-driven
  or presetless adapter, the value becomes `thinking.default`, so a call can
  override it. It must now be a boolean, and a profile that also sets a
  different `thinking.default` is rejected. Prefer `thinking.default` in new
  profiles.

### Llama Nemotron LLM removed

The `services/llama-nemotron-llm` server (`llama_nemotron_llm_server`, port
8106) and the `llama_nemotron` client preset are removed; no sample or
deployment profile used them. Stop a running server with the old checkout
before updating. Its Docker container is `xr-ai-vllm-llama-nemotron-llm-server`.
Point custom profiles that set `preset: llama_nemotron` or
`kind: preset:llama_nemotron` at another LLM, such as `nemotron3_nano`.

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

(routed-agents-publish-their-own-output)=
For routing and conversation output contracts, refer to
[the routing contract](agent-sdk-routing.md#response-progress-and-routing-context)
and {doc}`agent-sdk-conversation`.
