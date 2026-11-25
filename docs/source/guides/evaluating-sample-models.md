<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Measure sample speech performance

Use `tests/run_sample_eval.py` to compare model services with the same five-minute
Conversation, Procedure Capture, Procedure Guide, Tea, Lab Instrument Monitoring
or XR Render Demo replay. It exercises
native workers through test hub IPC and records speech delivery, response timing
and STT backlog. Model, speech, Clef and Tea RAG services remain caller-owned.

Functional quality stays with the existing sample evaluations. See
{doc}`../reference/conversation-example` for routing and alert evaluations,
{doc}`../reference/procedure-capture-guide` for caption and guide scenarios, and
{doc}`../reference/tea-making-sample` for Tea routing evaluations. This runner
neither scores answers nor inspects internal workflow state.

## Prepare and run

From the repository root, prepare the tests environment:

```bash
uv --config-file uv.toml sync --project tests
```

Start the selected sample's services using its existing model-server workflow;
see {doc}`customizing-model-servers` and the sample references above. Supply the
resulting model profile. Conversation also needs Clef and Speaker STT; Procedure
uses batch STT; Tea needs Speaker STT and RAG. `--voice-gate` can override speech
endpoints and `--rag-endpoint` can override Tea's RAG endpoint.
Use `--worker-config` to supply native worker YAML; its relative paths retain
the sample loader's semantics. This replaces the complete native worker YAML, so
copy the sample config and preserve its other settings when changing endpoints.
Worker settings are loaded during preflight and
reused for the run. Bad native configuration fails before output creation.
Lab uses the services described in {doc}`../reference/lab-instrument-monitoring`.
XR Render additionally needs dedicated scene and tracking services; see
{doc}`../reference/xr-render-demo`. For XR, `--worker-config` is required and must
explicitly set `scene_endpoint` and `openxr_endpoint` for dedicated test services.
The default scene/tracking ports (8320/8330 respectively) are rejected on every host.
The runner queries `get_scene_state` before creating its output directory and
refuses a scene whose `objects` list is not empty. Start a fresh empty scene for
each configuration; the replay sends real scene mutations to the supplied service.
The XR replay changes objects through scene RPC and uses explicit coordinates;
it does not exercise headset-relative placement or rendering performance.
Lab event output and XR text memory are scoped to the run directory.

To start dedicated services, work from `agent-samples/xr-render-demo/`. Copy
`scene/scene_service.yaml` to `scene/scene_service.eval.yaml` and
`yaml/openxr_service.yaml` to `yaml/openxr_service.eval.yaml`, preserving the other
settings and relative paths. In the scene copy set `endpoint: tcp://127.0.0.1:18320`
and a separate `scene_socket: ipc:///tmp/xr_eval_scene`; in the OpenXR copy set
`endpoint: tcp://127.0.0.1:18330`. Start each in its own terminal:

```bash
uv run --project scene xr_render_scene --config scene/scene_service.eval.yaml
uv run --project ../../services/openxr-service openxr_service --config yaml/openxr_service.eval.yaml
```

Copy the full worker YAML alongside its original and set
`scene_endpoint: tcp://127.0.0.1:18320` and `openxr_endpoint: tcp://127.0.0.1:18330`.
Pass that copy to `--worker-config`. Restart the dedicated scene service between
configurations to provide an empty scene. The sample reference describes the LOVR
and OpenXR prerequisites. The shared service package also has its own default
`services/openxr-service/openxr_service.yaml`; `--config` selects the sample copy
shown here.


Prepare speech fixtures once, outside the timed replay:

```bash
uv --config-file uv.toml run --project tests python tests/run_sample_eval.py prepare \
  --sample conversation --models-config /path/to/models.json \
  --audio-dir /tmp/xr-ai-speech/fixtures
```

Repeat preparation for each sample. WAV bytes are hash-pinned; missing or changed
pinned files are errors. Reuse the same fixture directory for all model and
hardware comparisons. Input is resampled to mono 16 kHz float32. Replay files live beside each sample's existing evals in
`eval/speech/*.yaml` or `evaluation/speech/*.yaml`; generated scenes reuse the
Procedure renderer. Tea requires a legible branded Chamomile
package image through `--image tea_package=/path/to/package.jpg`. Supplied images
are converted to RGB and bounded to a 1280-pixel maximum dimension.

The Lab replay requires `--image lab_before=/path/to/before.jpg` and
`--image lab_after=/path/to/after.jpg`. Both images must show the same instruments
with readable displays and markers listed in the sample's device map; the second
image must show changed readings. The sample's `eval/speech/README.md` shows how to generate raw camera fixtures
with its existing visual-eval renderer; captured instrument photos also work. Keep the same two images
across configurations. The replay exercises direct readings and background
instrument monitoring; its functional eval remains responsible for accuracy.

Run each configuration separately, with a new output directory:

```bash
uv --config-file uv.toml run --project tests python tests/run_sample_eval.py run \
  --sample conversation --models-config /path/to/models.json \
  --audio-dir /tmp/xr-ai-speech/fixtures --label baseline \
  --output-dir /tmp/xr-ai-speech/baseline
```

Repeat with the candidate profile and label. Endpoint aliases do not establish
actual weights: record model revisions, quantization, serving version, hardware
and concurrency settings with `--metadata /path/to/details.json`. These values
are operator-supplied. The profile selects endpoints; server launch settings
select weights, context and request limits. The runner does not restart services.

## Hosted model endpoints

The same `--models-config` accepts hosted HTTPS endpoints. Copy the sample's
model profile and replace the desired roles; preserve its other entries. For
example, use NVIDIA-hosted Nemotron Ultra for the LLM and Super VL for vision:

```yaml
llm:
  category: llm
  adapter:
    kind: openai_compat
    model_name: nvidia/nemotron-3-ultra-550b-a55b
    reasoning_field: reasoning_content
    thinking: {control: chat_template, default: false}
    capabilities: {streaming: true, tool_calls: true, reasoning: true}
  endpoint:
    base_url: https://integrate.api.nvidia.com
    api_key_env: NGC_API_KEY
    timeout: 180
    readiness: none
vlm:
  category: vlm
  adapter:
    kind: openai_compat
    model_name: nvidia/nemotron-3.5-super-vl-120b-a12b
    reasoning_field: reasoning_content
    thinking: {control: chat_template, default: false}
    capabilities: {streaming: true, tool_calls: true, vision: true, reasoning: true}
  endpoint:
    base_url: https://integrate.api.nvidia.com
    api_key_env: NGC_API_KEY
    timeout: 180
    readiness: none
```

These are entries under the existing `models` mapping. The SDK appends
`/v1/chat/completions`, so omit `/v1` from `base_url`. `readiness: none` skips
local-style health URLs that the hosted API does not expose; inference requests
still report authentication, connection and provider errors.

Set the environment variable named by `api_key_env`, or save it using the
credential helper in {doc}`../getting_started/credentials`. Both `prepare` and
`run` load the existing credential cache. The key stays outside the profile.

Run the same prepare/run commands with the copied profile. LLM and VLM need no
local model servers or GPU allocation when both roles are hosted. STT, TTS and
sample capability services retain their own configuration and still need to be
available. Hosted vision calls send selected camera frames to the provider;
remote model latency includes network and provider scheduling time.

## Measurements

The sender and test hub run in a process separate from the native worker. Audio
is sent in 20 ms chunks and camera frames at 1 Hz, beginning after worker readiness
and participant setup. No client audio is excluded as warm-up. The checked-in replays last
300 seconds with scripted speech and silence, followed by a ten-second response
drain. Host scheduling can still delay input; every chunk is preserved. A run
is timing-comparable only when completed with maximum sender lateness at most
0.1 seconds and no observed model/runtime errors during the measurement.
This checks scheduling and execution errors; answer quality remains a separate evaluation.

Each run writes a manifest, the fixed scenario, raw source/worker event traces,
their merge in `events.jsonl`, and one `report.json`. The manifest records source,
harness, worker-config, loaded worker-settings, guide, profile, effective voice-gate and fixture hashes
plus sanitized endpoint aliases. The scenario source path is repository-relative.
Workers use private snapshots of the validated model and voice profiles. Fixture
and output directories are gitignored; native samples may save media there.
The observer records no raw audio payloads or model HTTP bodies.

The JSON includes counts, median/worst timing, handler coverage, unattributed
requests, sender lateness and Speaker-STT queue/drop/reset observations. Missing
measurements are `unmeasured` or `not applicable`. Counts matter: an untranscribed
input contributes no response latency and must be noticed in input coverage.
Backlog drop/reset counts do not measure transcription accuracy.

`error_events` counts ERROR/CRITICAL log events, WARNING logs carrying an exception,
and construction of typed `ToolLoopError` failures, including quiet tool-loop
fallbacks. Recovered retries and background failures also count, such as Tea's
workflow observer and XR's verification retry; they make timing non-comparable
even when the user receives a good reply. Only events between measurement start
and the marker before worker teardown count, including the response drain. One
failed operation can log more than one event. The observer stores the logger,
function and exception type, without log messages or exception bodies. It does
not classify answer wording. Semantic mistakes and fallbacks outside these signals
are not counted. In particular, XR's no-mutation-evidence fallback can replace a
reply with "I couldn't make that change" and emit only a plain WARNING; that case
is not counted. The terminal summary prints handler coverage and error counts
alongside execution status and timing comparability.

Runtime request IDs are matched to the unique source-audio PTS window containing
each query timestamp, with a 160 ms lead allowance for Speaker-STT encoder hops.
Missing or ambiguous timestamps remain unattributed. Speech delivery measures
the last matching handler query; completed reply text measures all matching
foreground reply streams. Both use the speech-end marker, recorded before the
last 20 ms chunk is sent. Text completion is not completed spoken playback.
First observed audio can be an acknowledgment, background notice or delayed
prior output; it is not request-matched model time-to-first-token.

Compare source `revision` and `tracked_diff_sha256`, plus scenario, worker-config,
loaded worker-settings, guide, profile, voice-gate, fixture and harness hashes before comparing runs.
Loaded dataclass settings include nested values and contents of exposed file paths
(such as Tea's workflow). Opaque native objects without public serialization,
including Lab's DeviceMap, are listed in `unserialized_worker_settings`; their
internal state is not serialized. Lab's public `device_map_yaml` file is separately
hashed as `device_map_sha256`. Repeat trials to measure variation. This scripted native-IPC benchmark does not establish
dense-speech capacity, browser/LiveKit behavior, GPU memory use or general model
quality. The observer restores SDK hooks on exit; only one active replay per process is allowed and concurrent calls fail immediately.

## Add or select a replay

The runner discovers these sample-owned YAML files at startup. Their filenames
supply the `--sample` choices; `run --help` lists the available replays. Each
file contains `duration_s`, `worker` and `events`. For example, from a sample's
`eval/speech/` directory:

```yaml
duration_s: 300
worker:
  package: conversation_example_worker
  config: ../../yaml/conversation_example_worker.yaml
events:
- at_s: 0
  say: Start talking to me
  handler: false
- at_s: 25
  say: What do you see?
  scene: {rectangle: blue}
```

Worker config paths resolve relative to the replay file. The optional
`worker.launch` mapping supplies the native Procedure launch request's `mode`
and `guide_path`; guide paths also resolve beside the replay file. Duplicate
filenames across samples are rejected. Add a replay file to extend the catalog;
the runner has no central sample list. The worker must implement the native
`config.load_config()` and `app.run_app()` entry points.
