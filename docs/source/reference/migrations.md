<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Release migration

## LiveKit sample credentials

Shipped DeviceIOHub YAML files now leave `api_key` and `api_secret` blank.
The YAML loader generates a fresh random pair when both values are missing or
blank, so samples require no credential setup. Existing private configurations
and environment overrides remain supported; partial pairs still fail startup.
Refer to {doc}`/getting_started/credentials` for fixed-pair configuration.

## Swift microphone cleanup errors

`StreamSession.disconnect()` and `StreamingBackend.disconnect()` now throw
cleanup errors after closing the transport. Update callers to use `try await`
and handle failures: capture may still be running even though the connection
has closed. Retain the same session and retry `disconnect()` or `stopAudio()`
until cleanup succeeds before discarding it. Connecting that same LiveKit
backend also retries its pending cleanup before opening another connection.
A custom backend must close its transport even when cleanup fails. Existing nonthrowing backend implementations can still
satisfy the throwing protocol requirement.

## Launcher failure exit statuses

Update shell scripts and supervisors that interpret sample launcher exit
statuses. An unexpected monitored process exit now fails the stack; it
previously returned success. A process that exits during startup now reports
its failure status; startup failures previously returned 130.

The launcher preserves positive child exit codes, maps signal termination to
`128 + signal number`, and maps an unexpected zero exit to 1. User cancellation
during startup returns 130. Refer to
{doc}`../components/launcher-and-process-model` for failure diagnostics and
shutdown behavior.

## Voice sample conversation defaults

The shipped voice samples now require `Hey agent, let's start talking` after
connecting, then accept speech without a wake phrase until
`Hey agent, let's stop talking`. Exact phrase fragments can span utterances
within the configured window. The existing STT adapters and NIM model stack
remain unchanged.

Set `conversation.require_wake_phrase: true` in the sample's voice-gate YAML to
keep wake phrases during an active conversation. Set
`conversation.enabled: false` to restore the original wake-only behavior, or
always-on dispatch when `magic_phrases` is empty. Restart workers after editing
these settings. Refer to {ref}`voice-conversation-controls`.

Only final transcripts during an active conversation reach
`VOICE_TRANSCRIPT_TOPIC`; start and stop controls are withheld. Applications
that need all final transcripts retain that behavior by disabling conversation
controls.

## Local model-server CLI

The `model-servers` sample launches only local server wrappers. Its default
Parakeet, Pocket TTS, Nemotron Omni, Cosmos, and embedding stack is unchanged.
Named `--models` selections are retired:

- Replace `model_servers --models default` with `model_servers`.
- Replace `model_servers --models vlm_llm_nim` with the separate
  {doc}`NIM sample </reference/model-servers-nim>`. That stack uses Magpie TTS
  and the original agent endpoints; it is not the former mixed NIM profile.
- Pass custom deployments by JSON path, such as
  `model_servers --models ./yaml/models.my-stack.json`, instead of a bare name.
  Custom deployments can no longer select `llm-nim`, `vlm-nim`, `stt-nim`, or
  `tts-nim` through the local launcher.

Before updating a checkout that ran the former NIM profile, stop it with the
old checkout's `uv run model_servers --stop`. The updated local launcher no
longer discovers the removed NIM-only ports. Stop the old stack before starting
the separate NIM sample because they share client ports. Caches are not removed.

`--help` now displays usage and exits. Unknown options are rejected instead of
being silently ignored; `--dry-run` remains unsupported by `model-servers`.

## StreamKit image-capture dependencies

Request-driven image capture uses LiveKit byte streams. Builds that previously
used older client SDKs must upgrade Android to `client-sdk-android` 2.28.2,
Apple platforms to `client-sdk-swift` 2.16.0, and native C++ to
`client-sdk-cpp` 1.10.2 or newer. The native SDK root now uses the upstream
`include/` and `lib/` directory layout.

## DeviceIOHub rename

Update `services/xr-media-hub/` to `services/device-io-hub/`, the
`xr-media-hub` distribution to `device-io-hub`, and the `xr_media_hub` import
and command to `device_io_hub`. Rename `xr_media_hub.yaml` to
`device_io_hub.yaml` and `XR_MEDIA_HUB_NO_WEB_CLIENT` to
`DEVICE_IO_HUB_NO_WEB_CLIENT`. The rename has no compatibility aliases.

## Shared model-server sample location

The shared model-server launcher moved from `agent-samples/model-servers/` to
`model-server-samples/model-servers/`. Update scripts and configuration paths
that name the old directory. From the repository root, run:

```bash
uv run --project model-server-samples/model-servers model_servers
```

From an agent sample directory, use
`uv run --project ../../model-server-samples/model-servers model_servers`.
The `model_servers` command, model profiles, service ports, and cache locations
are unchanged. Run `uv sync` from the new sample directory to recreate its local
environment; an environment tied to the old directory does not need to be moved.

(consumer-model-readiness)=
## Consumer model readiness

Workers can report ready without STT, TTS, LLM, or VLM availability. Consumer
workers no longer poll model health at startup. `VoiceAgent` waits only for
explicitly supplied `probes`; it no longer adds STT and TTS probes. Start the
shared model-server stack and wait for its launcher to return first. Server
wrappers retain model startup and reuse checks.

{doc}`Simple VLM </reference/simple-vlm-example>` retains its streaming image
warmup, and {doc}`XR Render </reference/xr-render-demo>` retains its LLM warmup,
including for hosted LLMs. Tea making retains its RAG capability probe and
{ref}`embedding health checks <rag-embedding-health>`.

Endpoint `readiness` and `health_path` settings still control explicit `health()`
calls, but do not enable automatic worker polling. Out-of-tree applications
that require the former behavior can explicitly pass
`probes={"stt": stt.health, "tts": tts.health}` to `VoiceAgent`.

## Operator-visible runtime changes

- When `video_history_enabled` is omitted from the xr-render-demo worker YAML,
  recorded-video perception is now enabled. Set it to `false` to opt out. See
  {ref}`xr-render-recording-prerequisites` for the required recording settings.
- `Subscribe.ALL` is deprecated because it names the pre-file-transfer set of
  data, audio, and video subscriptions rather than every available category.
  Use `Subscribe.REALTIME` for the same behavior. Files remain opt-in through
  `Subscribe.REALTIME | Subscribe.FILE` and require `file_sub_addr`.
- A `VoiceOutput` sent with both `response_id` and `final=True` now closes that
  response key even when the response contains only one message. Later output
  from the same participant and producer with that `response_id` is dropped,
  and the voice runtime warns once while the key remains among its 1,024 most
  recent closures. Use a new identifier for each inbound query, such as
  `ctx.metadata.message_id`, or omit `response_id` for a finite one-message
  response.
- DeviceIOHub now waits for the hub to acknowledge shared-memory attachment
  before connecting the LiveKit room or creating its ready file. Missing
  segments trigger bounded recreation; incompatible layouts and acknowledgement
  timeouts fail startup. Check the registration error in the hub logs rather
  than treating a running process as ready.
- DeviceIOHub no longer falls back to embedded LiveKit development credentials.
  The YAML loader generates a fresh pair when both values are missing or blank.
  Set both `api_key` and `api_secret` in a private `device_io_hub.yaml`, or inject
  `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`, to use a fixed pair.
- Boolean service settings now require YAML booleans or the strings `true`,
  `false`, `yes`, `no`, `on`, `off`, `1`, or `0` (case-insensitive). Numeric
  `1`/`0`, null values, and arbitrary strings now fail at startup instead of
  being interpreted by Python truthiness. This applies to vLLM eager, tool, and
  scheduling flags, Nemotron-Omni BF16 selection, voice-gate
  `listening_chime`, lab-monitoring `capture_marker_scans`, and tea-workflow
  `complete_on_skip`.
- Return audio is paced before IPC by the built-in voice transport and bounded
  independently for each participant in DeviceIOHub.
  `return_audio_max_buffer_s` defaults to 3 seconds; a custom or faulty producer
  that exceeds the queued-audio duration limit loses its oldest queued frames.
  The built-in voice transport requires at least `0.12` to maintain its 120 ms
  reserve. Increase the value for intentionally bursty custom producers, or
  decrease it for a tighter memory and latency bound when using a compatible
  custom producer.

## Local speech service

The local speech service changed from Piper to Pocket TTS with no compatibility
alias. Replace the `piper_tts` model preset, `piper_tts_server` command, and
`services/piper-tts/` path with `pocket_tts`, `pocket_tts_server`, and
`services/pocket-tts/`. Pocket TTS voice names differ from Piper voice names;
the checked-in profiles use the CC0 `bill_boerst` voice.

Pocket TTS now selects a GPU automatically by default, and the checked-in
model-server profiles require CUDA on GPU 0. Set `device: cpu` for CPU-only
execution or `cuda_visible_devices` to change GPU placement. CUDA warmup runs
within `startup_timeout_s`, so increase that timeout when cold initialization
exceeds 600 seconds. The service now resolves the PyPI Torch build instead of
the CPU-only index; Linux environments therefore include the CUDA library
footprint even when execution falls back to CPU.

## Removed SDK compatibility surfaces

This release removes deprecated SDK aliases and the standalone Pipecat
compatibility package. Update out-of-tree code as follows:

| Removed surface | Replacement |
|---|---|
| `xr_ai_agent` | Import `ProcessorEndpoint` and IPC types from `xr_ai_hub`. |
| `BrainProcessor` and `make_voice_pipeline` | Put application behavior in an `xr_ai_runtime.Agent` subscriber and let `xr_ai_voice.VoiceAgent` own the voice pipeline. |
| `run_voice_pipeline` | Configure `VoiceAgent` directly and run it with `await VoiceAgent.run(runtime)`; its media session is private. |
| `XRMediaHubTransport` | Construct `xr_ai_voice.HubVoiceTransport` and pass it to `VoiceAgent` only when another application component must share that existing hub boundary. |
| `VoiceSession` | Configure `VoiceAgent` with STT, TTS, VAD, gating, readiness probes, and closeables, then run it with the shared `AgentRuntime`. The lower-level media session is no longer public. |
| `VadConfig` | Import the unchanged tuning model from `xr_ai_voice`. |
| `GatedQueryFrame` | Subscribe to the application query topic carrying `xr_ai_voice.UserQuery`. |
| `ParticipantJoinedFrame`, `ParticipantLeftFrame`, and `InterruptionFrame` | Subscribe to application topics carrying `VoiceParticipantJoined`, `VoiceParticipantLeft`, and `VoiceInterrupted`. Participant identity comes from runtime metadata; voice-gate greetings remain session-owned. |
| `BrainResponseEndFrame` | Publish a finite `VoiceOutput`, or terminate an incremental response with `final=True`. |
| `VadSttProcessor`, `VoiceGateProcessor`, and `StreamingTtsProcessor` | Configure `VoiceAgent` with `VadConfig`, `VoiceGateConfig`, and `text_topic`; pipeline processors are private implementation details. |
| `SttClient` and `TtsClient` | Construct services through `xr_ai_models.make_stt` and `make_tts`, or use `OpenAICompatSTT` and `OpenAICompatTTS` directly. |
| `http_probe`, `mcp_probe`, and `wait_for_services` | Pass additional readiness callables through `VoiceAgent(probes=...)`; MCP readiness is no longer part of the voice SDK. |
| `xr_ai_pipecat.audio` conversion helpers | Let `VoiceAgent` own media conversion. If an application truly needs raw hub media, use `xr_ai_hub` types and own the format conversion. |
| `VoiceAgent.text_transform` and `text_ignore_topics` | `VoiceAgent` treats only untopiced client data as direct text. Use `text_input=False` to disable it; transform application queries in their subscribing agent. |
| `xr_ai_models.config`, `factory`, `openai_compat`, and `protocols` | Import public names directly from `xr_ai_models`. This includes `KIND_OPENAI_COMPAT`, `ModelKind`, `Category`, and `Spec`. |
| `LiveVisionTool` and `StreamingVisionTool` | Select with `CurrentFrameTool`, then pass its `ImageReference` to `ImageQueryTool` or `StreamingImageQueryTool`. |
| `HistoricalVisionTool` | Select with `VideoMemoryTools.get_historical_frame`, then pass its `ImageReference` to `ImageQueryTool`. |
| Recorded `query_video(start_us, end_us)` RPC | Use `get_historical_video(start_us, duration_seconds)` or `get_latest_video(duration_seconds)`. The new `query_video` name is VLM inference over caller-selected `TimedImage` values. |
| `get_frame_from_time(reference_time_us, second_ago)` | Subtract the offset in the caller and use `get_historical_frame(start_us)`. |
| `HistoricalFrameResult.path` and `SampledVideoFrame.path` | Read the canonical exported-frame location from `result.image.uri`. |
| `xr_ai_tools.qr_code.QRCodeTool` | Initialize `xr_ai_tools.marker_tracking.MarkerTrackingTool`; select QR and/or ArUco with `marker_types`, then use the same `track_markers` request and result contract for either family. |
| Implicit development credentials from `LiveKitConnectorConfig()` or `LiveKitConnector()` | Pass a `LiveKitConnectorConfig` with explicit `api_key` and `api_secret`. The YAML loader also accepts `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`. |

Pipecat remains an internal implementation detail of `xr-ai-voice`; applications
no longer assemble or subclass its frame processors.

The source directories now match their Python imports:
`agent-sdk/xr-ai-hub-client/` became `agent-sdk/xr-ai-hub/` and
`agent-sdk/xr-ai-agent-runtime/` became `agent-sdk/xr-ai-runtime/`. Distribution
names remain `xr-ai-hub-client` and `xr-ai-agent-runtime`, so package dependency
names do not change.

If upgrading a checkout that already downloaded model weights, follow
{ref}`the model-cache migration <migrating-model-caches-from-ai-services>` to
reuse the ignored caches rather than downloading them again.
