<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Build your application

Read this when building an XR AI application that will not be contributed as a
repository sample. To prepare a repository sample, follow
{doc}`adding-a-sample` instead. The directions below assume the application is
under `agent-samples/<your-app>/`. The XR AI SDK distributions are not published
to a package registry; they resolve through relative `[tool.uv.sources]` entries
in each application's `pyproject.toml`.

## Choose the layers your application needs

```{list-table}
:header-rows: 1
:widths: 35 65

* - Package
  - Use it for
* - {doc}`xr-ai-launcher </reference/python/xr_ai_launcher/index>`\
    Import: `xr_ai_launcher`\
    Source: `utils/xr-ai-launcher`
  - Stdlib-only process management for starting DeviceIOHub, workers, and
    application services in dependency order.
* - {doc}`xr-ai-hub-client </reference/python/xr_ai_hub/index>`\
    Import: `xr_ai_hub`\
    Source: `agent-sdk/xr-ai-hub`
  - IPC with DeviceIOHub, participant events, data and audio messages, and live
    frame access.
* - {doc}`xr-ai-logging </reference/python/xr_ai_logging/index>`\
    Import: `xr_ai_logging`\
    Source: `utils/xr-ai-logging`
  - Shared logging setup used by the copied orchestrator and worker.
* - {doc}`xr-ai-models </reference/python/xr_ai_models/index>`\
    Import: `xr_ai_models`\
    Source: `agent-sdk/xr-ai-models`
  - Typed LLM, VLM, STT, TTS, and embedding clients selected from deployment
    profiles.
* - {doc}`xr-ai-agent-runtime </reference/python/xr_ai_runtime/index>`\
    Import: `xr_ai_runtime`\
    Source: `agent-sdk/xr-ai-runtime`
  - Typed agent lifecycle, registration, publication, and participant-scoped
    subscriptions.
* - {doc}`xr-ai-tools </reference/python/xr_ai_tools/index>`\
    Import: `xr_ai_tools`\
    Source: `agent-sdk/xr-ai-tools`
  - Typed `Tool` and `ToolSet` objects. Add `relay` for `run_tool_loop()`,
    `frames` for live-frame selection, and `vision` for VLM query tools.
* - {doc}`xr-ai-voice </reference/python/xr_ai_voice/index>`\
    Import: `xr_ai_voice`\
    Source: `agent-sdk/xr-ai-voice`
  - The `VoiceAgent` runtime for hub transport, STT, voice gating, TTS,
    interruption, readiness, and cleanup.
* - {doc}`xr-ai-voicegate </reference/python/xr_ai_voicegate/index>`\
    Import: `xr_ai_voicegate`\
    Source: `utils/xr-ai-voicegate`
  - Wake-phrase and follow-up-turn gating used by the copied voice application.
* - {doc}`xr-ai-web-events </reference/python/xr_ai_web_events/index>`\
    Import: `xr_ai_web_events`\
    Source: `agent-sdk/xr-ai-web-events`
  - A bounded browser view for compact application events explicitly selected
    by the application.
```

The distribution names, import names, and source directories are not always
identical. In particular, `xr-ai-hub-client` comes from
`agent-sdk/xr-ai-hub`, and `xr-ai-agent-runtime` comes from
`agent-sdk/xr-ai-runtime`. Refer to {doc}`/components/agent-sdk` for package
ownership boundaries and composition guidance.

## Copy the reference application

`agent-samples/simple-vlm-example/` is the smallest complete voice-and-vision
application. Copy its tracked files from the repository root and rename its
Python package and worker configuration:

```bash
mkdir agent-samples/my-app
git archive HEAD agent-samples/simple-vlm-example | tar -x --strip-components=2 -C agent-samples/my-app
mv agent-samples/my-app/worker/simple_vlm_example_worker agent-samples/my-app/worker/my_app_worker
mv agent-samples/my-app/yaml/simple_vlm_example_worker.yaml agent-samples/my-app/yaml/my_app_worker.yaml
```

Replace the sample names consistently:

| Existing name | Application name |
|---|---|
| `simple_vlm_example_worker` | `my_app_worker` |
| `simple-vlm-example` | `my-app` |
| `simple_vlm_example` | `my_app` |
| `simple-vlm` | `my-app` |
| `simple_vlm` | `my_app` |
| `SimpleVlmAgent` | `MyAppAgent` |

The copied projects already contain working relative sources for the assumed
`agent-samples/<your-app>/` location. Keep each `[tool.uv.sources]` entry paired
with its distribution in `[project].dependencies`. Every unpublished XR AI
dependency must retain a repository source mapping.

Use this map to separate reusable scaffolding from the VLM example:

| Files | Reuse | Replace or adapt |
|---|---|---|
| `main.py` and the root `pyproject.toml` | Keep the `run_stack` entry point, DeviceIOHub process, worker process, and launcher source entries. | Rename the project, command, logging namespace, worker command, and worker YAML path. Keep only the reused service declarations the application needs. |
| `yaml/device_io_hub.yaml` | Keep the typed DeviceIOHub configuration shape. | Set the room, ports, web client, and network behavior. |
| `worker/pyproject.toml`, `__init__.py`, and `__main__.py` | Keep the named-package layout, console entry point, argument parsing, and delegation to `run_app()`. | Rename the distribution, package, and entry point. Remove dependencies only after their imports and features are gone. |
| `worker/.../config.py` and `yaml/*_worker.yaml` | Keep typed loading and paths resolved relative to the YAML file. | Replace the prompt, frame, model, voice, and application-specific settings with the fields the application consumes. |
| `worker/.../app.py` | Keep the `AgentRuntime` composition and the `VoiceAgent` lifecycle when the application uses voice. | Replace the VLM warmup, model roles, and paired `CurrentFrameTool` and `StreamingImageQueryTool` composition with the application's services and agents. |
| `worker/.../agent.py` and `worker/.../prompts/system.txt` | Keep the participant-scoped task ownership, detached-task context and scope, and cancellation patterns. Refer to {doc}`/reference/agent-sdk-runtime`. | Replace the vision question-and-answer workflow, topics, tool calls, and prompt. |
| `yaml/models.json` | Keep the adapter, endpoint, and deployment separation. | Declare only the logical model roles and endpoints the application needs. |
| `yaml/voice_gate.yaml` | Keep it when speech input needs wake phrases and follow-up turns. | Tune the behavior or remove it with the voice runtime. |
| `README.md` | Keep a short configure and run path. | Describe the application, its owned and reused processes, settings, and exact start commands. |

The copied DeviceIOHub YAML contains development-only credential placeholders.
They can remain in place: `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` from the
environment overwrite them at startup. Never put actual keys or tokens in
source files. Refer to {doc}`/getting_started/credentials` for the supported
credential sources.

Frame selection and vision inference use separate public paths: import
live-frame selection from `xr_ai_tools.current_frame` and image inference from
`xr_ai_tools.vision`.

## Resolve and run the application

From `agent-samples/my-app/`, resolve both projects before starting the stack:

```bash
uv --config-file ../../uv.toml sync
uv --config-file ../../uv.toml sync --project worker
```

The shared model launcher requires `HF_TOKEN` by default. Configure it through
one of the sources in {doc}`/getting_started/credentials` before starting the
copied VLM application. If the weights are already cached or unauthenticated
rate limits are acceptable, append `--allow-anonymous` to the `model_servers`
command below.

The copied application reuses STT, VLM, and TTS services. Start those shared
services first and wait for the launcher to report that all processes are
ready:

```bash
uv --config-file ../../uv.toml run --project ../model-servers model_servers
```

Then start the application from the same directory:

```bash
uv --config-file ../../uv.toml run my_app
```

Open the authenticated web-client URL printed by DeviceIOHub, grant the media
permissions the application needs, and connect.

The copied `launch_mode="reuse"` entries document local services that must
already be running; the launcher neither starts nor probes them. The copied
`VoiceAgent` session checks STT and TTS health plus the supplied VLM warmup
probe before it announces readiness. An application that replaces the voice
runtime must own equivalent checks for the services it needs. When a model role
changes, update `yaml/models.json` and the worker's service construction, then
update the reuse entries so `PROCESSES` continues to describe the local
shared-service assumptions. An external endpoint does not need a local reuse
entry.

## Verify without hardware

From the repository root, load the worker configuration before starting
services. This check imports the renamed package, reads the application YAML,
and resolves packaged files such as the system prompt:

```bash
uv --config-file uv.toml run --project agent-samples/my-app/worker python -c 'from pathlib import Path; from my_app_worker.config import load_config; load_config(Path("agent-samples/my-app/yaml/my_app_worker.yaml"))'
```

Exercise model wire behavior with `tests/_stub_openai.py`, which provides an
`httpx.MockTransport` implementation of OpenAI-compatible endpoints. The wire
tests exercise `xr_ai_models` clients without importing the application worker.
Use `tests/test_simple_vlm_example_wire.py` as the small STT, VLM, and TTS
wire-format pattern; use `tests/test_xr_render_demo_wire.py` for LLM tool-call
flows. For worker-level tests, follow the `sys.path.insert()` setup in
`tests/test_simple_vlm_example_worker.py` so the application worker package can
be imported without adding it to `tests/pyproject.toml`.

The `hub`, `make_connector`, and `make_processor` fixtures in
`tests/conftest.py` run DeviceIOHub IPC over local ZMQ sockets. They cover
participant routing and worker-facing endpoint behavior without LiveKit, a
camera, a microphone, Docker, or a GPU. Put application tests in
`tests/test_my_app_*.py`, point model clients at `StubOpenAI`, and run the
CPU-only selection from the repository root:

```bash
uv --config-file uv.toml run --project tests pytest -v -k my_app -m "not gpu"
```

Refer to {doc}`testing` for the complete test commands and marker rules.

## Avoid upstream repository checks

The repository's pre-commit hooks are optional and run only after they are
installed. When installed, the SPDX hook can add the NVIDIA copyright header to
staged application files, and a staged `pyproject.toml` triggers the dependency
generator, which inventories every project in the source tree. The full
repository documentation and test suites also treat top-level directories
under `agent-samples/` as repository samples.

For an application that is not a repository sample, leave the repository's
pre-commit hooks uninstalled and use the focused application test command
above. To use those hooks or the full repository checks, first edit the SPDX
hook exclusions, dependency-generator ignore rules, and sample catalogs for the
chosen application path.
