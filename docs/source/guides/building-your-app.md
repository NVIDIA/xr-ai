<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Build your application

Read this when building an XR AI application. The directions below assume the
application is under `apps/<your-app>/`. This directory is for applications
that are not part of the repository's sample catalog. The XR AI SDK
distributions are not published to a package registry; they resolve through
relative `[tool.uv.sources]` entries in each application's `pyproject.toml`.

## Start from a source checkout

Use the latest stable release by default, falling back to the latest
prerelease when no stable release exists. Use `main` when the application
needs unreleased changes. The documentation and checkout must use the same
ref: use the `/latest/` documentation with the selected release or `/main/`
with `main`.

Create the checkout:

```bash
git clone https://github.com/NVIDIA/xr-ai.git
cd xr-ai
```

To use the latest stable release, or the latest prerelease when no stable
release exists, run:

```bash
release_tag="$(git tag --list 'v*' | python3 .github/scripts/select_latest_docs_release.py)"
if [ -n "$release_tag" ]; then
  git checkout "$release_tag"
else
  echo "No release tag found; continuing on main." >&2
fi
```

After checking out a release, confirm that
`docs/source/guides/building-your-app.md` and
`skills/build-your-app/SKILL.md` exist in the checkout. If either is absent,
ask before switching the documentation, skill, and checkout to `main`.

Read `AGENTS.md`, then review the
{doc}`system requirements </getting_started/requirements>` and
{doc}`credential sources </getting_started/credentials>` before resolving the
application. Choose local model services or hosted endpoints. If setup already
selected one, keep that choice. The copied reference application uses local
model services by default. For hosted inference, refer to
{doc}`/guides/deploying-with-nim`; the hub still needs NVENC and NVDEC on its
host as described in the system requirements.

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
mkdir -p apps/my-app
git archive HEAD agent-samples/simple-vlm-example |
  tar -x --strip-components=2 -C apps/my-app
mv apps/my-app/worker/simple_vlm_example_worker apps/my-app/worker/my_app_worker
mv apps/my-app/yaml/simple_vlm_example_worker.yaml apps/my-app/yaml/my_app_worker.yaml
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

Substitute the application's real kebab-case name for `my-app`, then derive its
snake-case package and CamelCase class names mechanically. Apply the
replacements in the listed order across the entire copied tree, including
comments and docstrings. Update any remaining `agent-samples/`
working-directory references to `apps/`, and change prose that still describes
the application as a repository sample. The following check should produce no
output:

```bash
grep -rniE 'simple[-_ ]?vlm' apps/my-app
```

The `apps/` and `agent-samples/` directories are at the same depth, so the
copied projects' relative sources continue to work. Keep each
`[tool.uv.sources]` entry paired with its distribution in
`[project].dependencies`. Every unpublished XR AI dependency must retain a
repository source mapping.

Use this map to separate reusable scaffolding from the VLM example:

| Files | Reuse | Replace or adapt |
|---|---|---|
| `main.py` and the root `pyproject.toml` | Keep the `run_stack` entry point, DeviceIOHub process, worker process, and launcher source entries. | Rename the project, command, logging namespace, worker command, and worker YAML path. Update `project.description` to describe the application. Add only application-owned service processes. |
| `yaml/device_io_hub.yaml` | Keep the typed DeviceIOHub configuration shape. | Set the room, ports, web client, and network behavior. Refer to the [DeviceIOHub reference YAML](https://github.com/NVIDIA/xr-ai/blob/main/services/device-io-hub/device_io_hub.yaml) for coupled port and URL changes. |
| `worker/pyproject.toml`, `__init__.py`, and `__main__.py` | Keep the named-package layout, console entry point, argument parsing, and delegation to `run_app()`. | Rename the distribution, package, and entry point. Remove dependencies only after their imports and features are gone. |
| `worker/.../config.py` and `yaml/*_worker.yaml` | Keep typed loading and paths resolved relative to the YAML file. | Replace the prompt, frame, model, voice, and application-specific settings with the fields the application consumes. |
| `worker/.../app.py` | Keep the `AgentRuntime` composition and the `VoiceAgent` lifecycle when the application uses voice. | Replace the VLM warmup, model roles, and paired `CurrentFrameTool` and `StreamingImageQueryTool` composition with the application's services and agents. |
| `worker/.../agent.py` and `worker/.../prompts/system.txt` | Keep the participant-scoped task ownership, detached-task context and scope, and cancellation patterns. Refer to {doc}`/reference/agent-sdk-runtime`. | Replace the vision question-and-answer workflow, topics, tool calls, and prompt. |
| `yaml/models.json` | Keep the adapter and endpoint separation. | Declare only the logical model roles, client adapters, and shared endpoints the application needs. Do not add model-server deployment or readiness settings. |
| `yaml/voice_gate.yaml` | Keep it when the application uses voice. | Choose conversation controls or direct speech using {ref}`voice-conversation-controls`, and test the application's first spoken request. |
| `README.md` | Keep a short configure and run path. | Describe the application, its owned and reused processes, settings, and exact start commands. |

Rewrite the copied README as application-owned documentation. Replace its title
and the “Simple VLM example” and “This sample” descriptions, and remove links to
the Simple VLM sample reference. In particular, make sure the name replacements
did not leave a nonexistent `/reference/my-app.html` link. Change its working
directory to `apps/my-app/` and use the model-server command shown below.

The copied DeviceIOHub YAML leaves credentials blank, so the hub generates a
fresh pair at startup. To use a fixed pair, set `LIVEKIT_API_KEY` and
`LIVEKIT_API_SECRET` in the launch environment. Never put actual keys or tokens
in source files. Refer to {doc}`/getting_started/credentials` for the supported
credential sources.

Frame selection and vision inference use separate public paths: import
live-frame selection from `xr_ai_tools.current_frame` and image inference from
`xr_ai_tools.vision`.

## Resolve and run the application

From `apps/my-app/`, resolve both projects before starting the stack:

```bash
uv --config-file ../../uv.toml sync
uv --config-file ../../uv.toml sync --project worker
```

### Local model services

The model launcher requires `HF_TOKEN`. Configure it through one of the
sources in {doc}`/getting_started/credentials` before starting the application.
The copied application reuses speaker ASR, VLM, and TTS services. Start those shared
services first and wait for the launcher to report that all processes are
ready:

```bash
uv --config-file ../../uv.toml run \
  --project ../../agent-samples/simple-vlm-example/model-servers simple_vlm_example_model_servers
```

### Hosted endpoints

Follow {doc}`/guides/deploying-with-nim` to configure and validate the STT, VLM,
and TTS roles in the application's `yaml/models.json`. Keep a copy of the local
profile before replacing its entries. Apply the guide's voice-gate settings
for hosted speech to disable the copied local speaker backend. Start the
application after endpoint validation.

### Start the application

After the selected model environment is ready, start the application from
`apps/my-app/`:

```bash
uv --config-file ../../uv.toml run my_app
```

Open the full web-client URL printed by DeviceIOHub, including its scheme and
port. For a client on another device, replace `localhost` with the XR AI host's
IP address or hostname reachable by that device. HTTPS is enabled by default.
Refer to {ref}`tls-for-the-web-client` for certificate trust on the client
device and {ref}`web-client-empty-response` if the browser reports an empty
response.

For the copied voice-and-vision application, select **Connect**, grant camera
and microphone access, choose **On-demand images** or **Live video** under
**Camera Mode**, then select **Start Microphone**. Use the conversation controls
configured in `yaml/voice_gate.yaml` before speaking the application's request.
The web client's **Send** control delivers a typed query to the agent and
bypasses speech gating. It checks the query path without exercising STT or
conversation controls.

Write the application's README with its intended client, exact connection
steps, voice mode, and an example request. Check the adapted commands and link
targets before handing the application to its user.

The application launcher owns only DeviceIOHub, the worker, and any other
application processes. The separate model-server launcher owns shared model
startup and readiness. The voice runtime checks required speaker ASR
readiness, and the copied worker exercises VLM inference as an application
warmup before announcing readiness. Keep or replace that probe according to the
capabilities the application must warm or verify. When a model role changes,
update `yaml/models.json` and the worker's service construction. Do not add
shared model services to the application's `PROCESSES` list. If a process
fails, use its launch context and run-log directory to investigate through
{doc}`troubleshooting`.

## Verify without hardware

Turn each requirement in the application brief into an observable acceptance
case. Include configuration defaults when a key is omitted, the intended first
request, and the recovery behavior for unavailable inputs. Keep these checks in
`apps/my-app/tests/` using the layout and command below.

From the repository root, load the worker configuration before starting
services. This check imports the renamed package, reads the application YAML,
and resolves packaged files such as the system prompt:

```bash
uv --config-file uv.toml run --project apps/my-app/worker \
  python - <<'PY'
from pathlib import Path
from my_app_worker.config import load_config

load_config(Path("apps/my-app/yaml/my_app_worker.yaml"))
PY
```

Exercise model wire behavior with `tests/_stub_openai.py`, which provides an
`httpx.MockTransport` implementation of OpenAI-compatible endpoints. The wire
tests exercise `xr_ai_models` clients without importing the application worker.
Use `tests/test_simple_vlm_example_wire.py` as the small STT, VLM, and TTS
wire-format pattern; use `tests/test_xr_render_demo_wire.py` for LLM tool-call
flows.

The `hub`, `make_connector`, and `make_processor` fixtures in
`tests/conftest.py` run DeviceIOHub IPC over local ZMQ sockets. They demonstrate
participant routing and worker-facing endpoint coverage without LiveKit, a
camera, a microphone, Docker, or a GPU. Pytest does not load that file for
tests under `apps/my-app/` because it is not an ancestor of the application
test directory. Keep application-owned fixtures in
`apps/my-app/tests/conftest.py`; copy and adapt only the repository fixtures the
application needs.

Add the application's pytest settings to `apps/my-app/pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
pythonpath = ["worker"]
markers = [
  "gpu: requires local GPU, Docker, or NVENC",
  "integration: starts a real service process",
]
```

Point model clients at a `StubOpenAI`-style transport, then run the application
test directory explicitly with the repository test environment:

```bash
uv --config-file uv.toml run --project tests \
  pytest -v apps/my-app/tests -m "not gpu"
```

Refer to {doc}`testing` for the complete test commands and marker rules.

Before reporting that the application is complete, match each requirement to
its test or client-interaction result. Exercise the intended first request
through a real client after the hardware-free checks pass. Record the selected
model environment, configuration, observed response, and any simulated media
or unverified requirement. Distinguish typed input from spoken input and record
whether STT and the configured conversation controls were exercised. Passing
tests and service readiness do not replace that client check.

## Understand the application boundary

Repository file checks exclude the top-level `apps/` directory. Application
projects do not enter `DEPENDENCIES.md` or the dependency manifest, and the
repository's Ruff and SPDX checks do not inspect application-owned files. The
sample documentation, configuration catalogs, and test discovery remain scoped
to `agent-samples/` and `tests/`. Git ignores `apps/*` by default so private
application work is not staged accidentally. Remove that ignore rule when the
application owner wants to track the application in the fork.

The rest of the checkout remains repository-owned. Commit-wide hooks such as
DCO sign-off still run when installed, and changes outside `apps/` continue to
follow `AGENTS.md` and the repository's contribution checks.
