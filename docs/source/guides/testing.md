<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Testing

The `tests/` project exercises SDKs, utilities, service wrappers, sample
behavior, generated documentation invariants, and the DeviceIOHub IPC path.
Most tests are CPU-only and run without Docker or LiveKit. Node.js 24 is also
required for the browser camera tests.

The browser camera tests also import presets from the LiveKit version in the
existing web dependency lock. A fresh checkout must install that artifact
before running the suite. From the repository root, use the same setup as CI:

```bash
camera_sdk_dir="$(mktemp -d)"
camera_sdk_version="$(node -p "require('./dependency-manifest/web-xr-build/package-lock.json').packages['node_modules/livekit-client'].version")"
npm install --prefix "$camera_sdk_dir" --ignore-scripts --no-package-lock --no-save "livekit-client@${camera_sdk_version}"
export STREAMKIT_LIVEKIT_SDK_MODULE="file://${camera_sdk_dir}/node_modules/livekit-client/dist/livekit-client.esm.mjs"
```

An existing `client-samples/web-xr-build/node_modules` installation can also
supply the SDK when the environment override is unset. No LiveKit server or
camera device is needed for these tests.

## Run the suite

From the repository root:

```bash
uv --config-file uv.toml sync --project tests
uv --config-file uv.toml run --project tests pytest -v
```

CI runs the same selection on Python 3.11 through 3.14 from `tests/`, with the
project environment at the repository root for nltk compatibility. The
equivalent repository-root command is:

```bash
uv --config-file uv.toml run --project tests \
  pytest -v --tb=short --color=yes -m "not gpu"
```

Tests marked `integration` may start real CPU subprocesses and still run in CI.
Tests that require a real GPU, Docker, or NVENC use the `gpu` marker and run
only on a suitably configured developer host:

```bash
bash tests/run_local_gpu_tests.sh
```

Pass extra pytest arguments after the script name to select a file or test.

Run GPU tests serially, without pytest-xdist (`-n`). Each invocation generates
a fresh session ID and exports `XR_AI_TEST_SESSION` only while a GPU test runs.
`xr_ai_vllm` labels newly created containers `xr-ai-test-session=<id>` without
changing their launch fingerprint. Before and after each GPU test, the fixture
removes only containers with that exact session label, using immutable IDs.
Test-mode launcher operations also check ownership before reusing, restarting,
stopping, or removing a container. Ordinary sample launches are unchanged.

If Docker ownership inventory fails or times out, the test is skipped rather
than assuming the host is empty. Any foreign `xr-ai-vllm-*` or `xr-ai-nim-*`
container, running or stopped, causes **all** GPU tests to skip, including
NVENC and LiveKit tests that launch no model server. This conservative guard
avoids GPU contention and accidental reuse or replacement of stopped servers.

Fixture teardown handles test and fixture failures, but cannot run after
pytest is forcibly killed. A later session does not adopt the dead session's
label or automatically reclaim its containers. To recover, inspect
`docker ps -a --filter label=xr-ai-test-session` and confirm the owner of each
leftover container is no longer running. Remove only that confirmed container
with `docker rm -f <container-id>`, then rerun the tests. Never purge by label
presence alone: another worktree may still be using a different session.

The two Docker cleanup integration tests are marked `gpu` and additionally
require a local `busybox` image. They use uniquely named, labelled throwaway
containers; CPU runs do not contact Docker during their collection. The nightly
workflow retains its separate dedicated-runner cleanup policy.

## NeMo service qualification

The STT, Magpie TTS, and speaker STT projects override NeMo's Hydra and
Lightning compatibility caps to use Hydra 1.3.7, Lightning 2.6.6, and a patched
fsspec release. Successful lock resolution and CPU tests do not establish
compatibility with real model checkpoints or CUDA inference. Qualify all three
services on the intended Linux GPU before deploying this dependency stack.

From the repository root, synchronize each isolated inference environment:

```bash
uv --config-file uv.toml sync --project services/stt-server
uv --config-file uv.toml sync --project services/magpie-tts
uv --config-file uv.toml sync --project services/speaker-stt
```

Run the existing STT and Magpie GPU smoke tests serially on an idle test host:

```bash
bash tests/run_local_gpu_tests.sh test_gpu_stt_server.py test_gpu_magpie_tts.py
```

Confirm that the tests execute rather than skip. The STT smoke test uses a
sine wave and checks endpoint wiring; it does not measure recognition quality.
Also transcribe a known speech recording, synthesize and listen to Magpie
output, and compare startup time, inference latency, and GPU memory with the
previous environment. Exercise both cached checkpoint restoration and the
model-download path.

The CPU speaker tests substitute model objects. For speaker STT, run the real
service with its pinned ASR and diarization checkpoints and test streaming
speech, two-speaker overlap, enrollment, interruption, session reset, and
reconnection. Confirm speaker attribution, transcript ordering, bounded memory,
and final decoding. Refer to {doc}`/components/ai-services` for the private
speaker service and its configuration. Record exact revisions, resolved
versions, hardware, test inputs, results, and remaining failures with the
qualification evidence.

## Apple microphone transaction tests

With Xcode 26.3 or newer selected, run the real StreamKit package tests on an
installed iOS Simulator. From `client-samples/ios-visionos/StreamKit`:

```bash
xcodebuild -scheme StreamKit \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  CODE_SIGNING_ALLOWED=NO test
```

Replace the simulator name with one installed in your Xcode environment. The
`ios-microphone-tests` workflow runs these tests and builds the unsigned iOS
and visionOS app targets from a clean checkout with dependency resolution
enabled. It compiles the actual app and LiveKit backend.
The transaction tests substitute physical capture and transport operations to
check rollback failures, retained cleanup resources, cancellation, request ordering,
and disabled audio.
They do not verify usable remote audio or the operating system's microphone
indicator; those checks require a physical device and a running server.

## Coverage boundaries

The suite includes:

- package API and model-profile behavior;
- launcher lifecycle, credentials, GPU selection, and persistent services;
- native tools, voice runtime, VAD, voice gating, and web events;
- service and sample configuration, wire contracts, and application behavior;
- documentation, CLI, dependency-map, and repository-layout invariants;
- participant subscription, attribution, and return-traffic isolation over real
  ZMQ IPC endpoints.

The routing tests model connectors as distinct clients and processor endpoints
as distinct agents. `test_cross_talk.py` is the canonical no-cross-talk suite:
it covers multi-client and multi-agent fan-out, interleaving, join and leave
events, filter isolation, attribution, ordering, and participant-targeted return
traffic.

## Measure coverage

CI measures coverage on Python 3.12 and publishes a per-project table to the job
summary, with the raw reports attached as the `coverage` artifact. The report is
informational and does not gate the build. The measured packages are listed in
`source_pkgs` in `tests/pyproject.toml`: the SDK, utilities, and services that
the test project installs or puts on its `pythonpath`. A package missing from
that list does not appear in the table. To reproduce the report, run from
`tests/` with the project environment at the repository root:

```bash
UV_PROJECT_ENVIRONMENT=../.venv uv --config-file ../uv.toml run coverage run -m pytest -m "not gpu"
UV_PROJECT_ENVIRONMENT=../.venv uv --config-file ../uv.toml run coverage json
python3 ../.github/scripts/coverage_summary.py coverage.json
```

Beyond line and branch coverage, the table counts the `except` clauses in each
project and how many of them ran at least once. An `except` body runs only when
a test provokes the failure, so this number shows where failure handling is
untested. Add a test that exercises a handler before changing it.

## Adding tests

Keep shared endpoint and polling helpers in `conftest.py`, `_helpers.py`, or
`_helpers_subprocess.py`. Mark a test `gpu` whenever it needs GPU hardware,
Docker, or NVENC; otherwise it is expected to run on the stock CI runner. Use
the narrowest existing test module for the behavior, and add an integration
test only when a real process boundary is material to the contract.

## Build the documentation

From the repository root, install the pinned documentation toolchain in a
separate environment and run the same strict checked-out build as the docs PR
check:

```bash
uv venv docs/.venv
uv pip install --python docs/.venv/bin/python -r docs/requirements.txt
make -C docs strict SPHINXBUILD="$PWD/docs/.venv/bin/sphinx-build"
```

The build validates public API documentation, generates the CLI and
configuration catalogs, treats Sphinx warnings as errors, and checks the
rendered Python reference for private implementation details. Open
`docs/_build/current/index.html` to inspect the result. The deployment workflow
also runs `make -C docs strict-versions` with `sphinx-multiversion` installed
from the same requirements and full release-tag history available.
