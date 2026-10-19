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

`gpu` tests remove only the model-server containers their own session created
(labelled `xr-ai-test-session=<id>` by `xr_ai_vllm` via `XR_AI_TEST_SESSION`).
They never remove containers that were already running, such as persist-mode
servers in live use; if any `xr-ai-vllm-*` or `xr-ai-nim-*` container is
running, the `gpu` test is skipped with a message naming it.

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
