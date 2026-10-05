<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Testing

The `tests/` project exercises SDKs, utilities, service wrappers, sample
behavior, generated documentation invariants, and the DeviceIOHub IPC path.
Most tests are CPU-only and run without Docker or LiveKit. Node.js 24 is also
required for the browser camera tests.

## Run the suite

From the repository root:

```bash
uv --config-file uv.toml sync --project tests
uv --config-file uv.toml run --project tests pytest -v
```

CI runs the same selection on Python 3.11 and 3.12 from `tests/`, with the
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
check rollback failures, cancellation, disconnect ordering, and disabled audio.
The `AppModelTests` Xcode scheme compiles the full production AppModel and tests
microphone requests against a fake transport. Run it from
`client-samples/ios-visionos`:

```bash
xcodebuild -project StreamKitSample.xcodeproj -scheme AppModelTests \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  CODE_SIGNING_ALLOWED=NO test
```

Both test suites run in the Apple client workflow.
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

## Adding tests

Keep shared endpoint and polling helpers in `conftest.py`, `_helpers.py`, or
`_helpers_subprocess.py`. Mark a test `gpu` whenever it needs GPU hardware,
Docker, or NVENC; otherwise it is expected to run on the stock CI runner. Use
the narrowest existing test module for the behavior, and add an integration
test only when a real process boundary is material to the contract.
