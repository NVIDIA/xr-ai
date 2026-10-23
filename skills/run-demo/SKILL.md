---
name: run-demo
description: Run the NVIDIA xr-ai Simple VLM browser demo and return its connection link. Use for trying or demonstrating xr-ai in a new or existing checkout, locally or on an existing remote GPU host such as Brev.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Run the XR AI browser demo

Execute the quickstart document's Simple VLM example and return a working
browser link, using `setup` when preparation is needed. Repository paths are
relative to the selected checkout's root, not the installed skill directory.

## Choose the environment

Use the checkout, repository ref, and model environment already selected by
the user. If the ref is unknown, ask whether to use the latest release or
`main`. Follow `setup` when the checkout, ref, or model environment is not
selected, or prerequisites are unverified. Retrieve it from
`https://raw.githubusercontent.com/NVIDIA/xr-ai/<ref>/skills/setup/SKILL.md`,
using the selected ref; no separate installation is needed.
Keep the skill, checkout, and docs on that ref. If this skill is absent at the
selected release, ask before using `main` for all three. Read the selected
checkout's `AGENTS.md`.

Default to the current host. If the user selects an existing remote host,
including Brev, obtain any missing connection details and execute there.
Provisioning a machine is a separate task. When needed, follow `setup` on that
host to choose local or hosted model inference.

## Prepare and launch

Before launch, follow `docs/source/guides/customizing-model-servers.md` for
existing services, reuse requirements, replacement behavior, and shutdown
scope. Reuse a compatible running demo; obtain permission before replacing
existing services.

Execute the Simple VLM procedure in
`docs/source/getting_started/quickstart.md`. For hosted inference, use the
endpoint configuration validated during setup and skip local model startup.

Keep the sample alive in a managed terminal task. Record its directory,
commands, GPU profile, stop handle, logs, and preexisting services, using reuse
messages and startup logs as evidence. Wait for readiness per the quickstart.
If the deployment uses `--state-file` for its models, keep that file on every
model start and stop command. Stop the consuming agent before reconciling a
different sample; do not adopt untracked services or discard the state file.
For failures, follow `docs/source/guides/troubleshooting.md` and the model-server
guide, recording retained services. Report actionable blockers.

## Connect and verify

Check the web-client URL from the browser's machine. For a remote host, follow
`docs/source/getting_started/networking.md` for the browser address, TLS, and
WebRTC media access. Obtain permission before changing firewall rules or
broadening network exposure beyond the configured deployment.

Open the page when browser tooling is available. Follow the quickstart's
connection and interaction steps, using a camera frame when media permissions
allow it. Verify a connected client receives an agent reply. If browser
interaction is unavailable, state that the conversation remains unverified.

Keep the handoff short: the browser link, remaining certificate and connection
actions, one interaction from the quickstart, verification status, any
configuration edits, and how to stop this run. Leave the demo running for the
user. In the stop instructions, identify the recorded launcher task or PID and
explain how to send it Ctrl+C (SIGINT) or SIGTERM so it cleans up its children
when the user is finished.

Give model shutdown instructions from the model-server guide using the launch
checkout and GPU profile, explaining their whole-stack scope. If services
predate this run or ownership is uncertain, leave models running unless the
user authorizes that shutdown.
