---
name: run-demo
description: Run the NVIDIA xr-ai Simple VLM browser demo and return its connection link. Use for trying or demonstrating xr-ai in a new or existing checkout, locally or on an existing remote GPU host such as Brev.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Run the XR AI browser demo

Execute the setup and launch, then give the user a browser link and a first
interaction. Default to `agent-samples/simple-vlm-example`. All repository
paths below are relative to the selected checkout's root, not the installed
skill directory.

## Choose the environment

Use the checkout, repository ref, and model environment already selected by
the user. For initial setup, retrieve and follow the getting-started workflow
from
`https://raw.githubusercontent.com/NVIDIA/xr-ai/<ref>/skills/getting-started/SKILL.md`,
using the selected ref; no separate installation is needed. If the ref is
unknown, ask which to use. Keep the skill, checkout, and documentation on that
ref. If this skill is absent at the selected release tag, ask before using
`main` for the skill, checkout, and docs. Read `AGENTS.md` in the selected
checkout before continuing.

Default to the current host. If the user selects an existing remote host,
including Brev, obtain any missing connection details and execute there.
Provisioning a machine is a separate task. A remote GPU host and hosted model
endpoints are different choices: the sample host still needs the hardware
codecs documented at
`https://nvidia.github.io/xr-ai/<ref>/getting_started/requirements.html`.

## Prepare and launch

Read `agent-samples/simple-vlm-example/README.md` and
`docs/source/getting_started/quickstart.md` for commands, connection steps, and
an interaction. Follow `docs/source/getting_started/requirements.md` and
`docs/source/getting_started/credentials.md` for prerequisite checks, including
Docker access, available GPU memory, and disk space. Use existing credential
sources without printing secrets.

Inspect running services and the ports configured by the sample before launch.
Reuse a compatible running demo. Before invoking a model launcher against
occupied ports, establish that it will reuse the services unchanged, or obtain
permission to replace them. The command can evict other xr-ai containers and
local model listeners, and recreate containers with changed configuration.
Docker reuse requires the expected container name and matching launch-config
fingerprint; a healthy endpoint alone does not establish reuse. Server
identities are shared across samples. Refer to
`docs/source/guides/customizing-model-servers.md` for the stack workflow.

Record the launch directory, commands, GPU profile, task handles, run-log
paths, and which services predate this run. Use service reuse messages and
startup logs as evidence; if ownership is uncertain, ask before stopping.

For local inference, run the sample's own model-server command and wait for
it to report readiness and return before starting the sample. For configured
hosted endpoints, follow `docs/source/guides/deploying-with-nim.md` to validate
the required roles and skip local model startup. Keep the sample alive in a
managed terminal task, retaining its output and stop handle.

Wait for the sample launcher's `All processes ready` message as described in
the quickstart. Monitor progress in the run logs. For launch failures, follow
`docs/source/guides/troubleshooting.md` using the failing process and its logs;
record any retained model services. Stop at an actionable blocker when
credentials, access, or a host change needs the user's input.

## Connect and verify

Check the web-client URL from the browser's machine. On a remote host, adapt
the printed `localhost` address using
`docs/source/getting_started/networking.md` for TLS and WebRTC media access;
forwarding the web port alone does not establish media connectivity. Obtain
permission before changing firewall rules or broadening network exposure
beyond the configured deployment.

Open the page when browser tooling is available. Follow the quickstart's
connection steps and try one request with a camera frame when browser access
and media permissions allow it. A connected client receiving an agent reply
is the conversation check. If browser interaction is unavailable, return the
link as ready to try and state that the conversation remains unverified.

Keep the handoff short: the browser link, remaining certificate and connection
actions, one interaction from the quickstart, verification status, any
configuration edits, and how to stop this run. Leave the demo running for the
user. Stop the recorded sample launcher with Ctrl+C (SIGINT) or SIGTERM so it
cleans up its children.

The sample's model command with `--stop` stops every recognized server on its
configured port set, including reused services or services used by another
sample. Use the same checkout and `--gpu-profile` selection as at launch.
Explain that scope in the stop instructions. If any affected service predates
this run, leave models running unless the user authorizes that whole-stack
shutdown; `--stop` cannot selectively stop only the newly started services.
