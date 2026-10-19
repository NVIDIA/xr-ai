<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Deployment inventory

The Brev deployment inventory is a read-only check for a Brev VM or a local
Linux GPU host. It discovers samples from the checkout instead of maintaining
a separate sample list. It does not deploy anything, install dependencies,
change drivers or firewall rules, or download models.

From the repository root, using Python 3.11 or newer:

```bash
python3 deployment/brev/catalog.py "$PWD"
python3 deployment/brev/catalog.py "$PWD" --resolve auto
```

The first command prints a JSON inventory with GPU facts, per-sample blockers,
and a recommended sample. JSON is the machine-readable result, not a second
configuration format. The second prints a tab-separated sample ID, project
directory, and console command; it exits nonzero if no sample qualifies.
Pass a sample ID instead of `auto` to resolve a particular sample. Simple VLM
is recommended when it qualifies; otherwise the first qualifying sample is
selected.

## Discovery and qualification

A sample has a top-level `agent-samples/*/pyproject.toml`, depends on
`xr-ai-launcher`, and declares exactly one console command. The inventory uses
the launcher's existing GPU-profile detection and requires a matching
`model-servers/yaml/<profile>.yaml` in that sample. Unrelated projects and
model-server launchers are excluded. There are no historical-release layout
adapters in this tool.

Host checks require Linux, NVIDIA driver 580 or newer, Docker 24 or newer with
the NVIDIA runtime, the NVENC and NVDEC driver libraries, and at least one GPU
reporting NVENC support. NVENC is reported as a capability, not an encoder count.
Spark GPUs remain visible when dedicated GPU memory is unavailable; their
`memory_mib` is `null`, not zero. XR Render additionally needs Node.js and npm
for its web client build.

Qualification is a preflight check, not a successful deployment guarantee.
Model credentials, free capacity, image availability, network access, and
inference warmup are verified only when running the sample. Refer to
{doc}`customizing-model-servers` for the normal model-server workflow and
{doc}`/getting_started/networking` for externally reachable client ports.
