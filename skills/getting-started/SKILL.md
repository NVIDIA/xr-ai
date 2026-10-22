---
name: getting-started
description: Get set up to build on the NVIDIA xr-ai stack (XR AI / DeviceIOHub). Use when a user asks to install, clone, or do initial setup of xr-ai; not for routine work in an already-set-up clone.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Getting started with xr-ai

Pointers only; the linked docs hold the procedures. Paths are relative to the
repo root, so clone before following them.

## First: two setup questions

Use choices the user already supplied, ask for any missing choice, then tailor
the pointers below to the answers.

1. **Release or main?** Build against the latest release tag (default) or
   `main`. This selects `<ref>` for the skill, checkout, and documentation.
   Before giving setup steps, require a copy of this skill from `<ref>`. If
   this copy's source ref differs or is unknown, retrieve it from
   `https://raw.githubusercontent.com/NVIDIA/xr-ai/<ref>/skills/getting-started/SKILL.md`
   and continue with that copy. If the file is absent at a release tag, that
   release predates the skill bank: stop and ask before using `main` for both
   the skill and the checkout. For other retrieval failures, stop and ask
   whether to retry. Use `https://nvidia.github.io/xr-ai/main/` for `main`, or
   replace `main` with the selected release tag.
   <https://nvidia.github.io/xr-ai/> opens the latest published release.
2. **Local or remote GPU?** Models on the local GPU, or a hosted/remote
   endpoint?
   - **Local:** read the matching version of the requirements page at
     `https://nvidia.github.io/xr-ai/<ref>/getting_started/requirements.html`,
     including the VRAM table. Replace `<ref>` with `main` or the selected
     release tag.
   - **Remote/hosted:** read the same versioned requirements page: the hub
     still needs an NVIDIA GPU with NVENC and NVDEC, even when inference is
     remote. Follow `docs/source/guides/deploying-with-nim.md` to configure
     and validate the application's required endpoints, including remote
     speech when selected. Read `docs/source/getting_started/credentials.md`
     for credential sources.

## Where everything is

- **Repo:** `git clone https://github.com/NVIDIA/xr-ai.git`, then check out
  the ref chosen above (latest release tag, run in the clone:
  `git tag --list 'v*' | python3 .github/scripts/select_latest_docs_release.py`;
  it prefers stable releases over prereleases, matching what the docs site
  serves). If the selector prints no tag, remain on `main` and tell the user
  that no release tag was found.
- **Working contract:** read `AGENTS.md` first. Every change must satisfy it,
  and its canonical-references table routes deeper topics.
- **Docs:** `docs/source/`; published versioned site at
  <https://nvidia.github.io/xr-ai/>.
- **Reference sample:** `agent-samples/simple-vlm-example/`, the one to study
  and copy.
- **Building your application:** `docs/source/guides/building-your-app.md`, the
  consumer path for choosing SDK layers, copying the reference, and verifying
  without hardware.
- **Preparing an upstream sample:** `docs/source/guides/adding-a-sample.md`,
  full repository boilerplate templates.
- **Quickstart order:** follow
  `https://nvidia.github.io/xr-ai/<ref>/getting_started/quickstart.html`, with
  `<ref>` replaced as above. For local models, start
  `agent-samples/simple-vlm-example/model-servers` and wait for it to report
  readiness before starting `agent-samples/simple-vlm-example`. For another
  sample, start that sample’s own `model-servers/` project instead. Stop the
  current model stack before switching samples, editing its server settings, or
  updating the checkout.
  For hosted models, validate the application's `yaml/models.json` through the
  deployment guide before starting the application; skip local model-server
  startup.
- **Launch failures:** follow `docs/source/guides/troubleshooting.md` and the
  failure message's process context and run-log directory. Model servers that
  had signaled readiness when cleanup began survive a startup failure or
  cancellation; rerun the same model command to reuse them. If the fix changes
  a server that already started, run the model command with `--stop` first so
  the new settings take effect. Use `--stop` to clear retained servers. Refer to
  `docs/source/guides/customizing-model-servers.md`.
