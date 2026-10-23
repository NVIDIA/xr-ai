---
name: setup
description: Prepare an NVIDIA xr-ai checkout and prerequisites. Use for initial installation, cloning, or development setup, not routine work in a prepared checkout. Use run-demo for a running browser demo.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Set up an XR AI checkout

Prepare a checkout and its prerequisites for development. The linked docs hold
the procedures. Paths are relative to the repo root, so clone before following
them.

## First: two setup questions

Use choices the user already supplied, ask for any missing choice, then tailor
the pointers below to the answers.

1. **Release or main?** Build against the latest release tag (default) or
   `main`. This selects `<ref>` for the skill, checkout, and documentation.
   Before giving setup steps, require a copy of this skill from `<ref>`. If
   this copy's source ref differs or is unknown, retrieve it from
   `https://raw.githubusercontent.com/NVIDIA/xr-ai/<ref>/skills/setup/SKILL.md`
   and continue with that copy. If the file is absent at a release tag, retrieve
   `https://raw.githubusercontent.com/NVIDIA/xr-ai/<ref>/skills/getting-started/SKILL.md`
   instead. Older releases use that name: follow only its checkout and
   prerequisite guidance, stopping before model-server or sample launch.
   If neither path exists at that tag, ask before using `main` for the skill,
   checkout, and docs. For other retrieval failures, stop and ask whether to
   retry. Use `https://nvidia.github.io/xr-ai/main/` for `main`, or
   replace `main` with the selected release tag.
   <https://nvidia.github.io/xr-ai/> opens the newest complete documentation
   set and can fall back to `main` when the release lacks current entry points.
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
  it prefers stable releases over prereleases). Use the selected tag
  in documentation URLs; the site landing page can fall back to `main`. If the
  selector prints no tag, remain on `main` and tell the user that no release tag
  was found.
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
- **Prerequisites:** check the selected environment against
  `docs/source/getting_started/requirements.md` and report missing requirements.
  Ask before privileged host changes, system package installation, daemon
  restarts, or reboots. Configure credentials using
  `docs/source/getting_started/credentials.md`. Use existing credential sources
  without printing secrets.
- **Running a sample:** the quickstart document is at
  `https://nvidia.github.io/xr-ai/<ref>/getting_started/quickstart.html`, with
  `<ref>` replaced as above.

## Setup complete

Return the checkout path and ref, selected model environment, and any remaining
prerequisite or credential blockers. Do not launch model servers or the sample.
