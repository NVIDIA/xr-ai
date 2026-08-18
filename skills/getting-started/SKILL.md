---
name: getting-started
description: Get set up to build on the NVIDIA xr-ai stack (XR AI / XR-Media-Hub). Use when a user asks to install, clone, or do initial setup of xr-ai; not for routine work in an already-set-up clone.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Getting started with xr-ai

Pointers only; the linked docs hold the procedures. Paths are relative to the
repo root, so clone before following them.

## First: two setup questions

Ask the user, then tailor the pointers below to the answers.

1. **Stable or latest?** Build against the latest release tag (default) or
   `main`. This should match the ref this skill was fetched from; re-fetch the
   skill from the chosen ref if it does not. The answer selects both the git
   ref to check out and the version of the docs site to read:
   <https://nvidia.github.io/xr-ai/> serves the released version once one is
   published, with `main` and per-tag versions available.
2. **Local or remote GPU?** Models on the local GPU, or a hosted/remote
   endpoint?
   - **Local:** the Requirements section of the root `README.md` applies,
     including the VRAM table.
   - **Remote/hosted:** the LLM and VLM can run on hosted endpoints, and no
     local GPU is needed for the agent or hub. STT and TTS stay local. Read
     `docs/source/components/ai-services.md` (section "Hosting models on
     NVIDIA NIM") and `docs/source/getting_started/credentials.md`.

## Where everything is

- **Repo:** `git clone https://github.com/NVIDIA/xr-ai.git`, then check out
  the ref chosen above (latest release tag: `make -s -C docs latest-release`,
  run in the clone; it prefers stable releases over prereleases, matching what
  the docs site serves).
- **Working contract:** read `AGENTS.md` first. Every change must satisfy it,
  and its canonical-references table routes deeper topics.
- **Docs:** `docs/source/`; published versioned site at
  <https://nvidia.github.io/xr-ai/>.
- **Reference sample:** `agent-samples/simple-vlm-example/`, the one to study
  and copy.
- **Scaffolding a new sample:** `docs/source/guides/adding-a-sample.md`, full
  boilerplate templates.
- **Quickstart order:** root `README.md` Requirements first. Then
  `agent-samples/simple-vlm-example` runs directly;
  `agent-samples/model-servers` is a prerequisite for
  `agent-samples/xr-render-demo`.
