---
name: build-your-app
description: >-
  Build, adapt, run, or verify an application on NVIDIA xr-ai, whether or not
  the repository has already been cloned.
---
<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Build your XR AI application

Use the release and model environment already selected during setup. Ask for
any missing choice: latest stable release (the default, with the latest
prerelease as a fallback when no stable release exists) or `main`, and local
model services or hosted endpoints. Then follow the build-your-application
guide that matches that ref. Before a checkout
exists, use the published
[`latest`](https://nvidia.github.io/xr-ai/latest/guides/building-your-app.html)
guide for the latest release or the
[`main`](https://nvidia.github.io/xr-ai/main/guides/building-your-app.html)
guide for `main`. Once the guide creates the checkout, confirm that
`skills/build-your-app/SKILL.md` and
`docs/source/guides/building-your-app.md` exist at the selected ref. If either
is absent, ask the user before using `main` for the skill, guide, and checkout.
Then read `AGENTS.md` and continue with the guide from the checkout.
Follow the guide's selected model-environment path and verification steps.
Derive application-owned acceptance checks from the user's brief, including
configuration defaults and unavailable-input recovery. After configuration
and hardware-free checks, verify the intended first request through a real
client and give the user the application's connection and interaction steps.
Report which requirements passed and which remain unverified. For
launch failures, follow `docs/source/guides/troubleshooting.md` using the
reported process context and run-log directory.
