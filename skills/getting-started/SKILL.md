---
name: getting-started
description: Compatibility entry point for older xr-ai setup prompts and installed skills. Routes to setup for a prepared development checkout or run-demo for a running browser demo.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Choose an XR AI workflow

Preserve the user's requested outcome and selected repository ref. If the ref
is unknown, ask whether to use the latest release or `main`. Retrieve and follow
the matching skill directly; no separate installation is needed:

- To try xr-ai with a running browser demo:
  `https://raw.githubusercontent.com/NVIDIA/xr-ai/<ref>/skills/run-demo/SKILL.md`.
- To prepare a checkout and prerequisites for development:
  `https://raw.githubusercontent.com/NVIDIA/xr-ai/<ref>/skills/setup/SKILL.md`.

Replace `<ref>` with the selected ref. If the requested outcome is unclear,
ask which workflow the user wants. A setup request does not authorize launching
the demo. If the destination skill is absent at a release tag, ask before using
`main` for the skill, checkout, and docs. Report other retrieval failures.
