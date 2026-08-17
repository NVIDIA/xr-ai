<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Skills

Skills are how a coding agent sets itself up to work with xr-ai: small
`SKILL.md` modules under the open [Agent Skills](https://agentskills.io) spec
that the agent installs and follows. This is the primary way to get started.
Paste this to your agent:

```text
Set up xr-ai for me: fetch
https://raw.githubusercontent.com/NVIDIA/xr-ai/main/skills/getting-started/SKILL.md,
install it as a skill (or just follow it), and walk me through the setup.
```

The skill has the agent ask two setup questions (latest release or `main`;
models on the local GPU or a hosted endpoint), clone the repo, and route
itself to the working contract, the docs, and the reference sample. An agent
with no skills mechanism can follow the `SKILL.md` contents directly.

Prefer to do it by hand? Follow {doc}`/getting_started/quickstart`.

## Available skills

| Skill | What it does |
|---|---|
| [`getting-started`](https://github.com/NVIDIA/xr-ai/blob/main/skills/getting-started/SKILL.md) | Sets an agent up to build on xr-ai: repo, working contract, docs, reference sample |

The bank lives at
[`skills/`](https://github.com/NVIDIA/xr-ai/tree/main/skills) in the
repository; new skills land there with the features they cover.
