<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai skill bank

Skills that set a coding agent up to work with
[xr-ai](https://github.com/NVIDIA/xr-ai). Each skill is a `SKILL.md` under the
open [Agent Skills](https://agentskills.io) spec.

## Available skills

| Skill | What it does |
|---|---|
| [`getting-started`](getting-started/SKILL.md) | Sets an agent up to build on xr-ai: repo, working contract, docs, reference sample |

## Setup

Download the skill into your agent's skills directory:

```bash
: "${SKILLS_DIR:?Set SKILLS_DIR to your agent's skills directory}"
curl -fsSL --create-dirs -o "$SKILLS_DIR/getting-started/SKILL.md" \
  https://raw.githubusercontent.com/NVIDIA/xr-ai/main/skills/getting-started/SKILL.md
```

No skills mechanism? Read
[`getting-started/SKILL.md`](getting-started/SKILL.md) and follow it directly
in the current session.
