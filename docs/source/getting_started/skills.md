<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Skills

Skills are how a coding agent sets itself up to work with xr-ai: small
`SKILL.md` modules under the open [Agent Skills](https://agentskills.io) spec
that the agent installs and follows. This is the primary way to get started.
Paste this to your agent:

```{literalinclude} /_snippets/agent-setup-prompt.txt
:language: text
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
| [`gh-develop-xr-ai`](https://github.com/NVIDIA/xr-ai/blob/main/skills/gh-develop-xr-ai/SKILL.md) | Develops small, self-contained XR-AI PRs and handles review feedback without scope creep |
| [`gh-review-xr-ai`](https://github.com/NVIDIA/xr-ai/blob/main/skills/gh-review-xr-ai/SKILL.md) | Reviews xr-ai PRs with strict scope discipline and comment-only feedback |
| [`gh-manage-xr-ai-reviews`](https://github.com/NVIDIA/xr-ai/blob/main/skills/gh-manage-xr-ai-reviews/SKILL.md) | Tracks an XR-AI review inbox with overall status, next actions, and safe batching |

The bank lives at
[`skills/`](https://github.com/NVIDIA/xr-ai/tree/main/skills) in the
repository; new skills land there with the features they cover.

## Manual installation

Set `REF` to the repository ref you will build against, choose a `SKILL` from
the table, and download it into the directory used by your coding agent:

```bash
SKILLS_DIR=/path/to/your/agent/skills
REF=main  # or a release tag such as v0.3.0
SKILL=getting-started
curl -fsSL --create-dirs -o "$SKILLS_DIR/$SKILL/SKILL.md" \
  "https://raw.githubusercontent.com/NVIDIA/xr-ai/$REF/skills/$SKILL/SKILL.md"
```

`gh-manage-xr-ai-reviews` composes `gh-review-xr-ai`; install both review
skills together. The GitHub workflow skills also ship `agents/openai.yaml`,
optional Codex interface metadata for their display name, short description,
and default prompt. After setting `SKILL` to a `gh-*` skill, install that
metadata with:

```bash
curl -fsSL --create-dirs -o "$SKILLS_DIR/$SKILL/agents/openai.yaml" \
  "https://raw.githubusercontent.com/NVIDIA/xr-ai/$REF/skills/$SKILL/agents/openai.yaml"
```

If the selected release predates the skill bank, use `main` for both the skill
and your checkout. An agent without a skill installation mechanism can read the
downloaded `SKILL.md` and follow it directly in the current session.
