<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Skills

## Overview

A coding agent can help turn your application idea into a running XR AI
project, but it needs the repository's setup and development conventions.
These skills give it those workflows so it can prepare the environment, launch
a demo, and build toward your application's goal.

Skills are small `SKILL.md` modules under the open
[Agent Skills](https://agentskills.io) spec that a coding agent installs and
follows.

## Run the browser demo

Choose `run-demo` to try xr-ai. Paste this to your coding agent:

```{literalinclude} /_snippets/agent-demo-prompt.txt
:language: text
```

`run-demo` handles preparation with `setup`, leaves the Simple VLM browser demo
running, and returns its link, interaction verification status, and stop
instructions. Local model servers remain running independently of the sample;
model shutdown is a separate action.

The demo uses the deprecated Simple VLM sample, which remains available for
trying the stack. For new voice-and-vision application work, use
{doc}`/reference/conversation-example` as the recommended foundation.

Prefer to do it by hand? Follow {doc}`/getting_started/quickstart`.

## Set up for development

To prepare a checkout and its prerequisites with `setup`, paste this
to your agent:

```{literalinclude} /_snippets/agent-setup-prompt.txt
:language: text
```

The skill has the agent ask two setup questions (latest release or `main`;
models on the local GPU or a hosted endpoint), clone the repo, and route
itself to the working contract, the docs, and the reference sample. An agent
with no skills mechanism can follow the `SKILL.md` contents directly.

## Build your application

For application work, `build-your-app` is also a complete entry point. It uses
the published build guide before the checkout exists and continues with the
same guide from the selected repository ref after cloning.

To start directly with an application, describe its goal in this prompt and
paste it to your coding agent:

```{literalinclude} /_snippets/app-build-prompt.txt
:language: text
```

## Available skills

| Skill | What it does |
|---|---|
| [`run-demo`](https://github.com/NVIDIA/xr-ai/blob/main/skills/run-demo/SKILL.md) | Runs the Simple VLM browser demo and returns its connection link and verification status |
| [`setup`](https://github.com/NVIDIA/xr-ai/blob/main/skills/setup/SKILL.md) | Prepares a checkout and prerequisites for development, with pointers to the working contract, docs, and reference sample |
| [`build-your-app`](https://github.com/NVIDIA/xr-ai/blob/main/skills/build-your-app/SKILL.md) | Guides your coding agent from checkout setup through building an application |
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
REF=main  # or a release tag such as v0.4.0-beta
SKILL=run-demo
curl -fsSL --create-dirs -o "$SKILLS_DIR/$SKILL/SKILL.md" \
  "https://raw.githubusercontent.com/NVIDIA/xr-ai/$REF/skills/$SKILL/SKILL.md"
```

`gh-manage-xr-ai-reviews` composes `gh-review-xr-ai`; install both review
skills together. The review skills also ship `agents/openai.yaml`, optional
Codex interface metadata for their display name, short description, and default
prompt. After setting `SKILL` to either review skill, install that metadata with:

```bash
curl -fsSL --create-dirs -o "$SKILLS_DIR/$SKILL/agents/openai.yaml" \
  "https://raw.githubusercontent.com/NVIDIA/xr-ai/$REF/skills/$SKILL/agents/openai.yaml"
```

If the requested skill is absent at the selected release, ask the user before
using `main` for the skill, checkout, and documentation. An agent without a skill
installation mechanism can read the downloaded `SKILL.md` and follow it
directly in the current session.
