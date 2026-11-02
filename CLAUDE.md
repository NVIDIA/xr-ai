<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Working instructions

Follow the shared engineering guidance and routing in
[`AGENTS.md`](AGENTS.md), plus the nearest package or application README.
For repository contributions, follow
[Contributing and conventions](docs/source/guides/contributing-and-conventions.md).
Run relevant tests and Ruff before committing, and sign repository commits
with `git commit -s`. After pushing an upstream PR, check `gh pr checks`;
the change is not complete until required CI is green.

Never launch models ad hoc (bare `vllm serve`, `from_pretrained` scripts).
Start servers only through the repo entry points (sample orchestrators and
the `services/` wrappers): those are the paths that must be tested, and they
default the HuggingFace cache to the repo `models/` dir.
