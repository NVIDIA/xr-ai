<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Conversation context evals

Live-model evals for context compaction and the answer-prompt layout. Refer to
the [conversation SDK reference](https://nvidia.github.io/xr-ai/latest/reference/agent-sdk-conversation.html#context-compaction)
for the behavior they check. Start an OpenAI-compatible LLM (for example the
`model-servers` Nemotron Omni LLM on port 8108), then run from this package:

```bash
uv run --project . --group eval python eval/conflict_eval.py
uv run --project . --group eval python eval/compaction_eval.py
```

Each script accepts `--base-url` and `--preset` (default
`http://localhost:8108` with `nemotron_omni`), or `--models path/to/models.json`
with `--model`, and `--samples` to override the sampling in its case files. A
script exits non-zero when a check fails.

| Eval | Cases | Checks |
|---|---|---|
| `conflict_eval.py` | `cases/conflicts.yaml` | A summary disagrees with material from a different time, in the same or another section: each case must be answered with the newer fact in most samples |
| `compaction_eval.py` | `cases/session_lab.yaml`, `cases/session_renovation.yaml` | Long sessions with corrections and oversized exchanges and updates: no failed compactions or prompt omissions, every probed fact still in the final prompt, and each probe answered correctly in most samples |

Cases are seeded through the engine's public flow. Questions ask for a closed
answer, graded by whole words, and each case lists accepted and rejected
example answers that the repository tests use to check the graders offline.
`session_renovation.yaml` and the `held-out-` conflict cases use entities that
appear nowhere in the compaction prompts.
