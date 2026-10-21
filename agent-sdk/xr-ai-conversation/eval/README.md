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
uv run --project . python eval/conflict_eval.py
uv run --project . python eval/compaction_eval.py
uv run --project . python eval/background_eval.py
```

Each script accepts `--base-url` and `--preset` (default
`http://localhost:8108` with `nemotron_omni`), or `--models path/to/models.json`
with `--model`, and `--samples` to override the sampling in its case file. A
script exits non-zero when a check fails.

| Eval | Cases | Checks |
|---|---|---|
| `conflict_eval.py` | `cases/conflicts.yaml` | A summary older than newer, contradicting material: answers follow the newer material |
| `compaction_eval.py` | `cases/session.yaml` | One long session: no prompt-time drops, summaries within limits, no failed compactions, every probed fact still in the final prompt, and probe recall |
| `background_eval.py` | `cases/background.yaml` | Compaction duration and kept facts per `compaction_thinking_budget`, and foreground first-token latency while compaction runs |
