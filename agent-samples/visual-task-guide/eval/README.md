<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Visual task guide deployed-model eval

The harness exercises both model-driven prompt paths against configured
services:

- The caption prompt evaluates generated two-finger and closed-fist fixtures
  through the deployed VLM with the same 40-token ceiling as the worker.
- `TaskGuideAgentConfig` performs bounded native dense retrieval, then uses
  one deployed NAT agent pass with real task state and a latest observation.

Start the shared model servers and the visual task guide stack, then run:

```bash
uv run --project agent-samples/visual-task-guide/eval visual_task_guide_eval
```

Run selected cases or save the complete report:

```bash
uv run --project agent-samples/visual-task-guide/eval visual_task_guide_eval \
  --case rag_hand_presentation_answer \
  --output agent-samples/visual-task-guide/eval/results/local.json
```

The harness checks the structured finger count, the 30-word guide limit,
native dense RAG use, and immutable task revision. Native workflow tests separately
cover deterministic next-step and current-step validation queries. Before
model calls the harness audits distinctive fixture markers against both prompts
to prevent test leakage.
