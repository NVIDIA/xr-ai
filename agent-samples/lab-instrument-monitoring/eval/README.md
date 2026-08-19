<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Foreground routing eval

`cases.yaml` verifies the prompt's tool-routing rules against the configured
LLM. Start `model-servers`, then run from the sample root:

```bash
uv run --project worker python eval/eval.py
```

The eval checks the complete first model action, including the exact tool-call
count and validation of every call against the worker's request model. Current
visible facts must select the argument-free live-frame tool; the worker passes
the original user request to the VLM. Recent events must select monitoring history, and ordinary conversation or general
knowledge must not select a tool. Background monitoring requests must select
its start, stop, or status control.

`visual_cases.yaml` exercises the VLM-facing prompts with generated images. It
covers monitor baseline, changed, unchanged, adversarial focus and visible
instruction text, plus same-device, missing-display, ambiguous-association, and
visible-instruction instrument cases:

```bash
uv run --project worker python eval/visual_eval.py
```
