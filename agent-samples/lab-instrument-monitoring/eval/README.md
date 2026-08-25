<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Foreground routing eval

`cases.yaml` checks foreground routing and `visual_cases.yaml` checks the
monitor and instrument VLM prompts. Refer to the
[sample guide](https://nvidia.github.io/xr-ai/latest/reference/lab-instrument-monitoring.html#routing-and-visual-evals)
for the evaluation contract and coverage. Start `model-servers`, then run from
the sample root:

```bash
uv run --project worker python eval/eval.py
uv run --project worker python eval/visual_eval.py
```
