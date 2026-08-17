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

The eval checks the first model action. Current visible facts must select the
live-frame tool, recent events must select monitoring history, and ordinary
conversation or general knowledge must not select a tool. Background monitoring
requests must select its start, stop, or status control.
