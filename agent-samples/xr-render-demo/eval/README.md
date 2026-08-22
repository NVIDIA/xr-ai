<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo eval harness

Offline and live regression tiers for the xr-render agent. The
[sample guide](../../../docs/source/reference/xr-render-demo.md#eval-harness)
owns the tier definitions, case placement, prompt-tuning policy, isolation
rules, and coverage boundaries. Run commands from the eval project:

```bash
cd agent-samples/xr-render-demo/eval && uv sync   # once

uv run xr_render_demo_eval
uv run xr_render_demo_eval utterances
uv run xr_render_demo_eval_supervisor
uv run xr_render_demo_eval_subagents placement
```
