<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo eval harness

Offline and live regression tiers for the xr-render agent. Refer to the
[sample guide](https://nvidia.github.io/xr-ai/latest/reference/xr-render-demo.html#eval-harness)
for tier definitions, case placement, prompt-tuning policy, isolation rules,
and coverage boundaries. Run commands from the evaluation project:

```bash
cd agent-samples/xr-render-demo/eval && uv sync   # once

uv run xr_render_demo_eval
uv run xr_render_demo_eval utterances
uv run xr_render_demo_eval_supervisor
uv run xr_render_demo_eval_subagents placement

uv run xr_render_demo_live_smoke
uv run xr_render_demo_live_pose_matrix
uv run xr_render_demo_live_manip
uv run xr_render_demo_live_garble
uv run xr_render_demo_live_explore
```

<!-- Compatibility anchors for headings consolidated into the documentation. -->
<a id="live-drivers"></a><a id="prompt-tuning-law"></a>
<a id="prompt-tuning-loop"></a><a id="writing-a-case"></a>
<a id="dont-train-on-the-test-set"></a>
<a id="what-the-harness-does-not-cover"></a>
