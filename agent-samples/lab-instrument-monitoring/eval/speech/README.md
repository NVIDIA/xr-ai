<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lab speech replay inputs

The replay reads the instruments, starts background monitoring, changes their
readings, asks for updates, and stops monitoring. It uses the existing native
foreground and instrument-monitor agents.

Supply `lab_before` and `lab_after` images with the same readable instruments
and device-map markers. Captured photos work, or generate fixed raw camera
fixtures with the existing visual-eval renderer. From the sample root:

```bash
PYTHONPATH=worker uv run --project worker python - <<'PY'
from pathlib import Path
from eval.visual_eval import _instrument_scene

out = Path("/tmp/xr-lab-speech")
out.mkdir(exist_ok=True)
for name, reading in (("before", "99.0 A"), ("after", "25.0 A")):
    (out / f"{name}.png").write_bytes(_instrument_scene(
        left_reading="12.0 V", right_reading=reading, annotate=False,
    ))
PY
```

Pass these files to the shared runner:

```text
--sample lab-instrument-monitoring
--image lab_before=/tmp/xr-lab-speech/before.png
--image lab_after=/tmp/xr-lab-speech/after.png
```

The images contain the checked-in `device-1` and `device-2` QR markers, without
VLM annotation overlays. Native marker detection and device mapping remain part
of the replay. Keep both image files unchanged across model/hardware comparisons.
See the [shared runner guide](https://nvidia.github.io/xr-ai/latest/guides/evaluating-sample-models.html)
for service setup, audio preparation and timing interpretation.
