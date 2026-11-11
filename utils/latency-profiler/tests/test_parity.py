# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-page JS calculator must price a design exactly like the Python estimator."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from xr_latency.estimate import evaluate

ROOT = Path(__file__).resolve().parents[1]

MODEL = {
    "llm": {
        "chat_overhead_tokens": 54,
        "ttft_cold_vs_ctx": {"p50": {"intercept": 5.0, "slope": 0.07}, "p90": {"intercept": 8.0, "slope": 0.09}},
        "ttft_warm_vs_ctx": {"p50": {"intercept": 100.0, "slope": 0.001}, "p90": {"intercept": 130.0, "slope": 0.002}},
        "decode_total_vs_tokens_chat": {"p50": {"intercept": 70.0, "slope": 5.8},
                                        "p90": {"intercept": 90.0, "slope": 6.5}},
        "thinking": {"first_text": [{"thinking": False, "p50": 120.0, "p90": 150.0},
                                    {"thinking": True, "p50": 700.0, "p90": 900.0}]},
    },
    "decision": {"total_vs_history_bytes_labels4": {"p50": {"intercept": 67.0, "slope": 0.055},
                                                      "p90": {"intercept": 80.0, "slope": 0.06}}},
    "tts": {
        "stream_first_chunk": [{"chars": 20, "p50": 33.0, "p90": 40.0}, {"chars": 60, "p50": 34.0, "p90": 45.0}],
        "batch_total_vs_chars": {"p50": {"intercept": 78.0, "slope": 6.0}, "p90": {"intercept": 90.0, "slope": 6.2}},
        "audio_seconds": [{"chars": 20, "p50": 1.7, "p90": 1.8}, {"chars": 60, "p50": 3.8, "p90": 3.9}],
    },
    "vlm": {"chat_total": [{"res": "672x672", "images": 2, "p50": 250.0, "p90": 350.0}],
            "cached_total": [{"res": "672x672", "images": 2, "p50": 150.0, "p90": 200.0}]},
    "stt": {"end_to_final": [{"mode": "conversation", "utterance": "medium", "frame_ms": 10,
                              "p50": 640.0, "p90": 700.0}]},
}
DESIGNS = [
    [{"kind": "endpoint"}, {"kind": "route", "path": "misroute_declined_once", "history_bytes": 8000, "admit_ms": 120},
     {"kind": "speak", "prompt_tokens": 3000, "out_tokens": 90, "first_sentence_tokens": 18, "cached": True,
      "strategy": "sentence", "playback_ms": 140}],
    [{"kind": "fixed", "name": "x", "p50": 100, "p90": 160},
     {"kind": "vlm", "res": "672x672", "images": 2},
     {"kind": "tool_hop", "prompt_tokens": 2000, "hops": 3, "tool_ms": 250, "thinking": True},
     {"kind": "speak", "prompt_tokens": 2600, "out_tokens": 50, "strategy": "batch", "thinking": True}],
    [{"kind": "llm", "prompt_tokens": 1400, "out_tokens": 60, "reasoning_tokens": 700},
     {"kind": "speak", "prompt_tokens": 1400, "out_tokens": 60, "reasoning_tokens": 400, "strategy": "stream_full"}],
    [{"kind": "llm", "prompt_tokens": 500, "out_tokens": 10},
     {"kind": "speak", "prompt_tokens": 500, "strategy": "stream_full"}],
]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for the JS port")
@pytest.mark.parametrize("design", DESIGNS)
def test_js_matches_python(design: list[dict]) -> None:
    script = (
        "const c=require(process.argv[1]);const [m,d]=JSON.parse(process.argv[2]);"
        "console.log(JSON.stringify(c.evaluate(m,d)));"
    )
    out = subprocess.run(
        ["node", "-e", script, str(ROOT / "xr_latency/report/calc.js"), json.dumps([MODEL, design])],
        capture_output=True, text=True, check=True,
    ).stdout
    js, py = json.loads(out), evaluate(MODEL, design)
    assert js["first_audio_p50"] == pytest.approx(py.first_audio_p50)
    assert js["done_p50"] == pytest.approx(py.done_p50)
    assert js["done_p90"] == pytest.approx(py.done_p90)
    assert [s["end_p50"] for s in js["stages"]] == pytest.approx([s.end_p50 for s in py.stages])
