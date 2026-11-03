# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patterns, suites, the store and the stack loader."""

from __future__ import annotations

import json
from pathlib import Path

from xr_latency._stack import Stack
from xr_latency._store import RunStore, read_meta
from xr_latency.compare import compare
from xr_latency.patterns import load_patterns, price, price_all
from xr_latency.suites import BENCHES, SUITES

from tests.test_parity import MODEL

ROOT = Path(__file__).resolve().parents[1]


def _priceable_model() -> dict:
    """The parity model plus the router and VLM-streaming terms the patterns use."""
    return {**MODEL,
            "router": {"decide_curve": {"p50": {"intercept": 90.0, "slope": 0.05}, "p90": {"intercept": 120.0,
                                                                                            "slope": 0.06}}},
            "streaming": {"chars_per_token": 4.0},
            "vlm": {**MODEL["vlm"],
                    "chat_total": [{"res": "1280x720", "images": 1, "p50": 400.0, "p90": 500.0}],
                    "stream_first_text": [{"res": "1280x720", "images": 1, "p50": 300.0, "p90": 380.0}],
                    "decode_ms_per_token": {"p50": 19.0, "p90": 21.0}}}


def test_every_pattern_prices() -> None:
    results = price_all(_priceable_model())
    assert set(results) == set(load_patterns())
    assert not [n for n, r in results.items() if isinstance(r, str)], results


def test_batch_delivery_is_slower_than_streamed() -> None:
    patterns, model = load_patterns(), _priceable_model()
    batch = price(model, patterns["voice-qa-batch"]).first_audio_p50
    streamed = price(model, patterns["voice-qa-streamed"]).first_audio_p50
    assert batch > streamed


def test_thinking_override_adds_time() -> None:
    pattern, model = load_patterns()["voice-qa-streamed"], _priceable_model()
    assert price(model, pattern, thinking=True).first_audio_p50 > price(model, pattern, thinking=False).first_audio_p50


def test_unpriceable_pattern_reports_reason() -> None:
    model = {k: v for k, v in _priceable_model().items() if k != "stt"}
    assert str(price_all(model)["routed-qa"]).startswith("not priced")


def test_compare_names_both_machines() -> None:
    a = {**_priceable_model(), "machine": {"host": "a", "gpus": [{"name": "GPU X"}]}}
    b = {**_priceable_model(), "machine": {"host": "b", "gpus": []}}
    text = compare(a, b)
    assert "a: 1x GPU X" in text and "b: no GPU" in text and "1.00x" in text


def test_suites_reference_registered_benchmarks() -> None:
    assert {s.bench for steps in SUITES.values() for s in steps} <= set(BENCHES)


def test_store_meta_merges(tmp_path: Path) -> None:
    store = RunStore.create(tmp_path, "x")
    store.write_meta({"machine": {"host": "h"}})
    store.write_meta({"tokens_per_char": 0.3})
    assert read_meta(store.root) == {"machine": {"host": "h"}, "tokens_per_char": 0.3}


def test_stack_host_override(tmp_path: Path) -> None:
    raw = {"models": {"llm": {"adapter": {"preset": "nemotron_omni"}, "endpoint": {"base_url": "http://localhost:8108"}}},
           "services": {"speaker_stt": "http://127.0.0.1:8102"}}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(raw))
    stack = Stack.load(str(path), host="10.0.0.5")
    assert stack.endpoints() == {"llm": "http://10.0.0.5:8108"}
    assert stack.service_url("speaker_stt") == "http://10.0.0.5:8102"
