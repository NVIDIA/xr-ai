# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the eval harness's scoring primitives; a silent checker bug voids every tier score."""

from xr_render_demo_eval.subagents import _args_match


def test_exact_values_match():
    assert _args_match({"obj_id": "cone-7", "x": 1.0}, {"obj_id": "cone-7"})
    assert not _args_match({"obj_id": "cone-7"}, {"obj_id": "ring-1"})


def test_range_is_inclusive_at_both_ends():
    assert _args_match({"x": 0.5}, {"x": (0.5, 1.0)})
    assert _args_match({"x": 1.0}, {"x": (0.5, 1.0)})
    assert not _args_match({"x": 1.01}, {"x": (0.5, 1.0)})


def test_missing_key_fails():
    assert not _args_match({"x": 0.5}, {"y": (0.0, 1.0)})
