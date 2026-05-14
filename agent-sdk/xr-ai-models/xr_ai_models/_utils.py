# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers shared between config and openai_compat."""
from __future__ import annotations

from typing import Any, Iterable


def merge_dicts(
    base: dict[str, Any],
    overlay: dict[str, Any],
    *,
    skip_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Shallow top-level merge with one-level nested dict merging.

    Neither input is mutated.  Nested dicts (e.g. ``chat_template_kwargs``)
    are merged by key; non-dict values from ``overlay`` replace ``base``.
    """
    skip = set(skip_keys)
    out = dict(base)
    for k, v in overlay.items():
        if k in skip:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out
