# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory ring-buffer lifecycle regressions."""
from __future__ import annotations

import uuid

from xr_ai_hub import ShmRingBuffer


def test_unlink_tolerates_already_removed_segment():
    ring = ShmRingBuffer(
        name=f"xr_test_{uuid.uuid4().hex[:12]}",
        num_slots=1,
        max_frame_bytes=64,
        create=True,
    )
    try:
        ring.unlink()
        ring.unlink()
    finally:
        ring.close()
