# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol-completion checks for typed VLM streaming tests."""
from __future__ import annotations

import json
from contextlib import contextmanager

import httpx


@contextmanager
def require_complete_vlm_stream(monkeypatch):
    """Require a terminal OpenAI event and reject streamed errors."""
    saw_done = False
    original_lines = httpx.Response.aiter_lines

    async def checked_lines(response):
        nonlocal saw_done
        async for line in original_lines(response):
            if line.startswith("data:"):
                event = line.removeprefix("data:").strip()
                if event == "[DONE]":
                    saw_done = True
                else:
                    data = json.loads(event)
                    assert "error" not in data, f"VLM stream error: {data['error']}"
            yield line

    with monkeypatch.context() as patch:
        patch.setattr(httpx.Response, "aiter_lines", checked_lines)
        yield
    assert saw_done, "stream ended without the OpenAI [DONE] event"
