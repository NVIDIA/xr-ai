# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate fresh answers without replaying controls or exposing discarded text."""

# Pytest discovers the shared fixtures imported below.
# ruff: noqa: F811
import asyncio
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from test_sop_guide import connected_engine, control, document, engine, observation, selected  # noqa: F401
from xr_ai_models import ChatResponse
from xr_ai_tools.types import EmptyRequest
from xr_ai_tools.vision import ImageQueryResult

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide"
sys.path.insert(0, str(_SAMPLE / "worker"))
from procedure_capture_guide_worker.guide.guiding import _ProgressChanged  # noqa: E402


async def complete_visual(engine):
    backend = engine._backend
    session = backend._sessions["alice"]
    image = io.BytesIO()
    Image.new("RGB", (16, 16)).save(image, format="PNG")
    backend._trigger = AsyncMock(side_effect=[
        replace(observation("YES", sequence), value=image.getvalue()) for sequence in (1, 2)
    ])
    backend._image_query.execute.side_effect = None
    backend._image_query.execute.return_value = ImageQueryResult(text="YES")
    for _ in range(2):
        session.next_tick = 0
        await backend._tick(session)
    assert session.state["base"]


async def test_read_only_status_does_not_bypass_progress_validation(engine):
    backend = engine._backend
    guidance = await backend.guidance("alice")
    await guidance.tools.get("workflow__status").execute(EmptyRequest())
    await complete_visual(engine)
    with pytest.raises(_ProgressChanged):
        await guidance.finish("It is not complete yet.")
    assert "Complete" in await guidance.finish("It is not complete yet.", refresh=False)


async def test_progress_during_answer_refreshes_context_and_discards_old_answer(engine):
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def chat(messages, **kwargs):
        context = json.loads(messages[1].content)
        seen.append(context)
        if len(seen) == 1:
            entered.set()
            await release.wait()
            return ChatResponse("STALE_INCOMPLETE_REPLY", None, None, "stop", {})
        assert context["state"]["base"] is True
        return ChatResponse("The parts are verified. Say next when ready.", None, None, "stop", {})

    engine._llm.chat.side_effect = chat
    pending = asyncio.create_task(engine._route("Is this step ready?", "alice"))
    await asyncio.wait_for(entered.wait(), 1)
    await complete_visual(engine)
    release.set()
    result = await asyncio.wait_for(pending, 2)
    assert result == "The parts are verified. Say next when ready."
    assert len(seen) == 2 and seen[0]["state"]["base"] is False
    assert engine._backend._sessions["alice"].step_id == "base"
    assert "guide changed" not in result


async def test_refresh_is_bounded_when_progress_changes_again(engine, document):
    document["state"]["checks"] = {"type": "integer", "initial": 0, "description": "Observed check count"}
    document["steps"][0]["writes"].append("checks")
    backend = engine._backend
    backend._guide = selected(document)
    backend._start("alice")
    session = backend._sessions["alice"]
    calls = 0

    async def chat(messages, **kwargs):
        nonlocal calls
        calls += 1
        assert calls <= 2
        accepted, _, _ = backend._commit(session, session.step, {"checks": calls}, "")
        assert accepted
        return ChatResponse("STALE_PROGRESS_REPLY", None, None, "stop", {})

    engine._llm.chat.side_effect = chat
    result = await engine._route("What are we doing?", "alice")
    assert calls == 2 and "current step: base" in result
    assert "STALE_PROGRESS" not in result and "guide changed" not in result


async def test_transition_during_answer_does_not_retry_old_question_or_control(engine):
    entered, release = asyncio.Event(), asyncio.Event()

    async def chat(*args, **kwargs):
        entered.set()
        await release.wait()
        return ChatResponse("Stale answer about the base", None, None, "stop", {})

    engine._llm.chat.side_effect = chat
    pending = asyncio.create_task(engine._route("Is this step ready?", "alice"))
    await asyncio.wait_for(entered.wait(), 1)
    await control(engine._backend, "advance", skip=True)
    release.set()
    assert "guide changed" in await asyncio.wait_for(pending, 2)
    engine._llm.chat.assert_awaited_once()
    assert engine._backend._sessions["alice"].step_id == "lid"


async def test_transition_between_refresh_attempts_does_not_retarget_question(engine, monkeypatch):
    backend = engine._backend
    original = backend.guidance
    advanced = False

    async def guidance(pid):
        nonlocal advanced
        snapshot = await original(pid)
        if backend._sessions[pid].state["base"] and not advanced:
            advanced = True
            tool = snapshot.tools.get("workflow__advance")
            await tool.execute(tool.request_model(skip=True))
            snapshot = await original(pid)
        return snapshot

    async def chat(*args, **kwargs):
        await complete_visual(engine)
        return ChatResponse("Stale answer about the base", None, None, "stop", {})

    monkeypatch.setattr(backend, "guidance", guidance)
    engine._llm.chat.side_effect = chat
    result = await engine._route("Is this step ready?", "alice")
    assert "guide changed" in result
    assert backend._sessions["alice"].step_id == "lid"
    engine._llm.chat.assert_awaited_once()
