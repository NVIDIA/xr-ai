# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model routing eval for idle and active tea-guide turns."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from tea_making_worker.background_context import BackgroundContextAgent
from tea_making_worker.change_watch import ChangeWatchAgent
from tea_making_worker.config import load_config
from tea_making_worker.foreground import ForegroundAgent
from tea_making_worker.spec import load_workflow
from tea_making_worker.transcript import TranscriptAgent
from tea_making_worker.video_log import VideoLogAgent
from tea_making_worker.workflow import GuidanceAgent
from xr_ai_models import ChatMessage, LLMService, load_models_config, make_llm
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.tool_calling import tool_definitions

_SAMPLE = Path(__file__).resolve().parents[1]


def _build_agents(llm: LLMService) -> tuple[ForegroundAgent, GuidanceAgent]:
    config = load_config(_SAMPLE / "yaml" / "tea_making_worker.yaml")
    images = SimpleNamespace(
        images=ImageRegistry(),
        get_current_frame=SimpleNamespace(),
    )
    placeholder = SimpleNamespace()
    guidance = GuidanceAgent(
        workflow=load_workflow(_SAMPLE / "yaml" / "workflow.yaml"),
        llm=llm,
        current_frame=images.get_current_frame,
        image_query=placeholder,
        rag=placeholder,
    )
    change_watch = ChangeWatchAgent(
        images=images,
        vlm=placeholder,
        llm=llm,
        caption_prompt="Caption the current view.",
        event_prompt="Compare the views.",
        default_instruction="important changes",
        interval_s=2.0,
    )
    transcript = TranscriptAgent(
        llm=llm,
        summary_prompt="Summarize the transcript.",
    )
    video_log = VideoLogAgent(
        images=images,
        vlm=placeholder,
        llm=llm,
        caption_prompt="Caption the current view.",
        delta_prompt="Compare the views.",
        interval_s=2.0,
    )
    foreground = ForegroundAgent(
        llm=llm,
        images=images,
        vlm=placeholder,
        rag=placeholder,
        guidance=guidance,
        background_context=BackgroundContextAgent(),
        change_watch=change_watch,
        transcript=transcript,
        video_log=video_log,
        prompt=config.foreground_prompt,
    )
    return foreground, guidance


def _active_route(
    guidance: GuidanceAgent,
    case: dict[str, Any],
    participant_id: str,
) -> None:
    target_step = str(case["step"])
    session = guidance.store.get(participant_id)
    guidance.store.start(session)
    while session.step_id != target_step:
        if not session.active or session.step_id is None:
            raise ValueError(f"cannot reach active step {target_step!r}")
        if session.step_id == "start_steeping" and target_step == "steep_timer":
            observation = "A tea bag is immersed in the water."
            guidance.store.observe(session, observation)
            guidance.store.observe(session, observation)
            result = guidance.store.commit(
                session,
                {"steeping_started_at_us": 1, "steeping_started": True},
                "",
            )
            if not result.accepted or not result.complete:
                raise ValueError("could not complete start_steeping for timer eval")
            guidance.store.advance(session, skip=False)
        else:
            guidance.store.advance(session, skip=True)
    state_updates = dict(case.get("state_updates", {}))
    if state_updates:
        for observation in case.get("observations", []):
            guidance.store.observe(session, observation)
        result = guidance.store.commit(session, state_updates, "")
        if not result.accepted:
            raise ValueError(f"state updates for {case['name']!r} were rejected: {state_updates!r}")
    if guidance.active_context(participant_id) is None:
        raise ValueError(f"case {case['name']!r} did not produce an active route")


def _normalize_response(text: str) -> str:
    return " ".join(text.split())


async def main() -> None:
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    llm = make_llm(load_models_config(_SAMPLE / "yaml" / "models.local.json"), "llm")
    foreground, guidance = _build_agents(llm)
    failures: list[str] = []
    try:
        for index, case in enumerate(cases):
            participant_id = f"tea-eval-{index}"
            if case.get("route", "root") == "active":
                _active_route(
                    guidance,
                    case,
                    participant_id,
                )
            system_prompt, tools, route = foreground._prepare_route(
                participant_id,
                ctx=None,
                timestamp_us=None,
            )
            expected_route = "tea" if case.get("route", "root") == "active" else "root"
            if route != expected_route:
                raise ValueError(
                    f"case {case['name']!r} prepared route {route!r}, expected {expected_route!r}"
                )
            response = await llm.chat(
                (
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=case["query"]),
                ),
                tools=tool_definitions(tools),
                max_tokens=512,
                temperature=0.0,
                enable_thinking=False,
            )
            calls = response.tool_calls or []
            actual_tools = [call.name for call in calls]
            expected_tool = case["expected_tool"]
            expected_tools = [] if expected_tool is None else [expected_tool]
            errors: list[str] = []
            for call in calls:
                tool = tools.get(call.name)
                if tool is None:
                    errors.append(f"unknown tool {call.name!r}")
                    continue
                try:
                    tool.request_model.model_validate_json(call.arguments)
                except ValueError as exc:
                    errors.append(f"invalid {call.name!r} arguments: {exc}")
            content = response.content or ""
            normalized_content = _normalize_response(content)
            expected_response = case.get("expected_response")
            if expected_response is not None and normalized_content != _normalize_response(
                str(expected_response)
            ):
                errors.append(f"response did not equal {expected_response!r}")
            expected_response_pattern = case.get("expected_response_pattern")
            if expected_response_pattern is not None and re.search(
                str(expected_response_pattern), normalized_content
            ) is None:
                errors.append(
                    f"response did not match {expected_response_pattern!r}"
                )
            forbidden_response = case.get("forbidden_response")
            if forbidden_response is not None and _normalize_response(
                str(forbidden_response)
            ) in normalized_content:
                errors.append(f"response contained forbidden {forbidden_response!r}")
            passed = actual_tools == expected_tools and not errors
            label = "PASS" if passed else "FAIL"
            print(f"{label} {case['name']}: tools={actual_tools!r} content={content!r}")
            if not passed:
                failures.append(
                    f"{case['name']}: expected {expected_tools!r}, received {actual_tools!r}; errors={errors!r}"
                )
    finally:
        await llm.close()
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
