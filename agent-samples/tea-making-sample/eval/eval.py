# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model routing eval for idle and active tea-guide turns."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from tea_making_worker.background_context import BackgroundContextAgent
from tea_making_worker.change_watch import ChangeDecision, ChangeWatchAgent
from tea_making_worker.config import load_config
from tea_making_worker.foreground import ForegroundAgent
from tea_making_worker.spec import load_workflow
from tea_making_worker.transcript import TranscriptAgent, TranscriptSummary
from tea_making_worker.video_log import VideoDelta, VideoLogAgent
from tea_making_worker.workflow import GuidanceAgent, _state_contract
from tea_making_worker.workflow_tools import workflow_commit_tool
from xr_ai_models import (
    ChatMessage,
    LLMService,
    ToolCall,
    load_models_config,
    make_llm,
)
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.tool_calling import tool_definitions

_SAMPLE = Path(__file__).resolve().parents[1]
_PROMPTS = _SAMPLE / "worker" / "tea_making_worker" / "prompts"
_MIN_PASS_RATE = 0.80
_CLASSIFIER_CASES = {
    "change_watch": (
        "change_watch_event_prompt.txt",
        "change_watch__commit",
        ChangeDecision,
    ),
    "transcript_summary": (
        "transcript_summary_prompt.txt",
        "transcript__commit_summary",
        TranscriptSummary,
    ),
    "video_delta": (
        "video_delta_prompt.txt",
        "video_log__commit",
        VideoDelta,
    ),
}


def _models_config() -> Path:
    override = os.environ.get("XR_AI_EVAL_MODELS_CONFIG")
    return Path(override) if override else _SAMPLE / "yaml" / "models.local.json"


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
            guidance.store.observe(session, "accepted")
            guidance.store.observe(session, "accepted")
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
        for _observation in case.get("observations", []):
            guidance.store.observe(session, "accepted")
        result = guidance.store.commit(session, state_updates, "")
        if not result.accepted:
            raise ValueError(f"state updates for {case['name']!r} were rejected: {state_updates!r}")
    if guidance.active_context(participant_id) is None:
        raise ValueError(f"case {case['name']!r} did not produce an active route")


def _normalize_response(text: str) -> str:
    return " ".join(text.split())


def _observation_turn(
    guidance: GuidanceAgent,
    case: dict[str, Any],
    participant_id: str,
) -> tuple[tuple[ChatMessage, ...], ToolSet]:
    session = guidance.store.get(participant_id)
    guidance.store.start(session)
    target_step = str(case["step"])
    while session.step_id != target_step:
        if not session.active or session.step_id is None:
            raise ValueError(f"cannot reach observation step {target_step!r}")
        if session.step_id == "start_steeping" and target_step == "steep_timer":
            guidance.store.observe(session, "accepted")
            guidance.store.observe(session, "accepted")
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
    step = guidance.workflow.step(target_step)
    quick = guidance._named_tools(session, step.agent.tools)
    commit = workflow_commit_tool(
        guidance.store,
        session,
        expected_step_id=step.id,
        expected_revision=session.revision,
    )
    tools = ToolSet({commit.name: commit, **dict(quick.items())})
    request = json.dumps(
        {
            "observation": case["observation"],
            "already_complete": False,
            "state": guidance.workflow.project(step, session.state),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    system = "\n".join(
        (
            guidance._observation_prompt,
            _state_contract(guidance.workflow, step),
            step.agent.prompt,
        )
    )
    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=request),
    ]
    prior_tool = case.get("prior_tool")
    if prior_tool is not None:
        name = str(prior_tool["name"])
        call_id = f"prior-{name}"
        messages.extend(
            (
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            name=name,
                            arguments=json.dumps(
                                prior_tool.get("arguments", {}),
                                separators=(",", ":"),
                            ),
                        )
                    ],
                ),
                ChatMessage(
                    role="tool",
                    content=json.dumps(
                        prior_tool["result"],
                        separators=(",", ":"),
                    ),
                    tool_call_id=call_id,
                ),
            )
        )
    return tuple(messages), tools


def _classifier_turn(case: dict[str, Any]) -> tuple[tuple[ChatMessage, ...], ToolSet]:
    prompt_name, tool_name, request_model = _CLASSIFIER_CASES[str(case["kind"])]

    async def commit(request: Any) -> Any:
        return request

    tool = Tool(
        tool_name,
        "Commit the prompt-controlled result exactly once.",
        request_model,
        request_model,
        commit,
        return_direct=True,
    )
    messages = (
        ChatMessage(
            role="system",
            content=(_PROMPTS / prompt_name).read_text(encoding="utf-8").strip(),
        ),
        ChatMessage(
            role="user",
            content=json.dumps(case["input"], ensure_ascii=False, separators=(",", ":")),
        ),
    )
    return messages, ToolSet((tool,))


async def main() -> None:
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", help="Case names; omit to run all")
    args = parser.parse_args()
    wanted = set(args.cases)
    cases = [case for case in cases if not wanted or case["name"] in wanted]
    if not cases:
        raise SystemExit(f"unknown cases: {args.cases}")
    llm = make_llm(load_models_config(_models_config()), "llm")
    foreground, guidance = _build_agents(llm)
    passed_count = 0
    try:
        for index, case in enumerate(cases):
            participant_id = f"tea-eval-{index}"
            kind = case.get("kind")
            observation_case = kind == "observation"
            classifier_case = kind in _CLASSIFIER_CASES
            if observation_case:
                messages, tools = _observation_turn(guidance, case, participant_id)
            elif classifier_case:
                messages, tools = _classifier_turn(case)
            else:
                if case.get("route", "root") == "active":
                    _active_route(
                        guidance,
                        case,
                        participant_id,
                    )
                turn = foreground._prepare_turn(
                    participant_id,
                    query=case["query"],
                    ctx=None,
                    timestamp_us=None,
                )
                expected_route = (
                    "tea" if case.get("route", "root") == "active" else "root"
                )
                if turn.route != expected_route:
                    raise ValueError(
                        f"case {case['name']!r} prepared route {turn.route!r}, expected {expected_route!r}"
                    )
                messages = (
                    ChatMessage(role="system", content=turn.agent.system_prompt),
                    ChatMessage(role="user", content=turn.user_message),
                )
                tools = turn.tools
            response = await llm.chat(
                messages,
                tools=tool_definitions(tools),
                max_tokens=512,
                temperature=0.0,
                enable_thinking=False,
            )
            calls = response.tool_calls or []
            content = response.content or ""
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
            expected_skip = case.get("expected_skip")
            if expected_skip is not None and calls:
                try:
                    arguments = json.loads(calls[0].arguments)
                except json.JSONDecodeError:
                    pass
                else:
                    if arguments.get("skip") is not bool(expected_skip):
                        errors.append(
                            f"advance skip was {arguments.get('skip')!r}, "
                            f"expected {bool(expected_skip)!r}"
                        )
            if observation_case and calls:
                try:
                    arguments = json.loads(calls[0].arguments)
                except json.JSONDecodeError:
                    pass
                else:
                    expected_updates = case.get("expected_updates")
                    if (
                        "expected_updates" in case
                        and arguments.get("updates") != expected_updates
                    ):
                        errors.append(
                            f"observation updates were {arguments.get('updates')!r}, "
                            f"expected {expected_updates!r}"
                        )
                    expected_updates_containing = case.get(
                        "expected_updates_containing"
                    )
                    actual_updates = arguments.get("updates")
                    if expected_updates_containing is not None and (
                        not isinstance(actual_updates, dict)
                        or any(
                            actual_updates.get(name) != value
                            for name, value in expected_updates_containing.items()
                        )
                    ):
                        errors.append(
                            f"observation updates were {actual_updates!r}, expected "
                            f"at least {expected_updates_containing!r}"
                        )
            expected_arguments = case.get("expected_arguments")
            if expected_arguments is not None and calls:
                try:
                    arguments = json.loads(calls[0].arguments)
                except json.JSONDecodeError:
                    pass
                else:
                    if arguments != expected_arguments:
                        errors.append(
                            f"arguments were {arguments!r}, expected {expected_arguments!r}"
                        )
            expected_argument_patterns = case.get("expected_argument_patterns", {})
            forbidden_argument_patterns = case.get("forbidden_argument_patterns", {})
            if (expected_argument_patterns or forbidden_argument_patterns) and calls:
                try:
                    arguments = json.loads(calls[0].arguments)
                except json.JSONDecodeError:
                    pass
                else:
                    for field, pattern in expected_argument_patterns.items():
                        value = str(arguments.get(field, ""))
                        if re.search(str(pattern), value) is None:
                            errors.append(
                                f"argument {field!r}={value!r} did not match {pattern!r}"
                            )
                    for field, pattern in forbidden_argument_patterns.items():
                        value = str(arguments.get(field, ""))
                        if re.search(str(pattern), value) is not None:
                            errors.append(
                                f"argument {field!r}={value!r} matched forbidden {pattern!r}"
                            )
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
            max_response_chars = case.get("max_response_chars")
            if max_response_chars is not None and len(normalized_content) > int(
                max_response_chars
            ):
                errors.append(
                    f"response exceeded {max_response_chars} characters: "
                    f"{len(normalized_content)}"
                )
            forbidden_response = case.get("forbidden_response")
            if forbidden_response is not None and _normalize_response(
                str(forbidden_response)
            ) in normalized_content:
                errors.append(f"response contained forbidden {forbidden_response!r}")
            passed = actual_tools == expected_tools and not errors
            label = "PASS" if passed else "MISS"
            print(f"{label} {case['name']}: tools={actual_tools!r} content={content!r}")
            for error in errors:
                print(f"  {error}")
            if passed:
                passed_count += 1
    finally:
        await llm.close()
    print(f"RESULT {passed_count}/{len(cases)} cases passed")
    pass_rate = passed_count / len(cases)
    if pass_rate < _MIN_PASS_RATE:
        raise SystemExit(
            f"overall pass rate {pass_rate:.1%} is below {_MIN_PASS_RATE:.0%}"
        )


if __name__ == "__main__":
    asyncio.run(main())
