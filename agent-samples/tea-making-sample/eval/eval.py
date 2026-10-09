# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model routing eval for idle and active tea-guide turns."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
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
from tea_making_worker.workflow import GuidanceAgent, _state_contract
from tea_making_worker.workflow_tools import workflow_commit_tool
from xr_ai_models import ChatMessage, LLMService, load_models_config, make_llm
from xr_ai_tools import ToolSet
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.tool_calling import tool_definitions

_SAMPLE = Path(__file__).resolve().parents[1]
_MIN_PASS_RATE = 0.80
_CLASSIFIER_KINDS = {"change_watch", "transcript_summary", "video_delta"}


class _MeasuredLLM:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    async def chat(self, messages: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        response = await self._llm.chat(messages, **kwargs)
        self.calls.append({
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "model": response.raw.get("model"),
            "tool_calls": [call.name for call in response.tool_calls or ()],
            "settings": {
                key: kwargs.get(key)
                for key in ("max_tokens", "temperature", "enable_thinking", "thinking_budget")
            },
            "prompt_sha256": hashlib.sha256(
                "\n".join(message.content for message in messages if message.role == "system").encode("utf-8")
            ).hexdigest(),
        })
        return response


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
        caption_prompt=config.change_watch_caption_prompt,
        event_prompt=config.change_watch_event_prompt,
        default_instruction=config.change_watch_default_instruction,
        interval_s=config.change_watch_interval_s,
    )
    transcript = TranscriptAgent(
        llm=llm,
        summary_prompt=config.transcript_summary_prompt,
        summary_interval_s=config.transcript_summary_interval_s,
    )
    video_log = VideoLogAgent(
        images=images,
        vlm=placeholder,
        llm=llm,
        caption_prompt=config.video_caption_prompt,
        delta_prompt=config.video_delta_prompt,
        interval_s=config.video_log_interval_s,
        history_size=config.background_history_size,
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


async def _classify(foreground: ForegroundAgent, case: dict[str, Any]) -> dict[str, Any]:
    """Run the owning agent's production classifier and return its typed fields."""

    payload = case["input"]
    kind = case["kind"]
    if kind == "change_watch":
        from tea_making_worker.change_watch import ChangeDecision

        state = SimpleNamespace(
            instruction=payload["watch_for"],
            captions=tuple(payload["previous"]),
        )
        result = await foreground._change_watch._decide(state, payload["current"])
        if not isinstance(result, ChangeDecision):
            raise TypeError(f"change-watch returned {type(result).__name__}")
        return result.model_dump()
    if kind == "transcript_summary":
        result = await foreground._transcript._generate_summary(tuple(payload["utterances"]))
        return result.model_dump()
    if kind == "video_delta":
        from collections import deque

        state = SimpleNamespace(captions=deque(payload["previous"], maxlen=5))
        result = await foreground._video_log._generate_delta(state, payload["current"])
        return result.model_dump()
    raise ValueError(f"unknown classifier case kind {kind!r}")


def _observation_turn(
    guidance: GuidanceAgent,
    case: dict[str, Any],
    participant_id: str,
) -> tuple[tuple[ChatMessage, ...], ToolSet]:
    session = guidance.store.get(participant_id)
    guidance.store.start(session)
    step = guidance.workflow.step(str(case["step"]))
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
    return (
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=request),
    ), tools


async def main() -> None:
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", help="Case names; omit to run all")
    args = parser.parse_args()
    wanted = set(args.cases)
    selected = [case for case in cases if not wanted or case["name"] in wanted]
    if not selected:
        raise SystemExit(f"unknown cases: {args.cases}")
    config_path = _SAMPLE / "yaml" / "models.local.json"
    llm = make_llm(load_models_config(config_path), "llm")
    measured_llm = _MeasuredLLM(llm)
    foreground, guidance = _build_agents(measured_llm)
    passed_count = 0
    prompt_hashes: set[str] = set()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_entry = config["models"]["llm"]
    model_profile = model_entry["adapter"].get("preset") or model_entry["adapter"].get("model_name")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    print(
        "RUN "
        + json.dumps(
            {
                "sample": "tea-making-sample",
                "model_profile": model_profile,
                "config_sha256": config_hash,
                "cases": len(selected),
                "sampling_settings": "logged_per_case",
            },
            sort_keys=True,
        )
    )
    try:
        for index, case in enumerate(selected):
            participant_id = f"tea-eval-{index}"
            observation_case = case.get("kind") == "observation"
            classifier_case = case.get("kind") in _CLASSIFIER_KINDS
            started = time.perf_counter()
            call_start = len(measured_llm.calls)
            classifier_fields: dict[str, Any] | None = None
            if observation_case:
                messages, tools = _observation_turn(guidance, case, participant_id)
                prior_tool = case.get("prior_tool")
                if prior_tool is not None:
                    from xr_ai_models import ToolCall

                    call_id = f"prior-{prior_tool['name']}"
                    messages = (*messages, ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=[ToolCall(
                            id=call_id,
                            name=str(prior_tool["name"]),
                            arguments=json.dumps(prior_tool.get("arguments", {}), separators=(",", ":")),
                        )],
                    ), ChatMessage(
                        role="tool",
                        content=json.dumps(prior_tool["result"], separators=(",", ":")),
                        tool_call_id=call_id,
                    ))
                response = None
            elif classifier_case:
                classifier_fields = await _classify(foreground, case)
                prompt = {
                    "change_watch": foreground._change_watch._event_prompt,
                    "transcript_summary": foreground._transcript._summary_prompt,
                    "video_delta": foreground._video_log._delta_prompt,
                }[str(case["kind"])]
                messages = (ChatMessage(role="system", content=prompt),)
                tools = None
                response = None
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
                response = None
            prompt_hash = hashlib.sha256(messages[0].content.encode("utf-8")).hexdigest()
            prompt_hashes.add(prompt_hash)
            if not classifier_case:
                response = await measured_llm.chat(
                    messages,
                    tools=tool_definitions(tools),
                    max_tokens=512,
                    temperature=0.0,
                    enable_thinking=False,
                )
            calls = response.tool_calls or [] if response is not None else []
            content = response.content or "" if response is not None else ""
            actual_tools = [call.name for call in calls]
            expected_tool = case["expected_tool"]
            expected_tools = [] if expected_tool is None else [expected_tool]
            if classifier_case:
                actual_tools = [
                    name
                    for call in measured_llm.calls[call_start:]
                    for name in call["tool_calls"]
                ]
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
                    if "expected_updates" in case and arguments.get("updates") != expected_updates:
                        errors.append(
                            f"observation updates were {arguments.get('updates')!r}, "
                            f"expected {expected_updates!r}"
                        )
                    expected_subset = case.get("expected_updates_containing", {})
                    actual_updates = arguments.get("updates")
                    if expected_subset and (
                        not isinstance(actual_updates, dict)
                        or any(actual_updates.get(key) != value for key, value in expected_subset.items())
                    ):
                        errors.append(
                            f"observation updates were {actual_updates!r}, "
                            f"expected at least {expected_subset!r}"
                        )
            if classifier_fields is not None:
                expected_arguments = case.get("expected_arguments", {})
                for key, value in expected_arguments.items():
                    if classifier_fields.get(key) != value:
                        errors.append(f"{key} was {classifier_fields.get(key)!r}, expected {value!r}")
                for key, pattern in case.get("expected_argument_patterns", {}).items():
                    if re.search(str(pattern), str(classifier_fields.get(key, ""))) is None:
                        errors.append(f"{key} did not match {pattern!r}: {classifier_fields.get(key)!r}")
                for key, pattern in case.get("forbidden_argument_patterns", {}).items():
                    if re.search(str(pattern), str(classifier_fields.get(key, ""))) is not None:
                        errors.append(f"{key} matched forbidden {pattern!r}: {classifier_fields.get(key)!r}")
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
            elapsed_ms = (time.perf_counter() - started) * 1000
            measured = measured_llm.calls[call_start:]
            model = next((call["model"] for call in reversed(measured) if call["model"]), None) or model_profile
            llm_ms = sum(call["elapsed_ms"] for call in measured)
            settings = sorted({json.dumps(call["settings"], sort_keys=True) for call in measured})
            print(
                f"{label} {case['name']}: tools={actual_tools!r} model={model!r} "
                f"prompt_sha256={prompt_hash[:12]} settings={settings} "
                f"llm_ms={llm_ms:.1f} elapsed_ms={elapsed_ms:.1f} "
                f"content={content!r}"
            )
            if passed:
                passed_count += 1
    finally:
        await llm.close()
    all_settings = sorted({json.dumps(call["settings"], sort_keys=True) for call in measured_llm.calls})
    print(
        f"RESULT {passed_count}/{len(selected)} cases passed; settings={all_settings} "
        f"prompt_hashes={sorted(prompt_hashes)}"
    )
    pass_rate = passed_count / len(selected)
    if pass_rate < _MIN_PASS_RATE:
        raise SystemExit(
            f"overall pass rate {pass_rate:.1%} is below {_MIN_PASS_RATE:.0%}"
        )


if __name__ == "__main__":
    asyncio.run(main())
