# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model routing eval for idle and active tea-guide turns."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from tea_making_worker import SOPAgent
from tea_making_worker.background_context import BackgroundContextAgent
from tea_making_worker.change_watch import ChangeWatchAgent
from tea_making_worker.config import load_config
from tea_making_worker.conversation._foreground import ForegroundAgent, _bind_arguments
from tea_making_worker.conversation._routing import _make_router
from tea_making_worker.flows._observations import _observe
from tea_making_worker.sop import load_workflow
from tea_making_worker.sop._engine import GuidanceAgent
from tea_making_worker.transcript import TranscriptAgent
from tea_making_worker.video_log import VideoLogAgent
from xr_ai_models import ChatMessage, DecisionService, LLMService, load_models_config, make_decision, make_llm
from xr_ai_routing import AgentTurn
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.rag import RetrieveResult
from xr_ai_tools.vision import ImageQueryResult

_SAMPLE = Path(__file__).resolve().parents[1]
_MIN_PASS_RATE = 0.80


def _build_agents(llm: LLMService, decisions: DecisionService) -> tuple[ForegroundAgent, GuidanceAgent]:
    config = load_config(_SAMPLE / "yaml" / "tea_making_worker.yaml")
    images = SimpleNamespace(
        images=ImageRegistry(),
        get_current_frame=SimpleNamespace(),
    )
    placeholder = SimpleNamespace()
    change_watch = ChangeWatchAgent(
        decisions=decisions,
        decision=config.change_watch_decision,
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
        decisions=decisions,
        decision=config.video_delta_decision,
        images=images,
        vlm=placeholder,
        llm=llm,
        caption_prompt="Caption the current view.",
        delta_prompt="Compare the views.",
        interval_s=2.0,
    )
    foreground = SOPAgent(
        decisions=decisions,
        llm=llm,
        images=images,
        vlm=placeholder,
        rag=placeholder,
        # Routing cases hold each stage explicitly; automatic progression is
        # exercised separately by the procedure integration tests.
        workflow=replace(load_workflow(_SAMPLE / "yaml/workflow.yaml"), advancement="manual"),
        image_query=placeholder,
        application_tools=lambda pid: ToolSet(
            tool for provider in (BackgroundContextAgent(), change_watch, transcript, video_log)
            for _, tool in provider.participant_tools(pid).items()
        ),
        prompt=config.foreground_prompt,
        current_view_prompt=config.current_view_prompt,
        context_budget_bytes=config.context_budget_bytes,
        compaction_input_budget_bytes=config.compaction_input_budget_bytes,
    )
    question_tools = foreground._question_tools

    def fixture_tools(participant_id):
        tools = {}
        for name, tool in question_tools(participant_id).items():
            if name == "current_view":
                async def unavailable(_request):
                    return ImageQueryResult(text="No camera image is supplied by this routing eval.", available=False)
                tool = Tool(name, tool.description, tool.request_model, tool.result_model, unavailable)
            elif name == "rag_lookup":
                async def empty_reference(_request):
                    return RetrieveResult(results=[])
                tool = Tool(name, tool.description, tool.request_model, tool.result_model, empty_reference)
            tools[name] = tool
        return ToolSet(tools)

    foreground._question_tools = fixture_tools
    return foreground, foreground._guidance


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


class _RecordingDecisions:
    def __init__(self, service: DecisionService) -> None:
        self.service = service
        self.calls: list[dict[str, Any]] = []

    async def decide(self, state, questions, **kwargs):
        started = time.perf_counter()
        result = await self.service.decide(state, questions, **kwargs)
        self.calls.append(
            {
                "state": state,
                "questions": {
                    key: {"instructions": question.instructions, "criteria": dict(question.criteria)}
                    for key, question in questions.items()
                },
                "choices": {key: list(question.criteria) for key, question in questions.items()},
                "answers": {key: answer.choice for key, answer in result.answers.items()},
                "probabilities": {key: dict(answer.probabilities) for key, answer in result.answers.items()},
                "model": result.model,
                "model_revision": result.raw.get("model_revision"),
                "duration_ms": (time.perf_counter() - started) * 1000,
            }
        )
        return result


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=_SAMPLE / "eval/cases.yaml")
    parser.add_argument("--output", type=Path, help="Write per-case results and model-call timings as JSONL.")
    args = parser.parse_args()
    cases = yaml.safe_load(args.cases.read_text(encoding="utf-8"))
    models = load_models_config(_SAMPLE / "yaml/models.local.json")
    llm = make_llm(models, "llm")
    decision_service = make_decision(models, "decision")
    decisions = _RecordingDecisions(decision_service)
    foreground, guidance = _build_agents(llm, decisions)
    # The production admissions/router are exercised with dry-run tool effects.
    # Only setup executes the native start tool to establish actual router focus.
    await foreground._router.stop()
    captured: dict[str, dict[str, Any]] = {}

    async def execute(turn, action, started, *, workflow=False):
        await started()
        row = captured[turn.participant_id]
        row["workflow_scope"] = workflow
        if action.tool is not None:
            tools = (
                foreground._workflow_tools(turn.participant_id)
                if workflow
                else foreground._application_tools(turn.participant_id)
            )
            tool = tools.get(action.tool)
            session = guidance.store.find(turn.participant_id)
            state = dict(session.state) if session else {}
            arguments = _bind_arguments(action.arguments, turn.query, state)
            request = tool.request_model.model_validate(arguments)
            row["tools"].append(action.tool)
            row["arguments"] = arguments
            if turn.turn_id.endswith(":setup") and action.tool == "workflow__start":
                await tool.execute(request)
            elif action.tool == "workflow__status":
                row["content"] = (await tool.execute(request)).message
        else:
            row["content"], calls, _ = await foreground._generate_answer(
                turn.query,
                turn.participant_id,
                workflow_context=workflow,
            )
            row["tools"].extend(calls)
            blocks = foreground._run_contexts.blocks(turn.participant_id)
            row["execution"] = next((
                json.loads(block.text.split("\n", 1)[1])
                for block in blocks if block.text.startswith("Source sop.execution ")
            ), {})
        return row["content"]

    async def stale(turn, started):
        await started()
        captured[turn.participant_id]["content"] = guidance.workflow.foreground.stale_message

    async def output(turn, _route, text):
        captured[turn.participant_id]["content"] = text

    router = _make_router(
        guidance,
        decisions,
        guidance.workflow.foreground,
        foreground._workflow_tools,
        foreground._application_tools,
        execute,
        stale,
        output,
    )
    rows = []
    try:
        for index, case in enumerate(cases):
            participant_id = f"tea-eval-{index}"
            captured[participant_id] = {"tools": [], "content": ""}
            errors = []
            started = time.perf_counter()
            decisions.calls = []
            try:
                if case.get("kind") == "observation":
                    # Caption extraction fixtures do not qualify physical camera accuracy.
                    policy = guidance.workflow.step(case["step"]).observation
                    result = await llm.chat(
                        (
                            ChatMessage(
                                role="system",
                                content=policy.extraction_question("Extract only facts in this description."),
                            ),
                            ChatMessage(role="user", content=case["observation"]),
                        ),
                        max_tokens=256,
                        temperature=0.0,
                        enable_thinking=False,
                    )
                    observed = await _observe(
                        policy, result.content, {}, decisions=decisions, llm=llm, tools=ToolSet(()), timeout_s=15.0
                    )
                    if observed.evidence != "unknown":
                        errors.append("unreadable observation was not unknown")
                else:
                    if case.get("route") == "active":
                        setup = await router.handle(
                            AgentTurn(participant_id, f"{index}:setup", "Start the tea-making guide now.")
                        )
                        if setup.disposition != "focus":
                            raise ValueError("setup did not establish workflow focus")
                        _active_route(guidance, case, participant_id)
                        captured[participant_id] = {"tools": [], "content": ""}
                    decisions.calls = []
                    started = time.perf_counter()
                    await router.handle(AgentTurn(participant_id, str(index), case["query"]))
                    actual = captured[participant_id]
                    expected = [] if case["expected_tool"] is None else [case["expected_tool"]]
                    alternatives = case.get("accepted_tools", [expected])
                    if actual["tools"] not in alternatives:
                        errors.append(f"expected tools in {alternatives!r}")
                    if "expected_context_operation" in case:
                        operation = actual.get("execution", {}).get("operations", {}).get(
                            case["expected_context_operation"]
                        )
                        if not operation or operation.get("available") is not True:
                            errors.append("expected live operation context is unavailable")
                    if "expected_skip" in case and actual.get("arguments", {}).get("skip") is not case["expected_skip"]:
                        errors.append("incorrect bound skip argument")
                    if (
                        "expected_workflow_scope" in case
                        and actual.get("workflow_scope") is not case["expected_workflow_scope"]
                    ):
                        errors.append("incorrect destination scope")
                    content = _normalize_response(actual["content"])
                    if (
                        "expected_response_pattern" in case
                        and re.search(case["expected_response_pattern"], content) is None
                    ):
                        errors.append("response pattern mismatch")
                    if "forbidden_response" in case and _normalize_response(case["forbidden_response"]) in content:
                        errors.append("forbidden response")
                    if "max_response_chars" in case and len(content) > case["max_response_chars"]:
                        errors.append("response too long")
            except Exception as exc:  # noqa: BLE001 -- record a failed case and evaluate the remaining independent scenarios.
                errors.append(f"{type(exc).__name__}: {exc}")
            row = {
                "case": case["name"],
                **captured[participant_id],
                "passed": not errors,
                "errors": errors,
                "duration_ms": (time.perf_counter() - started) * 1000,
                "decisions": list(decisions.calls),
            }
            rows.append(row)
            print(f"{'PASS' if not errors else 'MISS'} {case['name']}: {row['tools']!r} {errors!r}", flush=True)
            await router.participant_left(participant_id)
    finally:
        await router.stop()
        await foreground.stop()
        await llm.close()
        await decision_service.close()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
            )
    passed = sum(row["passed"] for row in rows)
    print(f"RESULT {passed}/{len(cases)}")
    if passed / len(cases) < _MIN_PASS_RATE:
        raise SystemExit(f"overall pass rate is below {_MIN_PASS_RATE:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
