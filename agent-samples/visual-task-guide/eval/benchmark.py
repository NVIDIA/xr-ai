# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the live-caption and guide prompts against deployed models."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from nat.builder.workflow_builder import WorkflowBuilder
from visual_task_guide_worker.agent import TaskGuideAgentConfig
from visual_task_guide_worker.models import GuideAgentRequest
from visual_task_guide_worker.task_functions import (
    TaskKnowledgeFunctionsConfig,
    TaskStateFunctionsConfig,
)
from visual_task_guide_worker.task_store import TaskStore
from xr_ai_models import load_models_config, make_llm, make_vlm
from xr_ai_nat.llm import ModelsLLMConfig

try:
    from .cases import GUIDE_CASES, LEAKAGE_MARKERS, VLM_CASES
except ImportError:
    from cases import GUIDE_CASES, LEAKAGE_MARKERS, VLM_CASES

_HERE = Path(__file__).resolve().parent
_SAMPLE = _HERE.parent
_CAPTION_PROMPT = _SAMPLE / "worker/visual_task_guide_worker/prompts/caption.txt"
_GUIDE_PROMPT = _SAMPLE / "worker/visual_task_guide_worker/prompts/guide_agent.txt"
_FIXTURES = _HERE / "fixtures"


class LoggingTaskStore(TaskStore):
    def __init__(self, task_directory: Path) -> None:
        super().__init__(task_directory)
        self.search_calls: list[str] = []

    def search(self, query: str, *, limit: int):
        self.search_calls.append(query)
        return super().search(query, limit=limit)


def audit_fixture_leakage() -> None:
    prompts = f"{_CAPTION_PROMPT.read_text()} {_GUIDE_PROMPT.read_text()}".casefold()
    leaked = [marker for marker in LEAKAGE_MARKERS if marker.casefold() in prompts]
    if leaked:
        raise ValueError(f"eval fixture details leaked into prompts: {leaked}")


async def run_eval(models_path: Path, selected: set[str] | None = None) -> dict[str, Any]:
    audit_fixture_leakage()
    models = load_models_config(models_path)
    llm = make_llm(models, "guide_llm")
    vlm = make_vlm(models, "vlm")
    results: list[dict[str, Any]] = []
    store = LoggingTaskStore(_SAMPLE / "tasks/hand-counting")
    try:
        store.start("eval-user")

        for case in VLM_CASES:
            if selected and case["name"] not in selected:
                continue
            try:
                response = await vlm.ask_image(
                    _FIXTURES / case["fixture"],
                    case["question"],
                    system_prompt=_CAPTION_PROMPT.read_text(encoding="utf-8").strip(),
                    max_tokens=40,
                    temperature=0.0,
                )
                text = (response.content or "").strip().casefold()
                passed = all(term in text for term in case["required_terms"])
            except Exception as error:
                text, passed = f"{type(error).__name__}: {error}", False
            results.append(
                {"stage": "live_caption", "name": case["name"], "passed": passed, "output": text}
            )

        async with WorkflowBuilder() as builder:
            await builder.add_llm(
                "guide_llm",
                ModelsLLMConfig(
                    service=llm,
                    model_name="visual-task-guide-eval",
                    temperature=0.0,
                    max_tokens=128,
                ),
            )
            await builder.add_function_group("task_state", TaskStateFunctionsConfig(store=store))
            await builder.add_function_group("task_knowledge", TaskKnowledgeFunctionsConfig(store=store))
            guide = await builder.add_function("task_guide_agent", TaskGuideAgentConfig())

            for index, case in enumerate(GUIDE_CASES):
                if selected and case["name"] not in selected:
                    continue
                before_revision = store.progress("eval-user").revision
                before_searches = len(store.search_calls)
                try:
                    reply = await guide.ainvoke(
                        GuideAgentRequest(
                            participant_id="eval-user",
                            user_text=case["question"],
                            latest_observation=case["observation"],
                        )
                    )
                    text = reply.response.casefold()
                    searched = len(store.search_calls) > before_searches
                    passed = (
                        all(term in text for term in case["required_terms"])
                        and len(reply.response.split()) <= case["max_words"]
                        and (searched or not case["requires_knowledge"])
                        and store.progress("eval-user").revision == before_revision
                    )
                    output: Any = reply.response
                except Exception as error:
                    passed = False
                    output = f"{type(error).__name__}: {error}"
                results.append({"stage": "guide", "name": case["name"], "passed": passed, "output": output})

    finally:
        await llm.close()
        await vlm.close()
    return {
        "profile": str(models_path),
        "passed": all(item["passed"] for item in results),
        "results": results,
    }


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=Path,
        default=_SAMPLE / "yaml/models.local.json",
        help="xr-ai-models deployment profile with reachable guide_llm and vlm endpoints.",
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(run_eval(args.models.resolve(), set(args.case) or None))
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
