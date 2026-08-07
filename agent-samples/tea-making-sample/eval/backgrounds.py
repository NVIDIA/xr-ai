# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run production background-agent prompts against the configured model."""

import argparse
import asyncio
import json
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import yaml
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict
from xr_ai_models import load_models_config, make_llm
from xr_ai_nat.functions.vision import LiveVisionResult

_SAMPLE = Path(__file__).parents[1]
sys.path.insert(0, str(_SAMPLE / "worker"))

from tea_making_worker.agents.factory import add_guidance_llm  # noqa: E402
from tea_making_worker.applications.change_watch import ChangeWatchApplication  # noqa: E402
from tea_making_worker.applications.transcript import TranscriptApplication  # noqa: E402
from tea_making_worker.applications.video_log import VideoLogApplication  # noqa: E402
from tea_making_worker.desktop.runtime import DesktopRuntime  # noqa: E402
from tea_making_worker.desktop.spec import DesktopSpec, load_desktop  # noqa: E402
from tea_making_worker.runtime.scope import current_invocation, invocation_scope  # noqa: E402
from tea_making_worker.runtime.state import SessionStore  # noqa: E402
from tea_making_worker.spec import load_workflow  # noqa: E402


class ViewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str


class ViewConfig(FunctionBaseConfig, name="voice_desktop_background_eval_view"):
    pass


@register_function(config_type=ViewConfig)
async def eval_view(config: ViewConfig, _builder: Builder):
    async def view(request: ViewRequest) -> LiveVisionResult:
        return LiveVisionResult(answer="")

    yield FunctionInfo.from_fn(view, description="Unused vision stub for background prompt evaluation.")


async def evaluate(models_path: Path, cases_path: Path) -> int:
    workflow = load_workflow(_SAMPLE / "yaml" / "workflow.yaml")
    desktop = load_desktop(_SAMPLE / "yaml" / "applications.yaml")
    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))["background_applications"]
    llm = make_llm(load_models_config(models_path), "agent_llm")
    outputs: list[tuple[str, str, str]] = []

    async def output(participant_id: str, label: str, message: str) -> None:
        outputs.append((participant_id, label, message))

    failures = 0
    try:
        with tempfile.TemporaryDirectory(prefix="voice-desktop-eval-") as directory:
            transcript_spec = desktop.application("transcript")
            transcript_spec = replace(
                transcript_spec,
                settings={
                    **transcript_spec.settings,
                    "output_dir": Path(directory),
                    "summary_interval_s": 120,
                },
            )
            video_spec = desktop.application("video_log")
            video_spec = replace(
                video_spec,
                settings={**video_spec.settings, "output_dir": Path(directory)},
            )
            eval_desktop = DesktopSpec(
                root_prompt=desktop.root_prompt,
                capabilities=desktop.capabilities,
                applications={
                    **desktop.applications,
                    "transcript": transcript_spec,
                    "video_log": video_spec,
                },
            )
            runtime = DesktopRuntime(eval_desktop)
            change = ChangeWatchApplication(eval_desktop.application("change_watch"), runtime, output)
            transcript = TranscriptApplication(transcript_spec, runtime, output)
            video = VideoLogApplication(video_spec, runtime)
            store = SessionStore(workflow)
            async with WorkflowBuilder() as builder:
                view = await builder.add_function("background_eval_view", ViewConfig())
                llm_ref = await add_guidance_llm(builder, llm)
                await change.build(builder, llm_ref, view)
                await transcript.build(builder, llm_ref)
                await video.build(builder, llm_ref, view)

                for name, case in cases["change_watch"].items():
                    participant_id = f"change-{name}"
                    session = store.get(participant_id)
                    await change.start(session, str(case["instruction"]))
                    state = change._states[participant_id]
                    state.captions.extend(map(str, case["previous"]))
                    caption = str(case["current"])
                    before = len(outputs)
                    trace_id = f"change-{name}"
                    if change._agent is None:
                        raise RuntimeError("change-watch agent was not built")
                    with invocation_scope(session, trace_id):
                        current_invocation().context["change_watch.caption"] = caption
                        payload = json.dumps(
                            {
                                "watch_for": state.instruction,
                                "previous": list(state.captions),
                                "current": caption,
                            },
                            separators=(",", ":"),
                        )
                        await change._agent.ainvoke(payload, to_type=str)
                    emitted = len(outputs) > before
                    labeled = not emitted or outputs[-1][1] == change.spec.title
                    passed = state.captions[-1] == caption and emitted is case["expected_notice"]
                    passed = passed and labeled
                    failures += not passed
                    print(f"{'PASS' if passed else 'FAIL'} change_watch.{name}")

                transcript_case = cases["transcript"]
                session = store.get("transcript")
                await transcript.start(session)
                for index, utterance in enumerate(transcript_case["utterances"]):
                    await transcript.on_transcription(session, str(utterance), f"transcript-{index}")
                transcript._states[session.participant_id].next_summary = 0
                before = len(outputs)
                await transcript.tick(session)
                summaries = [
                    message
                    for participant, label, message in outputs[before:]
                    if participant == "transcript" and label == transcript.spec.title
                ]
                sentence_count = (
                    len([part for part in re.split(r"[.!?]+", summaries[0]) if part.strip()])
                    if summaries
                    else 0
                )
                passed = bool(summaries) is transcript_case["expected_output"]
                passed = passed and 1 <= sentence_count <= int(transcript_case["max_sentences"])
                failures += not passed
                print(f"{'PASS' if passed else 'FAIL'} transcript.summary")

                video_case = cases["video_log"]
                session = store.get("video-log")
                await video.start(session)
                state = video._states[session.participant_id]
                state.captions.extend(map(str, video_case["previous"]))
                caption = str(video_case["current"])
                if video._agent is None:
                    raise RuntimeError("video-log agent was not built")
                with invocation_scope(session, "video-log"):
                    current_invocation().context.update(
                        {"video_log.state": state, "video_log.caption": caption}
                    )
                    payload = json.dumps(
                        {
                            "previous": list(state.captions)[-(video.history_size - 1):],
                            "current": caption,
                        },
                        separators=(",", ":"),
                    )
                    await video._agent.ainvoke(payload, to_type=str)
                delta = str(state.writes[-1]["delta"])
                passed = state.captions[-1] == caption and all(
                    str(term).lower() in delta.lower()
                    for term in video_case["expected_delta_terms"]
                )
                failures += not passed
                print(f"{'PASS' if passed else 'FAIL'} video_log.delta")
    finally:
        await llm.close()
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=_SAMPLE / "eval" / "cases.yaml")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(evaluate(args.models, args.cases)))


if __name__ == "__main__":
    main()
