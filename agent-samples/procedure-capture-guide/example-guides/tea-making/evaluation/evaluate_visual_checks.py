# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a guide's visual checks against a manifest of generated image scenes.

Run each step independently so an early recognition failure cannot hide later
cases. Save image and prompt provenance, raw responses, and separate recognition
and evidence-gate scores. Use the adjacent run_scenario.py for conversational and execution coverage.
No model server processes are started or stopped here.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import yaml
from _scene_rendering import load_scenes, render_scene
from _visual_expectations import matches_answer, visual_cases
from procedure_capture_guide_worker.guide.guides import load_guide
from procedure_capture_guide_worker.guide.visual_check import VisualCheckTrace, verify_image
from xr_ai_models import load_models_config, make_decision, make_vlm
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryTool

_SAMPLE = Path(__file__).resolve().parents[3]
_PROMPT = _SAMPLE / "worker/procedure_capture_guide_worker/guide/prompts/guide_vision.txt"


def score(case: dict, accepted: bool | None, text: str | None, available: bool) -> dict:
    if not available or accepted is None:
        # An unavailable model is not a correct rejection of a negative fixture.
        return {
            "recognition": matches_answer(case["answer"], text) if available and text is not None else None,
            "gate_correct": None, "accepted": None, "status": "unavailable",
        }
    return {
        "recognition": matches_answer(case["answer"], text) if text is not None else None,
        "gate_correct": accepted == case["accept"],
        "accepted": accepted,
        "status": "scored",
    }


def summarize(rows: list[dict]) -> dict:
    scored = [row for row in rows if row["status"] == "scored"]
    return {
        "attempts": len(rows),
        "scored": len(scored),
        "unavailable": len(rows) - len(scored),
        "recognition_scored": sum(row.get("recognition") is not None for row in rows),
        "recognition_correct": sum(row.get("recognition") is True for row in rows),
        "gate_correct": sum(row["gate_correct"] is True for row in scored),
        "false_accepts": sum(row["accepted"] is True and not row["expected_accept"] for row in scored),
        "false_rejects": sum(row["accepted"] is False and row["expected_accept"] for row in scored),
        "passed": bool(rows) and len(scored) == len(rows)
        and all(row["recognition"] is not False and row["gate_correct"] for row in scored),
    }


def prepare(manifest_path: Path, output: Path) -> tuple:
    manifest = yaml.safe_load(manifest_path.read_text())
    scenes = load_scenes(manifest_path, manifest)
    guide = load_guide(manifest_path.parent / manifest["guide_path"])
    cases = list(visual_cases(manifest).values())
    for case in cases:
        step = guide.workflow.steps[case["step"]]
        if step.trigger.function != "current_view" or step.evidence is None:
            raise ValueError("visual cases require a current_view step with an evidence gate")
        if case["scene"] not in scenes:
            raise ValueError("case scene must name a safe, defined scene")
    sources = [manifest_path, Path(__file__), Path(__file__).with_name("_scene_rendering.py"),
               Path(__file__).with_name("_visual_expectations.py"), _PROMPT, guide.path,
               _PROMPT.with_name("guide_condition.txt"), _PROMPT.with_name("guide_image_condition.txt"),
               _PROMPT.parent.parent / "verification.py", _PROMPT.parent.parent / "visual_check.py"]
    if manifest.get("scenes_file") is not None:
        sources.append(manifest_path.parent / manifest["scenes_file"])
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    output.mkdir(parents=True, exist_ok=False)
    provenance = []
    for case in cases:
        name = case["scene"]
        path = output / f"{name}.png"
        image = render_scene(scenes[name])
        image.save(path)
        provenance.append({"scene": name, "file": path.name,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (output / "provenance.json").write_text(json.dumps({
        "images": provenance,
        "source_sha256": source_hashes,
        "scenes": scenes, "manifest": manifest, "guide_sha256": guide.sha256,
        "system_prompt": _PROMPT.read_text(),
        "origin": "Deterministically drawn schematics; no photographs or recording frames.",
    }, indent=2) + "\n")
    (output / "guide.yaml").write_bytes(guide.path.read_bytes())
    return cases, guide


async def evaluate(manifest_path: Path, output: Path, models_path: Path, repeats: int, timeout_s: float) -> dict:
    if repeats < 1 or timeout_s <= 0:
        raise ValueError("repeats and timeout must be positive")
    cases, guide = prepare(manifest_path, output)
    models = load_models_config(models_path)
    vlm = make_vlm(models, "vlm")
    decisions = make_decision(models, "decision")
    images = ImageRegistry()
    tool = ImageQueryTool(images=images, vlm=vlm, system_prompt=_PROMPT.read_text())
    rows = []
    started = time.monotonic()
    try:
        for trial in range(repeats):
            for case in cases:
                step = guide.workflow.steps[case["step"]]
                question = step.trigger.arguments["question"]
                # Opaque references keep fixture names and expected answers out of model inputs.
                image = (output / f'{case["scene"]}.png').read_bytes()
                before = time.monotonic()
                trace = VisualCheckTrace()
                accepted = None
                error = None
                try:
                    decision = await verify_image(
                        decisions, tool, images=images, image=image,
                        condition=step.evidence.condition, question=question,
                        frame={"scene": case["scene"], "trial": trial + 1},
                        step_id=step.id, revision=trial,
                        decision_timeout_s=timeout_s, vision_timeout_s=timeout_s, trace=trace,
                    )
                    accepted = decision.choice == "yes"
                except (TimeoutError, ValueError, KeyError, RuntimeError, httpx.HTTPError) as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
                text = trace.caption.text if trace.caption is not None else None
                available = trace.caption.available if trace.caption is not None else accepted is not None
                caption_failed = trace.source == "caption" and (
                    trace.caption is None or not trace.caption.available or not trace.caption.text.strip()
                )
                row = {
                    "scene": case["scene"], "step": case["step"], "trial": trial + 1,
                    "question": question, "answer": text, "available": available,
                    "vlm_error": error if caption_failed else None,
                    "decision_error": error if not caption_failed else None,
                    "source": trace.source,
                    "image_decision": asdict(trace.image_answer) if trace.image_answer is not None else None,
                    "final_decision": asdict(trace.answer) if trace.answer is not None else None,
                    "expected_answer": case["answer"], "expected_accept": case["accept"],
                    "condition": step.evidence.condition,
                    "latency_s": time.monotonic() - before,
                    **score(case, accepted, text, available),
                }
                rows.append(row)
                with (output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                images.clear()
    finally:
        await vlm.close()
        await decisions.close()
        images.clear()
    result = {
        **summarize(rows),
        "by_step": {step: summarize([row for row in rows if row["step"] == step])
                    for step in sorted({case["step"] for case in cases})},
        "guide_sha256": guide.sha256,
        "models_config_sha256": hashlib.sha256(models_path.read_bytes()).hexdigest(),
        "elapsed_s": time.monotonic() - started,
        "coverage": "Production image-first CLEF with Cosmos caption fallback; isolated single-frame verification.",
        "image_only": sum(row["source"] == "image" and row["status"] == "scored" for row in rows),
        "caption_fallbacks": sum(row["source"] == "caption" for row in rows),
        "not_scored": ["consecutive-frame enforcement", "conversation", "transitions", "timer", "STT", "TTS"],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Visual expectations and image-scene definitions")
    parser.add_argument("--models", type=Path, default=_SAMPLE / "yaml/models.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument("--render-only", action="store_true", help="Inspect generated fixtures without calling models")
    args = parser.parse_args()
    if args.render_only:
        prepare(args.manifest.resolve(), args.output.resolve())
    else:
        result = asyncio.run(evaluate(args.manifest.resolve(), args.output.resolve(),
                                      args.models, args.repeats, args.timeout_s))
        raise SystemExit(0 if result["passed"] else 1)
