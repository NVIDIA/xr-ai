# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the shipped tea guide's visual checks on generated scenes.

Run each step independently so an early recognition failure cannot hide later
cases. Save image and prompt provenance, raw responses, and separate recognition
and evidence-gate scores. Use guide.py for conversational and execution coverage.
No model server processes are started or stopped here.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import yaml
from _tea_scenes import load_scenes, render_scene
from procedure_capture_guide_worker.guide.guides import select_guide
from xr_ai_models import load_models_config, make_vlm
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

_SAMPLE = Path(__file__).resolve().parents[1]
_PROMPT = _SAMPLE / "worker/procedure_capture_guide_worker/guide/prompts/guide_vision.txt"


def score(case: dict, evidence_pattern: str, text: str, available: bool) -> dict:
    if not available:
        # An unavailable model is not a correct rejection of a negative fixture.
        return {"recognition": None, "gate_correct": None, "accepted": None, "status": "unavailable"}
    accepted = re.fullmatch(evidence_pattern, text.strip()) is not None
    return {
        "recognition": re.fullmatch(case["answer"], text.strip(), re.IGNORECASE) is not None,
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
        "recognition_correct": sum(row["recognition"] is True for row in scored),
        "gate_correct": sum(row["gate_correct"] is True for row in scored),
        "false_accepts": sum(row["accepted"] is True and not row["expected_accept"] for row in scored),
        "false_rejects": sum(row["accepted"] is False and row["expected_accept"] for row in scored),
        "passed": bool(rows) and len(scored) == len(rows)
        and all(row["recognition"] and row["gate_correct"] for row in scored),
    }


def prepare(manifest_path: Path, output: Path) -> tuple:
    manifest = yaml.safe_load(manifest_path.read_text())
    scenes = load_scenes(manifest_path, manifest)
    guide = select_guide(manifest_path.parent / manifest["guides_dir"], manifest["guide"])
    cases = manifest["cases"]
    if not cases or len({case["scene"] for case in cases}) != len(cases):
        raise ValueError("visual cases must have unique scene names and must not be empty")
    for case in cases:
        step = guide.workflow.steps[case["step"]]
        if step.trigger.function != "current_view" or step.evidence is None:
            raise ValueError("visual cases require a current_view step with an evidence gate")
        if not isinstance(case["accept"], bool) or not isinstance(case["answer"], str):
            raise ValueError("cases require a boolean accept and a string answer regex")
        re.compile(case["answer"])
        if case["scene"] not in scenes or not re.fullmatch(r"[a-z0-9_]+", case["scene"]):
            raise ValueError("case scene must name a safe, defined scene")
    output.mkdir(parents=True, exist_ok=False)
    provenance = []
    for case in cases:
        name = case["scene"]
        path = output / f"{name}.png"
        image = render_scene(scenes[name])
        image.save(path)
        provenance.append({"scene": name, "file": path.name,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    sources = [manifest_path, manifest_path.parent / manifest["scenes_file"],
               Path(__file__), Path(__file__).with_name("_tea_scenes.py"), _PROMPT, guide.path]
    (output / "provenance.json").write_text(json.dumps({
        "images": provenance,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
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
    vlm = make_vlm(load_models_config(models_path), "vlm")
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
                reference = images.put((output / f'{case["scene"]}.png').read_bytes())
                before = time.monotonic()
                try:
                    async with asyncio.timeout(timeout_s):
                        result = await tool.execute(ImageQueryRequest(image=reference, query=question))
                    text, available = result.text, result.available
                except TimeoutError:
                    text, available = "Visual request timed out", False
                row = {
                    "scene": case["scene"], "step": case["step"], "trial": trial + 1,
                    "question": question, "answer": text, "available": available,
                    "expected_answer": case["answer"], "expected_accept": case["accept"],
                    "evidence_pattern": step.evidence.pattern,
                    "latency_s": time.monotonic() - before,
                    **score(case, step.evidence.pattern, text, available),
                }
                rows.append(row)
                with (output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                images.clear()
    finally:
        await vlm.close()
        images.clear()
    result = {
        **summarize(rows),
        "by_step": {step: summarize([row for row in rows if row["step"] == step])
                    for step in sorted({case["step"] for case in cases})},
        "guide_sha256": guide.sha256,
        "models_config_sha256": hashlib.sha256(models_path.read_bytes()).hexdigest(),
        "elapsed_s": time.monotonic() - started,
        "coverage": "Real configured VLM and production visual prompt; isolated single-frame evidence gates.",
        "not_scored": ["consecutive-frame enforcement", "conversation", "transitions", "timer", "STT", "TTS"],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("tea_visual_cases.yaml"))
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
