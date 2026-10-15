# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the procedure captioner using generated images with known contents.

Run this script independently of the sample to test captioning without a
connected camera. It creates images from caption_cases.yaml and sends them to
the configured VLM using the same prompt and captioning helpers as capture.

The script checks response format and saves the images, responses, and expected
observations for human review of visual accuracy. Refer to
docs/source/reference/procedure-capture-replay.md for usage and interpretation of results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from procedure_capture_replay_worker.capture.captioning import caption_query, parse_caption
from procedure_capture_replay_worker.config import load_config
from xr_ai_models import load_models_config, make_vlm
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import MultiImageQueryRequest, MultiImageQueryTool

_SAMPLE = Path(__file__).resolve().parents[1]


def render(objects: list, path: Path) -> None:
    """Make repeatable scenes whose visible facts are known without a recording."""
    image = Image.new("RGB", (512, 384), "white")
    draw = ImageDraw.Draw(image)
    for shape, color, *bounds in objects:
        if shape == "rectangle":
            draw.rectangle(bounds, fill=color)
        elif shape == "ellipse":
            draw.ellipse(bounds, fill=color)
        else:
            raise ValueError(f"unknown shape: {shape}")
    image.save(path)


async def evaluate(output: Path, config_path: Path) -> bool:
    """Run the visual controls and save images and VLM responses for review.

    Return whether every response satisfies the caption format. Expected visual
    facts are saved for inspection; they are not scored automatically.
    """
    # Never overwrite a previous evaluation's evidence.
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(config_path)
    models = load_models_config(config.models_config)
    vlm = make_vlm(models, "vlm")
    images = ImageRegistry()
    # Reuse production components so the evaluation tests the actual caption
    # contract rather than a separate, potentially easier prompt or parser.
    tool = MultiImageQueryTool(images=images, vlm=vlm, system_prompt=config.capture.caption_prompt)
    cases = yaml.safe_load(Path(__file__).with_name("caption_cases.yaml").read_text())["cases"]
    (output / "prompt.txt").write_text(config.capture.caption_prompt)
    passed = 0
    try:
        for case in cases:
            references = []
            for index, objects in enumerate(case["images"]):
                path = output / f'{case["id"]}-{index}.png'
                render(objects, path)
                references.append(images.put(path))
            # YAML image order is the comparison order: earlier, then current.
            query = caption_query(case.get("previous"))
            result = await tool.execute(MultiImageQueryRequest(images=references, query=query))
            record = {"case": case["id"], "query": query, "raw_response": result.text,
                      "available": result.available, "expected": case["expected"]}
            try:
                if not result.available:
                    raise ValueError("VLM unavailable")
                # This counts structural validity only. Expected visual facts
                # stay in the saved record for review, never in the model input.
                record["caption"] = parse_caption(result.text)
                passed += 1
            except ValueError as exc:
                record["error"] = str(exc)
            (output / f'{case["id"]}.json').write_text(json.dumps(record, indent=2) + "\n")
    finally:
        await vlm.close()
    print(f"Caption contract: {passed}/{len(cases)}. Inspect visual accuracy separately: {output}")
    return bool(cases) and passed == len(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for evaluation evidence")
    parser.add_argument("--config", type=Path, default=_SAMPLE / "yaml/worker.yaml")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(evaluate(args.output.resolve(), args.config)) else 1)
