# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run caption controls through the configured VLM and production request/parser.

Contract checks are automatic. Visual correctness and grouping require inspection
of the saved images, raw responses, and case expectations; JSON success alone
does not measure perception accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from sop_sample_worker.config import load_config
from sop_sample_worker.recorder import _caption_query, _json_object
from xr_ai_models import load_models_config, make_vlm
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import MultiImageQueryRequest, MultiImageQueryTool

_SAMPLE = Path(__file__).resolve().parents[1]


def render(objects: list, path: Path) -> None:
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


async def evaluate(output: Path, config_path: Path) -> None:
    # Never overwrite a previous evaluation's evidence.
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(config_path)
    models = load_models_config(config.models_config)
    vlm = make_vlm(models, "vlm")
    images = ImageRegistry()
    tool = MultiImageQueryTool(images=images, vlm=vlm, system_prompt=config.caption_prompt)
    cases = yaml.safe_load(Path(__file__).with_name("caption_cases.yaml").read_text())["cases"]
    (output / "prompt.txt").write_text(config.caption_prompt)
    passed = 0
    try:
        for case in cases:
            references = []
            for index, objects in enumerate(case["images"]):
                path = output / f'{case["id"]}-{index}.png'
                render(objects, path)
                references.append(images.put(path))
            query = _caption_query(case.get("previous"))
            result = await tool.execute(MultiImageQueryRequest(images=references, query=query))
            record = {"case": case["id"], "query": query, "raw_response": result.text,
                      "available": result.available, "expected": case["expected"]}
            try:
                if not result.available:
                    raise ValueError("VLM unavailable")
                record["caption"] = _json_object(result.text)
                passed += 1
            except ValueError as exc:
                record["error"] = str(exc)
            (output / f'{case["id"]}.json').write_text(json.dumps(record, indent=2) + "\n")
    finally:
        await vlm.close()
    print(f"Caption contract: {passed}/{len(cases)}. Inspect visual accuracy separately: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for evaluation evidence")
    parser.add_argument("--config", type=Path, default=_SAMPLE / "yaml/worker.yaml")
    args = parser.parse_args()
    asyncio.run(evaluate(args.output.resolve(), args.config))
