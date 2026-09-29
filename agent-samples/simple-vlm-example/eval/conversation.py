# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model evaluation of conversation continuity and current-view grounding.

Run from the repository root with a serving VLM::

    uv run --project agent-samples/common/shared-agents --with pyyaml \
        python agent-samples/simple-vlm-example/eval/conversation.py
"""

from __future__ import annotations

import argparse
import asyncio
import io
import time
from pathlib import Path
from types import SimpleNamespace

import yaml
from PIL import Image, ImageDraw, ImageFont
from xr_ai_models import load_models_config, make_llm, make_vlm
from xr_ai_sample_agents import ConversationExchange, QuickConversation
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import StreamingImageQueryTool

_SAMPLE = Path(__file__).resolve().parents[1]
_CASES = Path(__file__).with_name("conversation_cases.yaml")


def _image(kind: str) -> bytes:
    image = Image.new("RGB", (960, 540), "#e8e8e8")
    draw = ImageDraw.Draw(image)
    if kind in {"blue_square", "red_square"}:
        color = "#1458ef" if kind == "blue_square" else "#e5292d"
        draw.rectangle((280, 80, 680, 480), fill=color)
    elif kind == "red_left_blue_right":
        draw.rectangle((80, 130, 350, 410), fill="#e5292d")
        draw.ellipse((600, 130, 870, 410), fill="#1458ef")
    elif kind in {"open_sign", "closed_sign"}:
        draw.rectangle((80, 85, 880, 455), fill="#f7d833")
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 120)
        except OSError:
            font = ImageFont.load_default()
        label = "OPEN" if kind == "open_sign" else "CLOSED"
        draw.text((150, 170), label, fill="black", font=font)
    elif kind != "blank":
        raise ValueError(f"unknown evaluation image {kind!r}")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class _StaticFrame:
    def __init__(self, reference) -> None:
        self.reference = reference
        self.calls = 0

    async def execute(self, _request):
        self.calls += 1
        return SimpleNamespace(image=self.reference)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--corpus", type=Path, default=_CASES)
    parser.add_argument("--thinking-budget", type=int)
    args = parser.parse_args()
    cases = yaml.safe_load(args.corpus.read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["name"] in args.case]
    if not cases:
        raise SystemExit("no conversation evaluation cases selected")
    for case in cases:
        for field in ("any_of", "none_of"):
            if any(not isinstance(term, str) for term in case.get(field, ())):
                raise ValueError(f"{case['name']}: {field} must contain quoted strings")

    models = load_models_config(_SAMPLE / "yaml/models.json")
    llm = make_llm(models, "llm")
    vlm = make_vlm(models, "vlm")
    prompt = (
        _SAMPLE / "worker/simple_vlm_example_worker/prompts/system.txt"
    ).read_text(encoding="utf-8")
    images = ImageRegistry()
    vision = StreamingImageQueryTool(images=images, vlm=vlm, system_prompt=prompt)
    passed = 0
    try:
        for case in cases:
            reference = images.put(_image(case["image"]))
            frame = _StaticFrame(reference)
            conversation = QuickConversation(  # type: ignore[arg-type]
                frame,
                vision,
                llm=llm,
                thinking_budget=args.thinking_budget,
            )
            history = tuple(
                ConversationExchange(**exchange)
                for exchange in case.get("history", ())
            )
            started = time.perf_counter()
            answer = "".join(
                [
                    text
                    async for text in conversation.stream(
                        case["query"],
                        "eval-participant",
                        history=history,
                        app_context=case.get("app_context", ""),
                    )
                ]
            ).strip()
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            normalized = answer.casefold()
            expected = case.get("any_of", ())
            forbidden = case.get("none_of", ())
            route = "current_view" if frame.calls else "direct"
            ok = route == case["route"] and bool(answer) and (
                not expected or any(term.casefold() in normalized for term in expected)
            ) and not any(term.casefold() in normalized for term in forbidden)
            passed += ok
            print(
                f"{'PASS' if ok else 'FAIL'} {case['name']} "
                f"route={route} {elapsed_ms}ms: {answer}",
                flush=True,
            )
    finally:
        await llm.close()
        await vlm.close()
    score = passed / len(cases)
    print(f"conversation eval: {passed}/{len(cases)} ({score:.1%})")
    if score < args.min_score:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
