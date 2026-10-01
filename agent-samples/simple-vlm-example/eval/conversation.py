# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate isolated queries and generated-reply conversation rollouts.

Run from the repository root with serving LLM and VLM endpoints::

    uv run --project agent-samples/common/shared-agents --with pyyaml \\
        python agent-samples/simple-vlm-example/eval/conversation.py

Lexical checks are coarse regression checks, not a semantic quality judge.
First-chunk latency measures available assistant text, not audible playback.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import re
import time
from collections import deque
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import yaml
from PIL import Image, ImageDraw, ImageFont
from xr_ai_hub import FrameUnavailable
from xr_ai_models import load_models_config, make_llm, make_vlm
from xr_ai_sample_agents import ConversationExchange, QuickConversation
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import StreamingImageQueryTool

_SAMPLE = Path(__file__).resolve().parents[1]
_EVAL = Path(__file__).resolve().parent
_ISOLATED = (_EVAL / "conversation_cases.yaml", _EVAL / "conversation_challenge.yaml")
_TRAJECTORIES = (_EVAL / "conversation_trajectories.yaml", _EVAL / "conversation_trajectories_challenge.yaml")
_IMAGES = {"blank", "blue_square", "red_square", "red_left_blue_right", "open_sign", "closed_sign"}


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
        draw.text((150, 170), "OPEN" if kind == "open_sign" else "CLOSED", fill="black", font=font)
    elif kind != "blank":
        raise ValueError(f"unknown evaluation image {kind!r}")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class _StaticFrame:
    def __init__(self, reference) -> None:
        self.reference = reference
        self.calls = 0
        self.unavailable = False

    async def execute(self, _request):
        self.calls += 1
        if self.unavailable:
            raise FrameUnavailable("The camera is unavailable right now.")
        return SimpleNamespace(image=self.reference)


def _validate_turn(turn: dict, label: str) -> None:
    if not isinstance(turn.get("query"), str) or not turn["query"].strip():
        raise ValueError(f"{label}: query must be nonempty")
    if turn.get("route") not in {"direct", "current_view"} or turn.get("image") not in _IMAGES:
        raise ValueError(f"{label}: invalid route or image")
    for field in ("any_of", "all_of", "none_of"):
        terms = turn.get(field, [])
        if not isinstance(terms, list) or any(not isinstance(term, str) or not term.strip() for term in terms):
            raise ValueError(f"{label}: {field} must contain nonempty strings")
    if not turn.get("any_of") and not turn.get("all_of"):
        raise ValueError(f"{label}: a positive answer expectation is required")
    for field in ("app_context", "participant"):
        if field in turn and not isinstance(turn[field], str):
            raise ValueError(f"{label}: {field} must be a string")
    if "participant" in turn and not turn["participant"].strip():
        raise ValueError(f"{label}: participant must be nonempty")
    if "camera_unavailable" in turn and not isinstance(turn["camera_unavailable"], bool):
        raise ValueError(f"{label}: camera_unavailable must be boolean")
    history = turn.get("history", [])
    if not isinstance(history, list) or any(
        not isinstance(exchange, dict) or set(exchange) != {"user", "assistant"}
        or any(not isinstance(value, str) for value in exchange.values())
        for exchange in history
    ):
        raise ValueError(f"{label}: invalid completed history")


def _load_corpus(paths: tuple[Path, ...], names: list[str]) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        corpus = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(corpus, list) or not corpus:
            raise ValueError(f"{path}: corpus must be a nonempty list")
        for item in corpus:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                raise ValueError(f"{path}: missing case name")
            name = item["name"]
            if name in seen:
                raise ValueError(f"duplicate case name {name!r}")
            seen.add(name)
            if "turns" in item:
                if item.get("split") not in {"dev", "challenge"} or not item["turns"]:
                    raise ValueError(f"{name}: trajectory needs a split and turns")
                coverage = item.get("coverage")
                if (
                    not isinstance(item["turns"], list) or not isinstance(coverage, list) or not coverage
                    or any(not isinstance(label, str) or not label.strip() for label in coverage)
                    or len(set(coverage)) != len(coverage)
                ):
                    raise ValueError(f"{name}: invalid trajectory turns or coverage")
                for index, turn in enumerate(item["turns"]):
                    if not isinstance(turn, dict) or "history" in turn:
                        raise ValueError(f"{name}/{index + 1}: trajectory history must come from generated replies")
                    _validate_turn(turn, f"{name}/{index + 1}")
            else:
                _validate_turn(item, name)
            items.append({**item, "corpus": path.name})
    unknown = set(names) - seen
    if unknown:
        raise ValueError(f"unknown case names: {', '.join(sorted(unknown))}")
    selected = [item for item in items if not names or item["name"] in names]
    if not selected:
        raise ValueError("no conversation evaluation cases selected")
    return selected


def _normalize(text: str) -> str:
    return " ".join(text.casefold().replace("’", "'").replace("‘", "'").split())


def _check(turn: dict, answer: str, route: str) -> bool:
    normalized = _normalize(answer)
    return bool(normalized) and route == turn["route"] and (
        not turn.get("any_of") or any(_contains(normalized, term) for term in turn["any_of"])
    ) and all(_contains(normalized, term) for term in turn.get("all_of", [])) and not any(
        _contains(normalized, term) for term in turn.get("none_of", [])
    )


def _contains(answer: str, term: str) -> bool:
    term = _normalize(term)
    if term.isdecimal() or len(term) == 1:
        return re.search(r"(?<![\w.])" + re.escape(term) + r"(?!\w|\.\d)", answer) is not None
    return term in answer


async def _run_item(item: dict, conversation, frame, references: dict) -> list[dict]:
    histories: dict[str, deque] = {}
    contexts: dict[str, str] = {}
    trajectory = "turns" in item
    results: list[dict] = []
    for index, turn in enumerate(item.get("turns", [item])):
        participant = turn.get("participant", "eval-participant")
        history = histories.setdefault(participant, deque(maxlen=4))
        if not trajectory:
            history.extend(ConversationExchange(**exchange) for exchange in turn.get("history", []))
        if "app_context" in turn:
            contexts[participant] = turn["app_context"][:600]
        context = contexts.get(participant, "")
        bounded_history = tuple(
            ConversationExchange(exchange.user[:240], exchange.assistant[:240]) for exchange in history
        )
        frame.reference = references[turn["image"]]
        frame.unavailable = turn.get("camera_unavailable", False)
        before_calls = frame.calls
        started = time.perf_counter()
        first_chunk_ms = None
        chunks: list[str] = []
        error = None
        try:
            async for text in conversation.stream(
                turn["query"], participant, history=bounded_history, app_context=context,
            ):
                if text and first_chunk_ms is None:
                    first_chunk_ms = round((time.perf_counter() - started) * 1000, 1)
                chunks.append(text)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        total_ms = round((time.perf_counter() - started) * 1000, 1)
        answer = "".join(chunks).strip()
        route = "current_view" if frame.calls > before_calls else "direct"
        results.append({
            "name": item["name"], "turn": index + 1, "corpus": item["corpus"],
            "split": item.get("split", "isolated"), "participant": participant,
            "query": turn["query"],
            "history": [{"user": exchange.user, "assistant": exchange.assistant} for exchange in bounded_history],
            "app_context": context,
            "history_chars": sum(len(exchange.user) + len(exchange.assistant) for exchange in bounded_history),
            "context_chars": len(context), "route": route, "expected_route": turn["route"],
            "frame_calls": frame.calls - before_calls, "answer": answer,
            "passed": error is None and frame.calls - before_calls <= 1 and _check(turn, answer, route),
            "first_chunk_ms": first_chunk_ms, "total_ms": total_ms, "error": error,
        })
        # Production retains completed replies only, including incorrect ones.
        if answer and error is None:
            history.append(ConversationExchange(turn["query"], answer))
    return results


def _report(results: list[dict]) -> None:
    groups = sorted({result["corpus"] for result in results})
    for group in groups:
        subset = [result for result in results if result["corpus"] == group]
        passed = sum(result["passed"] for result in subset)
        errors = sum(bool(result["error"]) for result in subset)
        print(f"{group} turn score: {passed}/{len(subset)} ({passed / len(subset):.1%}); errors={errors}")
        if subset[0]["split"] != "isolated":
            names = {result["name"] for result in subset}
            complete = sum(all(result["passed"] for result in subset if result["name"] == name) for name in names)
            print(f"  whole-trajectory score: {complete}/{len(names)}")
        for metric in ("first_chunk_ms", "total_ms"):
            values = sorted(result[metric] for result in subset if not result["error"] and result[metric] is not None)
            if values:
                p95 = values[math.ceil(len(values) * 0.95) - 1]
                print(f"  {metric}: median={median(values):.1f} p95={p95:.1f} n={len(values)} (completed calls)")
            else:
                print(f"  {metric}: no completed calls")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--corpus", type=Path, action="append")
    parser.add_argument("--suite", choices=("isolated", "trajectories", "all"), default="all")
    parser.add_argument("--output", type=Path, help="JSONL generated replies, bounded inputs, errors and timing")
    parser.add_argument("--models", type=Path, default=_SAMPLE / "yaml/models.json")
    args = parser.parse_args()
    default_paths = {"isolated": _ISOLATED, "trajectories": _TRAJECTORIES, "all": _ISOLATED + _TRAJECTORIES}
    items = _load_corpus(tuple(args.corpus or default_paths[args.suite]), args.case)
    models = load_models_config(args.models)
    llm = make_llm(models, "llm")
    vlm = make_vlm(models, "vlm")
    prompt = (_SAMPLE / "worker/simple_vlm_example_worker/prompts/system.txt").read_text(encoding="utf-8")
    images = ImageRegistry()
    references = {kind: images.put(_image(kind)) for kind in _IMAGES}
    vision = StreamingImageQueryTool(images=images, vlm=vlm, system_prompt=prompt)
    results: list[dict] = []
    output = args.output.open("w", encoding="utf-8") if args.output else None
    try:
        for item in items:
            frame = _StaticFrame(references["blank"])
            conversation = QuickConversation(frame, vision, llm=llm)  # type: ignore[arg-type]
            turns = await _run_item(item, conversation, frame, references)
            results.extend(turns)
            for result in turns:
                print(
                    f"{'PASS' if result['passed'] else 'FAIL'} {result['name']}/{result['turn']} "
                    f"route={result['route']} first={result['first_chunk_ms']}ms total={result['total_ms']}ms: "
                    f"{result['error'] or result['answer']}", flush=True,
                )
                if output:
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
            if output:
                output.flush()
    finally:
        if output:
            output.close()
        await llm.close()
        await vlm.close()
    _report(results)
    passed = sum(result["passed"] for result in results)
    print(f"conversation eval: {passed}/{len(results)} ({passed / len(results):.1%})")
    if passed / len(results) < args.min_score:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
