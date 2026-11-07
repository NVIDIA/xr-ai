# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live eval: does an LLM endpoint honour JSON Schema and forced tool calls?

Run it against any configured model before declaring ``json_schema`` or
``tool_choice`` in its capabilities. The eval enables both capabilities on the
client it builds, so it measures what the endpoint does rather than what its
preset claims. It exits non-zero when a JSON Schema mode returns any answer
that does not match its schema, or a forced tool mode returns no call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
import yaml
from xr_ai_models import (
    ChatMessage,
    LLMService,
    OpenAICompatLLM,
    ToolDef,
    load_models_config,
    load_models_config_from_dict,
    make_llm,
)
from xr_ai_models._openai_compat import _request_headers, _with_schema_instruction

CASES = Path(__file__).resolve().parent / "cases" / "structured_output.yaml"
TOOL = "commit_answer"

# Every answer in these modes must match its schema.
SCHEMA_MODES = ("json_schema", "json_schema/stream")
# Every response in these modes must contain a tool call. Their arguments are
# reported but not gated: servers may constrain only the call, not its schema.
FORCED_MODES = ("tool_named", "tool_named/stream", "tool_required")
CONSTRAINED = (*SCHEMA_MODES, *FORCED_MODES)
# References: what callers can do without the feature, and the ablation that
# checks the client's choice of when to state the schema in the prompt.
REFERENCE = ("prompt_only", "tools_auto", "json_schema_flipped")


@dataclass
class Sample:
    case: str
    mode: str
    thinking: bool
    valid: bool
    correct: bool
    finish: str | None
    seconds: float
    tokens: int | None
    error: str | None = None
    answer: str = ""
    called: bool | None = None


def _parser() -> argparse.ArgumentParser:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--models", type=Path, help="models.json to load the LLM from (overrides --base-url)")
    arguments.add_argument("--model", default="llm", help="model entry name in --models")
    arguments.add_argument("--base-url", default="http://localhost:8108", help="OpenAI-compatible LLM endpoint")
    arguments.add_argument("--preset", default="nemotron_omni", help="model preset used with --base-url")
    arguments.add_argument("--samples", type=int, default=5, help="answers sampled per case, mode and thinking setting")
    arguments.add_argument("--temperature", type=float, default=0.6)
    arguments.add_argument("--thinking-budget", type=int, default=512)
    arguments.add_argument("--concurrency", type=int, default=4)
    arguments.add_argument("--modes", nargs="*", default=[*CONSTRAINED, *REFERENCE])
    arguments.add_argument("--cases", nargs="*", help="case names to run (default: all)")
    arguments.add_argument("--thinking", choices=("off", "on", "both"), default="both")
    arguments.add_argument("--report", type=Path, help="write per-sample results as JSON")
    return arguments


def _make_llm(args: argparse.Namespace) -> LLMService:
    if args.models is not None:
        llm = make_llm(load_models_config(args.models), args.model)
    else:
        config = {"models": {"llm": {"adapter": {"preset": args.preset}, "endpoint": {"base_url": args.base_url}}}}
        llm = make_llm(load_models_config_from_dict(config, source="--base-url"), "llm")
    # Measure the endpoint, not its declared capabilities.
    llm.capabilities = replace(llm.capabilities, json_schema=True, tool_choice=True)
    return llm


def _schema_instruction(schema: dict[str, Any]) -> str:
    return "Respond with only a JSON value that conforms to this JSON Schema:\n" + json.dumps(
        schema, ensure_ascii=False, separators=(",", ":"),
    )


def _resolve(answer: Any, pointer: str) -> Any:
    value = answer
    for part in pointer.lstrip("/").split("/"):
        if part:
            value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def check(answer: Any, rule: dict[str, Any]) -> bool:
    """Return whether *answer* satisfies one case check."""

    try:
        value = _resolve(answer, rule["path"])
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    if "equals" in rule:
        return value in rule["equals"] and type(value) in {type(v) for v in rule["equals"]}
    if "contains" in rule:
        return isinstance(value, str) and any(s.lower() in value.lower() for s in rule["contains"])
    if "range" in rule:
        low, high = rule["range"]
        return isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high
    if "min_items" in rule:
        return isinstance(value, list) and len(value) >= rule["min_items"]
    raise ValueError(f"unknown check: {rule}")


def grade(text: str, schema: dict[str, Any], checks: list[dict[str, Any]]) -> tuple[bool, bool]:
    """Return (schema-valid, every check passed) for one raw answer."""

    from jsonschema import Draft202012Validator

    try:
        answer = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, False
    if any(Draft202012Validator(schema).iter_errors(answer)):
        return False, False
    return True, all(check(answer, rule) for rule in checks)


async def _flipped(llm: OpenAICompatLLM, messages, schema, options) -> tuple[str, str | None, int | None]:
    """Send the constraint with the client's schema statement reversed.

    The client states the schema only when reasoning may precede the answer.
    This ablation does the opposite, so a model that needs the other choice
    shows up as a better reference than ``json_schema``.
    """

    payload = llm._build_payload(messages, tools=None, stream=False, response_schema=schema, **options)
    if llm._may_reason(options["enable_thinking"]):
        # Every case has a system message; restore it without the schema.
        payload["messages"][0]["content"] = messages[0].content
    else:
        payload["messages"] = _with_schema_instruction(payload["messages"], schema)
    resp = await llm._client.post(llm._chat_url, json=payload, headers=_request_headers(llm._api_key, None))
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    return choice["message"].get("content") or "", choice.get("finish_reason"), data.get("usage", {}).get(
        "completion_tokens",
    )


async def run_one(llm, case, mode, thinking, args) -> Sample:
    schema = case["schema"]
    system = case["system"]
    if mode == "prompt_only":
        system = f"{system}\n\n{_schema_instruction(schema)}"
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=case["user"])]
    options: dict[str, Any] = {
        "temperature": args.temperature,
        "enable_thinking": thinking,
        "max_tokens": args.thinking_budget + 1024 if thinking else 1024,
        "thinking_budget": args.thinking_budget if thinking else None,
    }
    tools = [ToolDef(TOOL, "Commit the final answer.", schema)]
    start = time.perf_counter()
    text, finish, tokens, called = "", None, None, None
    try:
        if mode in {"prompt_only", "json_schema", "tools_auto", "tool_named", "tool_required"}:
            kwargs: dict[str, Any] = {}
            if mode == "json_schema":
                kwargs["response_schema"] = schema
            elif mode.startswith("tool"):
                kwargs["tools"] = tools
                if mode != "tools_auto":
                    kwargs["tool_choice"] = TOOL if mode == "tool_named" else "required"
            response = await llm.chat(messages, **kwargs, **options)
            # A tool loop treats text without a call as the final answer.
            text = response.tool_calls[0].arguments if response.tool_calls else response.content
            if "tools" in kwargs:
                called = bool(response.tool_calls)
            finish = response.finish_reason
            tokens = response.raw.get("usage", {}).get("completion_tokens")
        elif mode in {"json_schema/stream", "tool_named/stream"}:
            kwargs = {"response_schema": schema} if mode == "json_schema/stream" else {
                "tools": tools, "tool_choice": TOOL,
            }
            parts = []
            called = False if "tools" in kwargs else None
            async for event in llm.stream_events(messages, **kwargs, **options):
                if event.kind == "text":
                    parts.append(event.text)
                elif event.kind == "tool_call":
                    parts = [event.tool_call.arguments]
                    called = True
                elif event.kind == "finish":
                    finish = event.finish_reason
            text = "".join(parts)
        elif mode == "json_schema_flipped":
            text, finish, tokens = await _flipped(llm, messages, schema, options)
        else:
            raise ValueError(f"unknown mode {mode}")
    # Transport failures and stream validation errors count as failed samples.
    except (httpx.HTTPError, ValueError, RuntimeError) as exc:
        return Sample(case["name"], mode, thinking, False, False, finish, time.perf_counter() - start, tokens,
                      f"{type(exc).__name__}: {exc}"[:200], called=False if called is not None else None)
    valid, correct = grade(text, schema, case["checks"])
    return Sample(case["name"], mode, thinking, valid, correct, finish, time.perf_counter() - start, tokens,
                  answer=text[:2000], called=called)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarize(samples: list[Sample], modes: list[str]) -> tuple[str, list[str]]:
    rows = defaultdict(list)
    for sample in samples:
        rows[(sample.mode, sample.thinking)].append(sample)
    lines = [
        "| mode | thinking | called | valid | correct | truncated | errors | p50 s | p95 s | mean tokens |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    failures = []
    for mode in modes:
        for thinking in (False, True):
            group = rows.get((mode, thinking))
            if not group:
                continue
            n = len(group)
            valid = sum(s.valid for s in group)
            correct = sum(s.correct for s in group)
            truncated = sum(s.finish == "length" for s in group)
            errors = sum(s.error is not None for s in group)
            seconds = [s.seconds for s in group]
            tokens = [s.tokens for s in group if s.tokens is not None]
            mean_tokens = f"{statistics.mean(tokens):.0f}" if tokens else "-"
            calls = [s.called for s in group if s.called is not None]
            called = f"{sum(calls)}/{n}" if calls else "-"
            lines.append(
                f"| {mode} | {'on' if thinking else 'off'} | {called} | {valid}/{n} | {correct}/{n} | {truncated} "
                f"| {errors} "
                f"| {_percentile(seconds, 0.5):.2f} | {_percentile(seconds, 0.95):.2f} | {mean_tokens} |"
            )
            label = f"{mode} thinking={'on' if thinking else 'off'}"
            if mode in SCHEMA_MODES and valid < n:
                bad = sorted({s.case for s in group if not s.valid})
                failures.append(f"{label}: answer did not match its schema in {', '.join(bad)}")
            if mode in FORCED_MODES and sum(bool(s.called) for s in group) < n:
                bad = sorted({s.case for s in group if not s.called})
                failures.append(f"{label}: no tool call in {', '.join(bad)}")
    return "\n".join(lines), failures


async def main() -> int:
    args = _parser().parse_args()
    from jsonschema import Draft202012Validator

    cases = yaml.safe_load(CASES.read_text(encoding="utf-8"))["cases"]
    for case in cases:
        Draft202012Validator.check_schema(case["schema"])
    if args.cases:
        unknown = set(args.cases) - {case["name"] for case in cases}
        if unknown:
            raise SystemExit(f"unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in args.cases]
    llm = _make_llm(args)
    if "json_schema_flipped" in args.modes and not isinstance(llm, OpenAICompatLLM):
        args.modes = [m for m in args.modes if m != "json_schema_flipped"]
    thinking = {"off": (False,), "on": (True,), "both": (False, True)}[args.thinking]
    gate = asyncio.Semaphore(args.concurrency)

    async def bounded(*job):
        async with gate:
            return await run_one(llm, *job, args)

    jobs = [
        bounded(case, mode, think)
        for case in cases for mode in args.modes for think in thinking for _ in range(args.samples)
    ]
    try:
        samples = await asyncio.gather(*jobs)
    finally:
        await llm.close()
    table, failures = summarize(samples, args.modes)
    print(table)
    errors = sorted({s.error for s in samples if s.error})
    if errors:
        print("\nErrors:\n" + "\n".join(f"- {e}" for e in errors))
    per_case = defaultdict(lambda: [0, 0])
    for s in samples:
        if s.mode in CONSTRAINED:
            per_case[s.case][0] += s.correct
            per_case[s.case][1] += 1
    print("\nCorrect answers per case (constrained modes):")
    for name, (correct, total) in per_case.items():
        print(f"- {name}: {correct}/{total}")
    if args.report is not None:
        args.report.write_text(json.dumps([s.__dict__ for s in samples], indent=1), encoding="utf-8")
    if failures:
        print("\nFAIL:\n" + "\n".join(f"- {f}" for f in failures))
        return 1
    print("\nPASS: every JSON Schema answer matched its schema and every forced call was made.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
