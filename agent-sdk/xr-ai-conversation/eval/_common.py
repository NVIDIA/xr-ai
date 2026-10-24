# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the live conversation-context evals."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml
from xr_ai_models import ChatResponse, ChatStreamEvent, LLMService, load_models_config, load_models_config_from_dict
from xr_ai_models import make_llm as _make_llm

CASES = Path(__file__).resolve().parent / "cases"
CASE_FILES = ("conflicts.yaml", "session_lab.yaml", "session_renovation.yaml")
_DASHES = "‐‑‒–—"


def parser(description: str) -> argparse.ArgumentParser:
    """Return an argument parser selecting the model under evaluation."""

    arguments = argparse.ArgumentParser(description=description)
    arguments.add_argument("--models", type=Path, help="models.json to load the LLM from (overrides --base-url)")
    arguments.add_argument("--model", default="llm", help="model entry name in --models")
    arguments.add_argument("--base-url", default="http://localhost:8108", help="OpenAI-compatible LLM endpoint")
    arguments.add_argument("--preset", default="nemotron_omni", help="model preset used with --base-url")
    arguments.add_argument("--samples", type=int, help="answers sampled per question (overrides the case file)")
    return arguments


def make_eval_llm(args: argparse.Namespace) -> LLMService:
    """Build the LLM the same way applications do, from a models configuration."""

    if args.models is not None:
        return _make_llm(load_models_config(args.models), args.model)
    config = {"models": {"llm": {"adapter": {"preset": args.preset}, "endpoint": {"base_url": args.base_url}}}}
    return _make_llm(load_models_config_from_dict(config, source="--base-url"), "llm")


def load_cases(name: str) -> Any:
    return yaml.safe_load((CASES / name).read_text(encoding="utf-8"))


def graded_cases(suite: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield every graded question in a conflict suite or session."""

    yield from suite.get("cases", ())
    yield from suite.get("probes", ())


def validate_suite(suite: Mapping[str, Any]) -> None:
    """Check a case file's structure so a malformed case fails before reaching a model."""

    for case in graded_cases(suite):
        name = case.get("name")
        assert isinstance(name, str) and name, f"case without a name: {case}"
        assert isinstance(case.get("question"), str), f"{name}: question is required"
        assert case.get("any_of"), f"{name}: any_of is required"
        assert case.get("accept"), f"{name}: accept examples are required"
        assert case.get("reject"), f"{name}: reject examples are required"
        for step in case.get("steps", ()):
            assert len(step) == 1 and next(iter(step)) in {"say", "publish"}, f"{name}: unknown step {step}"
    for turn in suite.get("turns", ()):
        assert "user" in turn, f"session turn without user text: {turn}"


def normalize(text: str) -> str:
    text = text.lower()
    for dash in _DASHES:
        text = text.replace(dash, "-")
    return " ".join(text.replace("*", "").split())


def _contains(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(normalize(phrase))}(?!\w)", text) is not None


def matches(answer: str, expectation: Mapping[str, Any]) -> bool:
    """Grade an answer by whole words: one of ``any_of``, none of ``none_of``, and no ``not_starting_with``."""

    text = normalize(answer)
    return (
        any(_contains(text, item) for item in expectation.get("any_of", ()))
        and not any(_contains(text, item) for item in expectation.get("none_of", ()))
        and not any(_contains(text[: len(item) + 1], item) for item in expectation.get("not_starting_with", ()))
    )


def expand(spec: str | Mapping[str, Any]) -> str:
    """Expand a literal or a generated ``prefix``/``repeat``/``suffix`` text."""

    if isinstance(spec, str):
        return spec
    lines = [spec["repeat"].format(index=index, digit=index % 10) for index in range(spec["count"])]
    return "\n".join(filter(None, (spec.get("prefix"), *lines, spec.get("suffix"))))


class ScriptedLLM:
    """Answer and summarize from a script while seeding, and delegate to the real model afterwards.

    Seeding drives the engine's public flow, so compaction really runs; the
    script only fixes what the model would have said.
    """

    def __init__(self, llm: LLMService) -> None:
        self.llm = llm
        self.capabilities = llm.capabilities
        self.answer: str | None = None
        self.summaries: dict[str, str] = {}

    def stream_events(self, messages: Any, **kwargs: Any) -> AsyncIterator[ChatStreamEvent]:
        if self.answer is None:
            return self.llm.stream_events(messages, **kwargs)
        answer = self.answer

        async def scripted() -> AsyncIterator[ChatStreamEvent]:
            yield ChatStreamEvent(kind="text", text=answer)

        return scripted()

    async def chat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        system, request = messages[0].content, messages[-1].content
        source = "dialogue" if "completed conversation history" in system else request.split("\n", 1)[0]
        source = source.removeprefix("Source: ")
        if source not in self.summaries:
            return await self.llm.chat(messages, **kwargs)
        return ChatResponse(
            content=self.summaries[source], reasoning=None, tool_calls=None, finish_reason="stop", raw={}
        )


async def standalone_started() -> None:
    """Response-start signal for answers called without a router."""


async def settle(manager: Any, participant: str) -> None:
    """Wait until no compaction job is pending for the participant."""

    memory = manager._context._participants.get(participant)
    while memory is not None and (
        jobs := [job for job in (memory.dialogue_job, *memory.context_jobs.values()) if job is not None]
    ):
        await asyncio.gather(*jobs, return_exceptions=True)
        # Awaiting finished jobs does not yield; let their completion callbacks run.
        await asyncio.sleep(0)


async def ask(llm: LLMService, conversation: Any, participant: str, question: str) -> str:
    """Answer from the request the engine builds, without recording an exchange."""

    response = await llm.chat(conversation._answer_messages(participant, question))
    return response.content


class LogCounter(logging.Handler):
    """Count conversation-context log records by message prefix."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def count(self, prefix: str) -> int:
        return sum(message.startswith(prefix) for message in self.messages)

    def __enter__(self) -> LogCounter:
        logging.getLogger("xr_ai_conversation").addHandler(self)
        return self

    def __exit__(self, *_exc: object) -> None:
        logging.getLogger("xr_ai_conversation").removeHandler(self)
