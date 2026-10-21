# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the live conversation-context evals."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from xr_ai_conversation import Conversation
from xr_ai_conversation._conversation import _messages
from xr_ai_models import LLMService, load_models_config, load_models_config_from_dict, make_llm

CASES = Path(__file__).resolve().parent / "cases"
_DASHES = "‐‑‒–—"


def parser(description: str) -> argparse.ArgumentParser:
    """Return an argument parser selecting the model under evaluation."""

    arguments = argparse.ArgumentParser(description=description)
    arguments.add_argument("--models", type=Path, help="models.json to load the LLM from (overrides --base-url)")
    arguments.add_argument("--model", default="llm", help="model entry name in --models")
    arguments.add_argument("--base-url", default="http://localhost:8108", help="OpenAI-compatible LLM endpoint")
    arguments.add_argument("--preset", default="nemotron_omni", help="model preset used with --base-url")
    arguments.add_argument("--samples", type=int, help="answers sampled per probe (overrides the case file)")
    return arguments


def make_eval_llm(args: argparse.Namespace) -> LLMService:
    """Build the LLM the same way applications do, from a models configuration."""

    if args.models is not None:
        return make_llm(load_models_config(args.models), args.model)
    config = {"models": {"llm": {"adapter": {"preset": args.preset}, "endpoint": {"base_url": args.base_url}}}}
    return make_llm(load_models_config_from_dict(config, source="--base-url"), "llm")


def load_cases(name: str) -> Any:
    return yaml.safe_load((CASES / name).read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    text = text.lower()
    for dash in _DASHES:
        text = text.replace(dash, "-")
    return " ".join(text.split())


def matches(answer: str, expectation: Mapping[str, Any]) -> bool:
    """Check an answer against ``any_of``, ``none_of`` and ``not_starting_with`` substrings."""

    text = normalize(answer)
    return (
        any(normalize(item) in text for item in expectation.get("any_of", ()))
        and not any(normalize(item) in text for item in expectation.get("none_of", ()))
        and not any(text.startswith(normalize(item)) for item in expectation.get("not_starting_with", ()))
    )


def answer_prompt(conversation: Conversation, participant: str, question: str) -> tuple[Any, ...]:
    """Render the answer request the engine would send, without recording an exchange."""

    return _messages(
        question,
        instructions=conversation._instructions,
        vision_available=conversation._vision is not None,
        summaries=conversation._summary_blocks(participant),
        newer=conversation._newer_blocks(participant),
        context_budget_bytes=conversation._context_budget_bytes,
        participant_id=participant,
    )


async def answer_without_recording(llm: LLMService, conversation: Conversation, participant: str, question: str) -> str:
    response = await llm.chat(answer_prompt(conversation, participant, question))
    return response.content


async def settle(conversation: Conversation, participant: str) -> None:
    """Wait until no compaction job is pending for the participant."""

    memory = conversation._context._participants.get(participant)
    while memory is not None and (
        jobs := [job for job in (memory.dialogue_job, *memory.context_jobs.values()) if job is not None]
    ):
        await asyncio.gather(*jobs, return_exceptions=True)
        # Awaiting finished jobs does not yield; let their completion callbacks run.
        await asyncio.sleep(0)


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
