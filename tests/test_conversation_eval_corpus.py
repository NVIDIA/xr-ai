# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corpus and rollout contracts; fake replies never stand in for model accuracy."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_EVAL = _ROOT / "agent-samples/simple-vlm-example/eval"
_SPEC = importlib.util.spec_from_file_location("conversation_eval", _EVAL / "conversation.py")
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


def _turn(query: str = "A question", **extra) -> dict:
    return {"query": query, "route": "direct", "image": "blank", "any_of": ["answer"], **extra}


def _trajectory(turns: list[dict]) -> dict:
    return {"name": "rollout", "split": "dev", "coverage": ["continuity"], "corpus": "fake.yaml", "turns": turns}


class _Replies:
    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = iter(replies)
        self.inputs: list[dict] = []

    async def stream(self, request, participant_id, *, history, app_context):
        self.inputs.append({
            "request": request, "participant": participant_id, "history": history, "context": app_context,
        })
        reply = next(self.replies)
        if isinstance(reply, Exception):
            yield "Partial unfinished text"
            raise reply
        yield reply[:4]
        yield reply[4:]


def test_conversation_corpus_preserves_reference_cases_and_partitions() -> None:
    hashes = {
        "conversation_cases.yaml": "93ee907bd62e7d4ff5386558a095305e9f6a9ff50e3aa797654da0b896f73f48",
        "conversation_challenge.yaml": "da1759fd25822be466849d408cfc8dc159e20e56cda5e3b79703fa045692aadf",
    }
    for name, expected in hashes.items():
        assert hashlib.sha256((_EVAL / name).read_bytes()).hexdigest() == expected
    isolated = _RUNNER._load_corpus(_RUNNER._ISOLATED, [])
    assert len(isolated) == 49
    assert sum(item["corpus"] == "conversation_cases.yaml" for item in isolated) == 29
    assert sum(item["corpus"] == "conversation_challenge.yaml" for item in isolated) == 20


def test_trajectory_coverage_includes_long_horizons_and_background_boundaries() -> None:
    sessions = _RUNNER._load_corpus(_RUNNER._TRAJECTORIES, [])
    assert len(sessions) == 16
    assert sum(len(session["turns"]) for session in sessions) == 210
    assert sum(session["split"] == "dev" for session in sessions) == 12
    assert sum(session["split"] == "challenge" for session in sessions) == 4
    assert sum(len(session["turns"]) >= 16 for session in sessions) >= 2
    assert any(len(session["turns"]) >= 32 and session["split"] == "dev" for session in sessions)
    assert any(len(session["turns"]) >= 32 and session["split"] == "challenge" for session in sessions)
    coverage = {label for session in sessions for label in session["coverage"]}
    assert {
        "memory_expiry", "multiple_agents", "background_correction", "cancelled_job",
        "failure", "quoted_commands", "background_injection", "unrelated_updates",
        "published_result", "participant_isolation", "camera_unavailable", "stale_state",
        "ambiguous_reference", "physical_camera_boundary", "action_refusal", "visual_recall",
    } <= coverage
    for session in sessions:
        assert all("history" not in turn for turn in session["turns"])
        assert all(len(turn.get("app_context", "")) <= 600 for turn in session["turns"])
    assert any(turn["route"] == "current_view" for session in sessions for turn in session["turns"])


@pytest.mark.parametrize("mutation", [
    {"any_of": [True]}, {"any_of": "answer"}, {"any_of": [""]}, {"query": " "},
    {"route": "not_a_route"}, {"image": "not_an_image"}, {"camera_unavailable": "yes"},
    {"app_context": {}}, {"participant": ""}, {"history": [{"user": "query"}]},
    {"any_of": [], "all_of": []},
])
def test_malformed_turns_are_rejected(mutation: dict) -> None:
    with pytest.raises(ValueError):
        _RUNNER._validate_turn(_turn(**mutation), "bad")


@pytest.mark.parametrize("kind", ["empty", "duplicate", "gold_history", "bad_coverage", "bad_split"])
def test_malformed_corpora_are_rejected(tmp_path: Path, kind: str) -> None:
    item = _trajectory([_turn()])
    corpus = [item]
    if kind == "empty":
        corpus = []
    elif kind == "duplicate":
        corpus.append(item.copy())
    elif kind == "gold_history":
        item["turns"][0]["history"] = []
    elif kind == "bad_coverage":
        item["coverage"] = ["continuity", "continuity"]
    else:
        item["split"] = "anything"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(corpus), encoding="utf-8")
    with pytest.raises(ValueError):
        _RUNNER._load_corpus((path,), [])


def test_empty_or_unknown_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown case"):
        _RUNNER._load_corpus(_RUNNER._ISOLATED, ["not_in_the_corpus"])
    with pytest.raises(ValueError, match="no conversation"):
        _RUNNER._load_corpus((), [])


def test_normalization_is_uniform_and_empty_answers_fail() -> None:
    turn = _turn(any_of=["can't do"], none_of=["done"])
    assert _RUNNER._check(turn, "I can’t\n do that.", "direct")
    assert not _RUNNER._check(turn, "done, I can't do that", "direct")
    assert not _RUNNER._check(turn, "I can't do that", "current_view")
    assert not _RUNNER._check(turn, "  ", "direct")


def test_numeric_and_single_letter_terms_do_not_match_substrings() -> None:
    assert _RUNNER._check(_turn(any_of=["7"]), "The answer is 7.", "direct")
    assert _RUNNER._check(_turn(any_of=["7"]), "The answer is 7", "direct")
    assert not _RUNNER._check(_turn(any_of=["7"]), "The answer is 17", "direct")
    assert not _RUNNER._check(_turn(any_of=["7"]), "The answer is 7.5", "direct")
    assert _RUNNER._check(_turn(any_of=["K"]), "It was bin K", "direct")
    assert not _RUNNER._check(_turn(any_of=["K"]), "I think it was B", "direct")


async def test_rollout_feeds_actual_wrong_answers_forward_and_bounds_context() -> None:
    turns = [_turn("q" * 400, app_context="x" * 900) for _ in range(8)]
    replies = _Replies(["generated wrong reply " + str(index) + "a" * 400 for index in range(8)])
    frame = _RUNNER._StaticFrame("reference")
    results = await _RUNNER._run_item(_trajectory(turns), replies, frame, {"blank": "reference"})

    assert len(results) == 8
    assert not any(result["passed"] for result in results)
    assert results[0]["history"] == []
    assert results[1]["history"][0]["assistant"].startswith("generated wrong reply 0")
    assert "answer" not in results[1]["history"][0]["assistant"]
    assert len(replies.inputs[-1]["history"]) == 4
    assert all(len(exchange.user) == len(exchange.assistant) == 240 for exchange in replies.inputs[-1]["history"])
    assert replies.inputs[-1]["history"][0].assistant.startswith("generated wrong reply 3")
    assert all(result["context_chars"] == 600 for result in results)
    assert all(result["history_chars"] <= 1920 for result in results)
    assert all(0 <= result["first_chunk_ms"] <= result["total_ms"] for result in results)


async def test_turn_errors_are_scored_and_do_not_abort_or_enter_history() -> None:
    replies = _Replies(["first answer", RuntimeError("model rejected request"), "third answer"])
    frame = _RUNNER._StaticFrame("reference")
    results = await _RUNNER._run_item(_trajectory([_turn(), _turn(), _turn()]), replies, frame, {"blank": "reference"})
    assert [result["passed"] for result in results] == [True, False, True]
    assert results[1]["error"] == "RuntimeError: model rejected request"
    assert [exchange.assistant for exchange in replies.inputs[2]["history"]] == ["first answer"]


async def test_participant_state_and_trajectory_resets_are_independent() -> None:
    turns = [
        _turn(participant="alice", app_context="Alice's report"),
        _turn(participant="bob", app_context="Bob's report"),
        _turn(participant="alice", app_context=""),
        _turn(participant="bob"),
    ]
    replies = _Replies(["Alice answer", "Bob answer", "Alice next answer", "Bob next answer", "Reset answer"])
    frame = _RUNNER._StaticFrame("reference")
    results = await _RUNNER._run_item(_trajectory(turns), replies, frame, {"blank": "reference"})
    assert [exchange.assistant for exchange in replies.inputs[2]["history"]] == ["Alice answer"]
    assert [exchange.assistant for exchange in replies.inputs[3]["history"]] == ["Bob answer"]
    assert results[2]["app_context"] == ""
    assert results[3]["app_context"] == "Bob's report"
    await _RUNNER._run_item(_trajectory([_turn()]), replies, frame, {"blank": "reference"})
    assert replies.inputs[-1]["history"] == ()
    assert replies.inputs[-1]["context"] == ""


async def test_multiple_camera_calls_fail_even_with_correct_answer() -> None:
    frame = _RUNNER._StaticFrame("reference")

    class RepeatedView:
        async def stream(self, *_args, **_kwargs):
            await frame.execute(None)
            await frame.execute(None)
            yield "answer"

    results = await _RUNNER._run_item(
        _trajectory([_turn(route="current_view")]), RepeatedView(), frame, {"blank": "reference"},
    )
    assert not results[0]["passed"]
    assert results[0]["frame_calls"] == 2


def test_error_only_report_does_not_present_failure_latency_as_performance(capsys) -> None:
    result = {
        "name": "broken", "corpus": "challenge.yaml", "split": "challenge", "passed": False,
        "error": "ConnectionError", "first_chunk_ms": None, "total_ms": 1.0,
    }
    _RUNNER._report([result])
    report = capsys.readouterr().out
    assert "errors=1" in report
    assert "whole-trajectory score: 0/1" in report
    assert report.count("no completed calls") == 2
