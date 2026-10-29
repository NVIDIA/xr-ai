"""Boundary tests use scripted classifications, not claimed Clef accuracy."""

from __future__ import annotations

import asyncio
import math

import pytest
from semantic_turn_prototype._turns import _Policy, _TurnController
from xr_ai_models import DecisionAnswer, DecisionResponse


class _Decisions:
    def __init__(self):
        self.choice = "ready"
        self.relation = "separate"
        self.confidence = 0.95
        self.calls = []
        self.error = None
        self.entered = asyncio.Event()
        self.release = None

    async def decide(self, state, questions, **kwargs):
        self.calls.append((state, questions))
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if self.error:
            raise self.error
        answers = {}
        for key, question in questions.items():
            choice = self.choice if key == "disposition" else self.relation
            confidence = self.confidence
            probabilities = {
                k: (confidence if k == choice else (1 - confidence) / (len(question.criteria) - 1))
                for k in question.criteria
            }
            answers[key] = DecisionAnswer(choice, confidence, probabilities)
        return DecisionResponse(answers, "scripted-test-double", {}, {})


def _setup(**kwargs):
    decisions = _Decisions()
    controller = _TurnController(decisions, _Policy(stable_s=0, **kwargs))
    controller.open("a", "session")
    return controller, decisions


def _feed(
    controller, text="Open the panel", *, segment=0, revision=0, now=1, who="a", epoch="session"
):
    return controller.feed(who, epoch, segment=segment, revision=revision, text=text, now=now)


async def _poll(controller, now=1, who="a", epoch="session"):
    return await controller.poll(who, epoch, now=now)


@pytest.mark.asyncio
async def test_incomplete_query_survives_very_long_pause_without_repeated_inference():
    c, d = _setup()
    d.choice = "hold"
    _feed(c, "Put the label next to")
    assert (await _poll(c))[0].kind == "hold"
    assert await _poll(c, now=3600) == []
    assert len(d.calls) == 1
    _feed(c, "the red cylinder", segment=1, now=3601)
    d.choice = "ready"
    result = await _poll(c, now=3601)
    assert result[0].text == "Put the label next to the red cylinder"


@pytest.mark.asyncio
async def test_filler_does_not_dispatch_but_meaningful_word_like_is_preserved():
    c, d = _setup()
    d.choice = "filler"
    _feed(c, "um")
    assert (await _poll(c))[0].kind == "ignore"
    d.choice = "ready"
    _feed(c, "What is an expression like this called?", segment=1, now=2)
    result = await _poll(c, 2)
    assert "like" in result[0].text
    assert "um" not in result[0].text


@pytest.mark.asyncio
async def test_revision_replaces_snapshot_rather_than_appending_text():
    c, d = _setup()
    _feed(c, "move it red")
    _feed(c, "move it right", revision=1)
    result = await _poll(c)
    assert result[0].text == "move it right"
    assert len(d.calls) == 1
    assert _feed(c, "move it left", revision=2)[0].reason == "late_revision_of_released_segment"


@pytest.mark.asyncio
async def test_stability_delay_uses_text_changes_not_vad_or_duplicate_snapshots():
    d = _Decisions()
    c = _TurnController(d, _Policy(stable_s=0.2))
    c.open("a", "session")
    _feed(c)
    assert await _poll(c, 1.1) == []
    _feed(c, revision=1, now=1.15)
    assert (await _poll(c, 1.21))[0].kind == "dispatch"


@pytest.mark.asyncio
async def test_continuation_before_output_replaces_unspoken_request():
    c, d = _setup()
    _feed(c)
    first = (await _poll(c))[0]
    d.relation = "continuation"
    _feed(c, "on my left", segment=1, now=1.1)
    actions = await _poll(c, 1.1)
    assert [a.kind for a in actions] == ["supersede", "dispatch"]
    assert actions[0].request_id == first.request_id
    assert actions[1].text == "Open the panel\non my left"
    assert c.response("a", "session", first.request_id, "done") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stage,effects", [("playing", False), ("preparing", True)])
async def test_playback_or_effectful_continuation_never_replays_original_query(stage, effects):
    c, d = _setup()
    _feed(c, "Send the report")
    first = (await _poll(c))[0]
    c.response("a", "session", first.request_id, stage, effects_started=effects)
    d.relation = "continuation"
    _feed(c, "also include the appendix", segment=1, now=1.1)
    actions = await _poll(c, 1.1)
    assert [a.kind for a in actions] == ["interrupt", "dispatch"]
    assert actions[-1].text == "also include the appendix"
    assert actions[-1].previous_text == "Send the report"
    assert actions[-1].previous_request_id == first.request_id


@pytest.mark.asyncio
async def test_independent_query_queues_and_only_dispatches_after_output_done():
    c, _ = _setup()
    _feed(c)
    first = (await _poll(c))[0]
    _feed(c, "What time is it?", segment=1, now=1.1)
    queued = (await _poll(c, 1.1))[0]
    assert queued.kind == "queue"
    assert c.response("a", "session", "unrelated", "done") == []
    dispatched = c.response("a", "session", first.request_id, "done")
    assert dispatched[0].request_id == queued.request_id
    assert dispatched[0].text == "What time is it?"


@pytest.mark.asyncio
async def test_independent_interrupt_policy_is_application_controlled():
    c, _ = _setup(independent="interrupt")
    _feed(c)
    await _poll(c)
    _feed(c, "What time is it?", segment=1, now=1.1)
    assert [a.kind for a in await _poll(c, 1.1)] == ["interrupt", "dispatch"]


@pytest.mark.asyncio
async def test_continuation_of_queued_query_replaces_only_that_queued_query():
    c, d = _setup()
    _feed(c)
    first = (await _poll(c))[0]
    _feed(c, "Find a route to the station", segment=1, now=1.1)
    queued = (await _poll(c, 1.1))[0]
    d.relation = "continuation"
    _feed(c, "by bicycle", segment=2, now=1.2)
    actions = await _poll(c, 1.2)
    assert [a.kind for a in actions] == ["supersede", "queue"]
    assert actions[0].request_id == queued.request_id
    result = c.response("a", "session", first.request_id, "done")
    assert result[0].text == "Find a route to the station\nby bicycle"


@pytest.mark.asyncio
async def test_completed_query_followup_window_does_not_merge_old_requests():
    c, d = _setup(followup_s=2)
    _feed(c)
    first = (await _poll(c))[0]
    c.response("a", "session", first.request_id, "done")
    d.relation = "continuation"
    _feed(c, "show the weather", segment=1, now=60)
    result = await _poll(c, 60)
    assert result[0].text == "show the weather"
    assert result[0].previous_request_id == ""


@pytest.mark.asyncio
async def test_quick_followup_window_starts_after_a_long_response_completes():
    c, d = _setup(followup_s=2)
    _feed(c)
    first = (await _poll(c))[0]
    c.response("a", "session", first.request_id, "done", now=60)
    d.relation = "continuation"
    _feed(c, "and use the dark theme", segment=1, now=60.1)
    result = await _poll(c, 60.1)
    assert result[0].previous_request_id == first.request_id
    assert result[0].text == "and use the dark theme"
    assert d.calls[-1][0]["gap_s"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_stop_cancels_queued_and_active_requests():
    c, d = _setup()
    _feed(c)
    first = (await _poll(c))[0]
    _feed(c, "What's the forecast?", segment=1, now=2)
    queued = (await _poll(c, 2))[0]
    d.choice = "stop"
    _feed(c, "stop", segment=2, now=3)
    actions = await _poll(c, 3)
    assert [(a.kind, a.request_id) for a in actions] == [
        ("cancel", queued.request_id),
        ("interrupt", first.request_id),
    ]
    assert c.response("a", "session", first.request_id, "done") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError(), ValueError("invalid response")])
async def test_failure_holds_without_releasing_or_spinning_on_unchanged_input(error):
    c, d = _setup()
    d.error = error
    _feed(c)
    assert (await _poll(c))[0].reason.startswith("decision_failed:")
    assert await _poll(c, 10) == []
    assert len(d.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("choice,confidence", [("hold", 0.99), ("ready", 0.5)])
async def test_uncertain_disposition_holds_without_dispatch(choice, confidence):
    c, d = _setup()
    d.choice, d.confidence = choice, confidence
    _feed(c)
    assert (await _poll(c))[0].kind == "hold"


@pytest.mark.asyncio
async def test_uncertain_relation_does_not_guess_merge_or_separate():
    c, d = _setup()
    _feed(c)
    await _poll(c)
    d.relation = "uncertain"
    _feed(c, "the other one", segment=1, now=2)
    assert (await _poll(c, 2))[0].reason == "uncertain_followup_relation"


@pytest.mark.asyncio
async def test_revision_during_inference_discards_result_and_evaluates_latest_snapshot():
    c, d = _setup()
    d.release = asyncio.Event()
    _feed(c)
    task = asyncio.create_task(_poll(c))
    await d.entered.wait()
    _feed(c, "Close the panel", revision=1, now=2)
    assert await _poll(c, 2) == []
    d.release.set()
    assert await task == []
    assert (await _poll(c, 2))[0].text == "Close the panel"


@pytest.mark.asyncio
async def test_output_starts_while_classifying_prevents_unspoken_replay():
    c, d = _setup()
    _feed(c)
    first = (await _poll(c))[0]
    d.relation = "continuation"
    d.entered.clear()
    d.release = asyncio.Event()
    _feed(c, "a bit larger", segment=1, now=2)
    task = asyncio.create_task(_poll(c, 2))
    await d.entered.wait()
    c.response("a", "session", first.request_id, "playing")
    d.release.set()
    assert await task == []
    result = await _poll(c, 2)
    assert [a.kind for a in result] == ["interrupt", "dispatch"]
    assert result[-1].text == "a bit larger"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["reset", "replace", "close"])
async def test_departure_track_replacement_or_close_invalidates_inflight_decision(operation):
    c, d = _setup()
    d.release = asyncio.Event()
    _feed(c)
    task = asyncio.create_task(_poll(c))
    await d.entered.wait()
    if operation == "reset":
        c.reset("a", "session")
    elif operation == "replace":
        c.open("a", "next-session")
    else:
        c.close()
    d.release.set()
    assert await task == []
    with pytest.raises(ValueError):
        _feed(c)


@pytest.mark.asyncio
async def test_two_participants_do_not_share_context_or_overlap_clef_calls():
    c, d = _setup()
    c.open("b", "other")
    d.release = asyncio.Event()
    _feed(c)
    task = asyncio.create_task(_poll(c))
    await d.entered.wait()
    _feed(c, "Read my notes", who="b", epoch="other")
    assert await _poll(c, who="b", epoch="other") == []
    d.release.set()
    await task
    result = await _poll(c, who="b", epoch="other")
    assert result[0].text == "Read my notes"
    assert d.calls[-1][0]["previous_request"] == ""


@pytest.mark.asyncio
async def test_cancellation_drains_request_and_permits_explicit_retry():
    c, d = _setup()
    d.release = asyncio.Event()
    _feed(c)
    task = asyncio.create_task(_poll(c))
    await d.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    d.release.set()
    assert (await _poll(c))[0].kind == "dispatch"


@pytest.mark.asyncio
async def test_overflow_invalidates_previous_candidate_without_truncating_and_releasing_it():
    c, d = _setup(max_chars=20)
    _feed(c, "Open")
    actions = _feed(c, "x" * 100, revision=1)
    assert actions[0].reason == "overflow_requires_reset"
    assert await _poll(c) == []
    assert not d.calls


@pytest.mark.asyncio
async def test_queue_capacity_retains_candidate_until_output_drains():
    c, _ = _setup(max_queue=1)
    _feed(c)
    first = (await _poll(c))[0]
    _feed(c, "Second query", segment=1, now=2)
    await _poll(c, 2)
    _feed(c, "Third query", segment=2, now=3)
    assert (await _poll(c, 3))[0].reason == "queue_capacity"
    c.response("a", "session", first.request_id, "done")
    assert (await _poll(c, 3))[0].kind == "queue"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stable_s": math.nan},
        {"min_confidence": 1.1},
        {"max_sessions": 0},
        {"max_chars": True},
        {"independent": "guess"},
        {"decision_timeout_s": 0},
    ],
)
def test_invalid_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        _Policy(**kwargs)


def test_no_implicit_session_creation_and_capacity_is_bounded():
    c, _ = _setup(max_sessions=1)
    with pytest.raises(ValueError):
        c.open("b", "other")
    with pytest.raises(ValueError):
        _feed(c, who="missing")


@pytest.mark.asyncio
async def test_replacement_lifecycle_does_not_reuse_request_ids():
    c, _ = _setup()
    _feed(c)
    old = (await _poll(c))[0]
    c.reset("a", "session")
    c.open("a", "session")
    _feed(c)
    new = (await _poll(c))[0]
    assert new.request_id != old.request_id
    assert c.response("a", "session", old.request_id, "done") == []


@pytest.mark.asyncio
async def test_invalid_classifier_label_holds_and_unexpected_programming_error_is_visible():
    c, d = _setup()
    d.choice = "unknown"
    _feed(c)
    assert (await _poll(c))[0].reason == "decision_failed:ValueError"
    d.error = AttributeError("programming error")
    _feed(c, revision=1, text="Close the panel", now=2)
    with pytest.raises(AttributeError, match="programming error"):
        await _poll(c, 2)
