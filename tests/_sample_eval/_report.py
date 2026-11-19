# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech delivery, response timing and sender/backlog measurements."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Speaker STT timestamps the encoder hop, up to 160 ms before speech begins.
_SOURCE_LEAD_US = 160000


def _distribution(values: list[float]) -> dict:
    return (
        {"n": len(values), "median": statistics.median(values), "worst": max(values)}
        if values
        else {
            "n": 0,
            "median": "unmeasured",
            "worst": "unmeasured",
        }
    )


def _turns(rows: list[dict], duration: float) -> list[dict]:
    starts = [row for row in rows if row["event"] == "speech_input"]
    ends = [row for row in rows if row["event"] == "speech_input_end"]
    streams = defaultdict(list)
    for row in rows:
        if row["event"] == "runtime" and row["topic"] == "voice.output":
            message = row["message"]
            if message.get("request_id") is not None and message.get("response_id") is not None:
                streams[(row.get("source"), message["request_id"], message["response_id"])].append(row)
    completions = defaultdict(list)
    for (_source, request, _response), chunks in streams.items():
        finals = [chunk["elapsed"] for chunk in chunks if chunk["message"].get("final")]
        completions[request].append(finals[-1] if finals else None)

    assigned = {}
    for row in rows:
        if row["event"] != "request":
            continue
        pts = row["message"].get("timestamp_us")
        candidates = [
            index
            for index, start in enumerate(starts)
            if isinstance(pts, int)
            and start.get("audio_start_us") is not None
            and start["audio_start_us"] - _SOURCE_LEAD_US <= pts < start["audio_end_us"]
        ]
        if len(candidates) == 1:
            assigned[row["request_id"]] = candidates[0]
    turns = []
    for index, start in enumerate(starts):
        boundary = starts[index + 1]["elapsed"] if index + 1 < len(starts) else duration + 10
        chunk = [row for row in rows if start["elapsed"] <= row["elapsed"] < boundary]
        end = next((row["elapsed"] for row in ends if start["elapsed"] <= row["elapsed"] < boundary), None)
        request_ids = [key for key, value in assigned.items() if value == index]
        deliveries = [row for row in rows if row["event"] == "request" and row["request_id"] in request_ids]
        queries = [row["message"] for row in deliveries]
        replies = [completion for request in request_ids for completion in completions[request]]
        audio = [
            row["elapsed"]
            for row in chunk
            if row["event"] == "audio_out" and row["rms"] > 0.0001 and end is not None and row["elapsed"] >= end
        ]
        turns.append(
            {
                "spoken": start["text"],
                "start_s": start["elapsed"],
                "input_end_s": end,
                "queries": queries,
                "query_delivery_s": max(row["elapsed"] for row in deliveries) - end
                if deliveries and end is not None
                else None,
                "request_ids": request_ids,
                "scheduled_s": start.get("scheduled_s", "unmeasured"),
                "sender_start_lateness_s": max(0, start["elapsed"] - start["scheduled_s"])
                if "scheduled_s" in start
                else "unmeasured",
                "sender_end_lateness_s": next(
                    (
                        max(0, row["elapsed"] - row["scheduled_s"])
                        for row in ends
                        if row.get("scheduled_s") is not None and row["elapsed"] == end
                    ),
                    "unmeasured",
                ),
                "first_observed_audio_s": audio[0] - end if audio and end is not None else None,
                "completed_reply_text_s": max(replies) - end
                if end is not None and replies and None not in replies
                else None,
            }
        )
    return turns


def _write_report(directory: Path) -> dict:
    raw = (directory / "events.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    manifest = json.loads((directory / "manifest.json").read_text())
    scenario = json.loads((directory / "scenario.json").read_text())
    timed = [row for row in rows if row["elapsed"] is not None]
    turns = _turns(timed, scenario["duration_s"])
    planned = [event for event in scenario["events"] if "say" in event and event.get("handler", True)]
    expected_texts = {event["say"] for event in planned}
    expected = [turn for turn in turns if turn["spoken"] in expected_texts]
    finished = [row for row in rows if row["event"] == "run_finished"]
    completed = bool(
        finished and finished[-1]["status"] == "completed" and any(row["event"] == "session_end" for row in rows)
    )
    queues = [row["seconds"] for row in rows if row["event"] == "stt_queue"]
    enrolled = [
        row["elapsed"] for row in rows if row["event"] == "stt_event" and row["value"].get("kind") == "enrolled"
    ]
    sessions = [row for row in rows if row["event"] == "session_end"]
    lateness = sessions[-1]["max_sender_lateness_s"] if sessions else "unmeasured"
    measurement_end = next((row["elapsed"] for row in rows if row["event"] == "measurement_finished"), None)
    observed = bool(manifest.get("error_observer")) and measurement_end is not None
    errors = {
        component: sum(
            row["event"] == "error" and 0 <= row["elapsed"] <= measurement_end and row["component"] == component
            for row in timed
        ) if observed else "unmeasured"
        for component in ("model", "runtime")
    }
    error_total = sum(errors.values()) if observed else "unmeasured"
    matched = {request for turn in turns for request in turn["request_ids"]}
    requests = {row["request_id"] for row in rows if row["event"] == "request"}
    requests.update(
        row["message"]["request_id"]
        for row in rows
        if row["event"] == "runtime" and row["topic"] == "voice.output" and row["message"].get("request_id") is not None
    )
    report = {
        "manifest": manifest,
        "source_timestamp_lead_s": _SOURCE_LEAD_US / 1000000,
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution_status": "completed" if completed else "failed",
        "error_events": {**errors, "total": error_total},
        "first_observed_audio_s": _distribution(
            [
                turn["first_observed_audio_s"]
                for turn in turns
                if turn["queries"] and turn["first_observed_audio_s"] is not None
            ]
        ),
        "completed_reply_text_s": _distribution(
            [turn["completed_reply_text_s"] for turn in turns if turn["completed_reply_text_s"] is not None]
        ),
        "query_delivery_s": _distribution(
            [turn["query_delivery_s"] for turn in turns if turn["query_delivery_s"] is not None]
        ),
        "speaker_queue_s": _distribution(queues),
        "speaker_enrollment_s": enrolled,
        "speaker_drops": sum(row["event"] == "stt_drop" for row in rows) if queues else "not applicable",
        "speaker_resets": sum(row["event"] == "stt_reset" for row in rows) if queues else "not applicable",
        "handler_coverage": {"received": sum(bool(turn["queries"]) for turn in expected), "expected": len(planned)},
        "speech_duty_fraction": sum(row["duration_s"] for row in timed if row["event"] == "speech_input")
        / scenario["duration_s"],
        "unattributed_requests": len(requests - matched),
        "max_sender_lateness_s": lateness,
        "timing_comparable": completed and lateness <= 0.1 and error_total == 0
        if isinstance(lateness, (int, float)) and isinstance(error_total, int) else "unmeasured",
        "turns": turns,
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
