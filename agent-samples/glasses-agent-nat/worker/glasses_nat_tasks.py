# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task implementations used by registered glasses-agent NAT functions."""
from __future__ import annotations

import json
import logging
import re
import string
import time
from typing import Any

import httpx

from glasses_nat_schemas import AnalyzeRecordingInput
from glasses_nat_schemas import AnalyzeRecordingOutput
from glasses_nat_schemas import CondenseObservationsInput
from glasses_nat_schemas import CondenseObservationsOutput
from glasses_nat_schemas import GuidanceStepOutput

log = logging.getLogger("glasses_agent_nat.tasks")

_FILLER = frozenset({
    "next", "okay", "ok", "yeah", "yes", "no", "and", "then",
    "um", "uh", "hmm", "right", "sure", "alright",
})


def extract_json(text: str) -> str | None:
    depth, start, in_string, escape = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    return None


def _is_filler(text: str) -> bool:
    words = [w.strip(string.punctuation).lower() for w in text.split()]
    meaningful = [w for w in words if w]
    return bool(meaningful) and all(w in _FILLER for w in meaningful)


def _str_payload(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("result") or value.get("text") or next(iter(value.values()), "")
    return str(value).strip() if value else ""


def _payload(value: Any) -> dict[str, Any] | list[Any] | str | None:
    if value is None or isinstance(value, dict | list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    structured = getattr(value, "structured_content", None)
    if structured is not None:
        return structured
    content = getattr(value, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except Exception:
                return text
    return str(value)


async def _post_chat(
    *,
    base_url: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout: float,
    extra_body: dict[str, Any] | None = None,
) -> str:
    body: dict[str, Any] = {
        "model": "llm",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_body:
        body.update(extra_body)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(base_url.rstrip("/") + "/v1/chat/completions", json=body)
    if resp.is_error:
        raise RuntimeError(f"LLM {resp.status_code}: {resp.text[:300]}")
    return (resp.json()["choices"][0]["message"].get("content") or "").strip()


async def describe_current_view_impl(
    *,
    participant_id: str,
    question: str,
    get_latest_frame,
    list_live_participants,
    ask_image,
) -> str:
    async def _latest(pid: str) -> dict[str, Any] | None:
        frame = _payload(await get_latest_frame.ainvoke({"participant_id": pid}))
        return frame if isinstance(frame, dict) and frame.get("path") else None

    frame = await _latest(participant_id)
    if frame is None:
        participants = _payload(await list_live_participants.ainvoke({}))
        if isinstance(participants, list):
            for pid in participants:
                pid_str = str(pid)
                if pid_str and pid_str != participant_id:
                    frame = await _latest(pid_str)
                    if frame is not None:
                        break

    if not isinstance(frame, dict) or "path" not in frame:
        return "I don't have a live camera frame yet."
    image_path = str(frame.get("path") or "")
    if not image_path:
        return "I don't have a live camera frame yet."
    prompt = question.strip() or "Describe what you see in this image in 1-2 concise sentences."
    result = await ask_image.ainvoke({"question": prompt, "image_path": image_path})
    text = _str_payload(result)
    return text or "I can't describe the current view yet."


async def analyze_recording_impl(
    request: AnalyzeRecordingInput,
    *,
    agent_llm_server: str,
) -> AnalyzeRecordingOutput:
    frames = request.frames
    if not frames:
        return AnalyzeRecordingOutput()

    if len(frames) > 40:
        step = len(frames) / 30.0
        idxs = sorted({0, len(frames) - 1} | {int(i * step) for i in range(1, 30)})
        shown = [frames[i] for i in idxs]
    else:
        shown = frames

    meaningful_notes = [v for v in request.voice_notes if not _is_filler(v.text)]
    timeline: list[tuple[int, str]] = []
    for f in shown:
        rel = (f.timestamp_us - request.started_at_us) / 1_000_000
        timeline.append((
            f.timestamp_us,
            f"[Frame {f.frame_idx + 1}/{len(frames)} | +{rel:.1f}s]\n{f.description}",
        ))
    for v in meaningful_notes:
        rel = (v.timestamp_us - request.started_at_us) / 1_000_000
        timeline.append((v.timestamp_us, f"[Voice +{rel:.1f}s] \"{v.text}\""))
    timeline.sort(key=lambda x: x[0])
    desc_block = "\n\n".join(entry for _, entry in timeline)

    voice_guidance = ""
    if meaningful_notes:
        voice_guidance = (
            "\n[Voice +Xs] entries are the user's spoken narration.\n"
            "[Frame N/M | +Xs] entries are VLM descriptions of what the camera saw.\n\n"
            "CRITICAL RULES:\n"
            "  - Voice notes are the ONLY source of steps. "
            "NEVER create a step from a frame description alone.\n"
            "  - Strip the '+Xs' timing prefix from all output.\n"
            "  - MERGE consecutive notes that together describe one action.\n"
            "  - Each physical object should appear in at most one step unless "
            "explicitly picked up a second time.\n"
            "  - Use nearby frames only to add spatial detail to a voice-defined step.\n"
        )

    messages = [
        {
            "role": "system",
            "content": (
                f"You are analyzing a recorded demonstration: {request.name!r}\n\n"
                "Timeline is in ascending time order (smaller +Ns = earlier).\n"
                f"{voice_guidance}"
                "\nYOUR TASK:\n"
                "1. Steps MUST be in chronological order.\n"
                "2. Each step = one complete action. Merge action fragments.\n"
                "3. Output only steps grounded in voice notes.\n\n"
                "OUTPUT - a single JSON object, nothing else:\n"
                '{"overview": "one sentence", "steps": ["step 1", "step 2", ...]}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Demo: {request.name!r}\n\n"
                f"Timeline ({len(timeline)} entries):\n\n"
                f"{desc_block}\n\n"
                "Output the JSON."
            ),
        },
    ]
    raw = await _post_chat(
        base_url=agent_llm_server,
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
        timeout=90.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    json_str = extract_json(content or raw)
    if not json_str:
        log.warning("analysis: JSON parse failed: %s", raw[:300])
        return AnalyzeRecordingOutput()
    try:
        obj = json.loads(json_str)
    except Exception:
        log.warning("analysis: invalid JSON: %s", json_str[:300])
        return AnalyzeRecordingOutput()
    return AnalyzeRecordingOutput(
        overview=str(obj.get("overview", "")).strip(),
        steps=[str(s).strip() for s in obj.get("steps", []) if str(s).strip()],
    )


async def condense_observations_impl(
    request: CondenseObservationsInput,
    *,
    llm_server: str,
) -> CondenseObservationsOutput:
    if not request.observations:
        return CondenseObservationsOutput()

    obs_text = "\n".join(
        f"  [{time.strftime('%H:%M:%S', time.localtime(o.timestamp_us / 1e6))}|{o.timestamp_us}]  "
        f"{o.description}"
        for o in request.observations[-20:]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a scene context summarizer for smart glasses. "
                "Given a timeline of camera observations (each tagged with "
                "[HH:MM:SS|timestamp_us]), output ONLY valid JSON in this shape:\n"
                '{"overview":"1-2 sentence scene summary","events":['
                '{"timestamp_us":<int>,"time":"HH:MM:SS","description":"brief event"}]}\n'
                "Include only the 3-6 most significant events. "
                "Use the exact timestamp_us values from the input."
            ),
        },
        {"role": "user", "content": f"Observations:\n{obs_text}"},
    ]
    raw = await _post_chat(
        base_url=llm_server,
        messages=messages,
        max_tokens=256,
        temperature=0.1,
        timeout=20.0,
    )
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        structured = json.loads(raw)
    except Exception:
        log.warning("condenser output not JSON - storing as plain text")
        structured = {"overview": raw, "events": []}

    overview = str(structured.get("overview", "")).strip()
    events = structured.get("events", [])
    lines = [overview] if overview else []
    for ev in events if isinstance(events, list) else []:
        if not isinstance(ev, dict):
            continue
        ts_us = ev.get("timestamp_us", 0)
        hms = ev.get("time", "")
        desc = ev.get("description", "")
        lines.append(f"  [{hms} | {ts_us} us] {desc}")
    return CondenseObservationsOutput(
        overview=overview,
        events=events if isinstance(events, list) else [],
        summary_text="\n".join(lines),
    )


async def check_guidance_step_complete_impl(
    *,
    participant_id: str,
    instruction: str,
    get_latest_frame,
    ask_image,
) -> GuidanceStepOutput:
    frame = _payload(await get_latest_frame.ainvoke({"participant_id": participant_id}))
    if not isinstance(frame, dict) or "path" not in frame:
        return GuidanceStepOutput(raw="NO")
    image_path = str(frame.get("path") or "")
    if not image_path:
        return GuidanceStepOutput(raw="NO")
    result = await ask_image.ainvoke({
        "question": (
            f"Target step: {instruction}\n"
            "Look only at the current image. Is the target step visibly complete right now?\n"
            "Answer YES only if all required objects and actions in the target step are clearly visible. "
            "If the step mentions holding a controller, a controller must be visibly held. "
            "If any required object/action is missing or uncertain, answer NO and briefly say what is missing.\n"
            "Format: YES - <brief evidence> or NO - <brief missing evidence>."
        ),
        "image_path": image_path,
    })
    raw = _str_payload(result)
    return GuidanceStepOutput(
        completed=raw.strip().upper().startswith("YES"),
        raw=raw,
        image_path=image_path,
    )
