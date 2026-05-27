# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task implementations used by registered glasses-agent NAT functions."""
from __future__ import annotations

import json
import logging
import os
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

_STEP_NUMBER_WORDS = {
    "one", "first", "two", "second", "three", "third", "four", "fourth",
    "five", "fifth", "six", "sixth", "seven", "seventh", "eight", "eighth",
    "nine", "ninth", "ten", "tenth",
}


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


def _is_step_marker(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return False
    remainder = [
        token for token in tokens
        if token not in {"step", "is", "are"}
        and token not in _STEP_NUMBER_WORDS
        and not token.isdigit()
    ]
    return not remainder


def _clean_analyzed_steps(steps: list[str]) -> list[str]:
    cleaned: list[str] = []
    continuation_prefixes = (
        "and it ", "and put it ", "and place it ", "put it ",
        "place it ", "move it ", "to ", "on ", "onto ", "into ",
        "inside ", "under ", "over ", "beside ", "next to ", "with ",
    )
    for step in steps:
        text = step.strip()
        if not text or _is_step_marker(text):
            continue
        lower = text.lower().lstrip()
        word_count = len(re.findall(r"[a-z0-9]+", lower))
        if cleaned and word_count <= 6 and lower.startswith(continuation_prefixes):
            fragment = text.rstrip(".")
            fragment = fragment[:1].lower() + fragment[1:]
            cleaned[-1] = f"{cleaned[-1].rstrip('. ')} {fragment}."
        else:
            cleaned.append(text)
    return cleaned


_EVIDENCE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "be", "by", "for", "from", "in", "into", "is",
    "it", "next", "of", "on", "onto", "or", "over", "step", "task", "the",
    "then", "this", "to", "under", "with",
    "adjust", "adjusting", "align", "bring", "check", "complete", "fit",
    "get", "grab", "hold", "holding", "keep", "make", "move", "pick", "place",
    "position", "press", "put", "secure", "set", "take", "use", "wear",
    "wearing",
    *[str(i) for i in range(1, 21)],
    *_STEP_NUMBER_WORDS,
})


def _evidence_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _EVIDENCE_STOPWORDS and len(token) > 1
    }


def _filter_steps_by_evidence(steps: list[str], evidence_text: str) -> list[str]:
    evidence = _evidence_tokens(evidence_text)
    if not evidence:
        return steps
    filtered: list[str] = []
    for step in steps:
        required = _evidence_tokens(step)
        if not required:
            continue
        overlap = len(required & evidence)
        if overlap >= max(1, (len(required) + 1) // 2):
            filtered.append(step)
    return filtered


def _fallback_steps_from_frames(name: str, frames: list[FrameEntry]) -> list[str]:
    if not frames:
        return []
    name_tokens = _evidence_tokens(name)
    action_words = (
        "holding", "adjusting", "wearing", "putting", "placing", "picking",
        "pressing", "securing",
    )
    candidates: list[str] = []
    seen: set[frozenset[str]] = set()
    for frame in sorted(frames, key=lambda item: item.timestamp_us):
        sentences = re.split(r"(?<=[.!?])\s+", frame.description.strip())
        sentence = next(
            (s.strip() for s in sentences
             if any(word in s.lower() for word in action_words)
             and (not name_tokens or _evidence_tokens(s) & name_tokens)),
            "",
        )
        if not sentence:
            continue
        instruction = _sentence_to_instruction(sentence)
        key = frozenset(_evidence_tokens(instruction))
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(instruction)
        if len(candidates) >= 3:
            break
    return candidates


def _sentence_to_instruction(sentence: str) -> str:
    text = sentence.strip().rstrip(".")
    replacements = (
        (r"^(the person|the individual|they|he|she)\s+is\s+holding\b", "Hold"),
        (r"^(the person|the individual|they|he|she)\s+is\s+adjusting\b", "Adjust"),
        (r"^(the person|the individual|they|he|she)\s+is\s+wearing\b", "Wear"),
        (r"^(the person|the individual|they|he|she)\s+is\s+putting\b", "Put"),
        (r"^(the person|the individual|they|he|she)\s+is\s+picking\b", "Pick"),
        (r"^(the person|the individual|they|he|she)\s+are\s+holding\b", "Hold"),
        (r"^(the person|the individual|they|he|she)\s+are\s+adjusting\b", "Adjust"),
    )
    lower = text.lower()
    for pattern, replacement in replacements:
        if re.search(pattern, lower):
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            break
    return text[:1].upper() + text[1:] + "."


def _str_payload(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("result") or value.get("text") or next(iter(value.values()), "")
    return str(value).strip() if value else ""


def _coerce_completion_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "complete", "completed", "done"}:
            return True
        if normalized in {
            "false", "no", "n", "0", "incomplete", "not complete",
            "not completed", "wrong",
        }:
            return False
    return False


def _parse_completion_response(result: str) -> tuple[bool, str]:
    completed = False
    issue = ""
    json_str = extract_json(result)
    if json_str:
        try:
            obj = json.loads(json_str)
            completed = _coerce_completion_bool(
                obj.get("completed", obj.get("complete", obj.get("is_complete", False)))
            )
            issue = str(obj.get("issue", "")).strip()
            return completed, issue
        except Exception:
            return False, ""
    upper = result.upper()
    if upper.startswith("YES"):
        return True, ""
    if upper.startswith("NO"):
        issue = result[2:].lstrip(":;,. -").strip()
    return False, issue


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

    user_question = question.strip() or "What do you see?"
    prompt = (
        "Answer the wearer's question about what their camera currently sees. "
        "Reply in 1-2 short sentences. Speak in second person "
        "(\"You are looking at...\"). Do not volunteer lighting, atmosphere, "
        "aesthetics, or background design unless the user explicitly asks "
        "about them (e.g. 'is it bright?', 'how is the lighting?').\n\n"
        f"Question: {user_question}"
    )
    result = await ask_image.ainvoke({"question": prompt, "image_path": image_path})
    text = _str_payload(result).strip()
    if not text:
        return "I can't describe the current view yet."

    # Hard floor: Cosmos-Reason1-7B regularly ignores length hints in the
    # prompt and returns paragraph-length scene descriptions. Trim to the
    # first 2 sentences to keep the spoken reply within the 1-2 sentence
    # contract regardless of how the VLM behaves.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    trimmed = " ".join(s for s in sentences[:2] if s).strip()
    return trimmed or text


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
    actionable_notes = [
        v for v in meaningful_notes if not _is_step_marker(v.text)
    ]
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
    if actionable_notes:
        voice_guidance = (
            "\n[Voice +Xs] entries are the user's spoken narration.\n"
            "[Frame N/M | +Xs] entries are VLM descriptions of what the camera saw.\n\n"
            "CRITICAL RULES:\n"
            "  - Voice notes are the ONLY source of steps. "
            "NEVER create a step from a frame description alone.\n"
            "  - Strip the '+Xs' timing prefix from all output.\n"
            "  - MERGE consecutive notes that together describe one action.\n"
            "  - Notes that only say a step number are labels, not actions.\n"
            "  - Each physical object should appear in at most one step unless "
            "explicitly picked up a second time.\n"
            "  - Use nearby frames only to add spatial detail to a voice-defined step.\n"
        )
    elif meaningful_notes:
        voice_guidance = (
            "\n[Voice +Xs] entries are step labels only; they do not describe the actions.\n"
            "[Frame N/M | +Xs] entries are VLM descriptions of what the camera saw.\n\n"
            "CRITICAL RULES:\n"
            "  - Infer steps from visible frame changes around the step labels.\n"
            "  - Every step must mention only objects, colors, and actions visible in "
            "the frame descriptions or demo name.\n"
            "  - Do not introduce unrelated objects from memory or examples.\n"
            "  - If separate step boundaries are unclear, return one broad visible step.\n"
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
    cleaned_steps = _clean_analyzed_steps(
        [str(s).strip() for s in obj.get("steps", []) if str(s).strip()]
    )
    evidence_text = " ".join(
        [request.name, *(frame.description for frame in frames),
         *(note.text for note in meaningful_notes)]
    )
    filtered_steps = _filter_steps_by_evidence(cleaned_steps, evidence_text)
    if not filtered_steps and not actionable_notes:
        filtered_steps = _fallback_steps_from_frames(request.name, frames)
    return AnalyzeRecordingOutput(
        overview=str(obj.get("overview", "")).strip(),
        steps=filtered_steps,
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
    teacher_image_path: str = "",
    teacher_caption: str = "",
    get_latest_frame,
    ask_image,
    ask_frames=None,
) -> GuidanceStepOutput:
    frame = _payload(await get_latest_frame.ainvoke({"participant_id": participant_id}))
    if not isinstance(frame, dict) or "path" not in frame:
        return GuidanceStepOutput(issue="I cannot see a current frame.", raw="NO")
    image_path = str(frame.get("path") or "")
    if not image_path:
        return GuidanceStepOutput(issue="I cannot see a current frame.", raw="NO")

    raw = ""
    if ask_frames is not None and teacher_image_path and os.path.isfile(teacher_image_path):
        try:
            result = await ask_frames.ainvoke({
                "question": (
                    f"Task step instruction: {instruction}\n"
                    f"Teacher reference caption: {teacher_caption or 'not available'}\n\n"
                    "Image 1 is the teacher's completed state for this step. "
                    "Image 2 is the student's current live view. First identify the key "
                    "objects and constraints from the instruction and teacher caption. "
                    "Then compare those key objects only: identity, count, color, shape, "
                    "hand/action state, orientation, and spatial placement. Any explicit "
                    "requirement in the instruction or teacher caption is mandatory. "
                    "If the required color, object, shape, state, or placement differs, "
                    "completed must be false. "
                    "For example, a white circle does not complete a step requiring a blue circle.\n"
                    "Output only JSON: "
                    '{"completed": true/false, "issue": "brief visible mismatch if false"}'
                ),
                "image_paths": [teacher_image_path, image_path],
            })
            raw = _str_payload(result)
        except Exception as exc:
            log.warning("ask_frames guidance check failed; falling back to ask_image: %s", exc)

    completed, issue = _parse_completion_response(raw)
    if not raw or raw.startswith(("ask_frames:", "ask_image:")):
        result = await ask_image.ainvoke({
            "question": (
                f"Target step: {instruction}\n"
                f"Teacher completed-state caption: {teacher_caption or 'not available'}\n"
                "Extract the key objects and constraints from the target step and teacher "
                "caption, then look only at those objects in the current student image. "
                "Answer YES only if all required objects, colors, shapes, actions, states, "
                "orientations, and positions are clearly visible. Any explicit color, "
                "object, shape, state, or placement mismatch makes the answer NO; "
                "a white circle is wrong if the target says blue circle.\n"
                "Prefer JSON: "
                '{"completed": true/false, "issue": "brief visible mismatch if false"}'
            ),
            "image_path": image_path,
        })
        raw = _str_payload(result)
        completed, issue = _parse_completion_response(raw)

    return GuidanceStepOutput(
        completed=completed,
        issue=issue,
        raw=raw,
        image_path=image_path,
    )
