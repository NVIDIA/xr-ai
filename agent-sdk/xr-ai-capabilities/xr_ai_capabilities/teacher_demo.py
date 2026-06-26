# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable "teacher demonstration analysis" capability.

Turns a recording (frames + voice notes) into an overview plus an ordered list
of steps, then distills each step into per-step requirements and structured key
info used downstream for live student monitoring. Framework-agnostic over an
``xr_ai_models.LLMService`` — no nvidia-nat, langchain, or pydantic.
"""
from __future__ import annotations

import json
import logging
import re
import string
from dataclasses import dataclass, field
from typing import Any

from xr_ai_models import ChatMessage, LLMService

from ._textutil import extract_json

log = logging.getLogger("xr_ai_capabilities.teacher_demo")


@dataclass
class RecordingFrame:
    frame_idx: int
    timestamp_us: int
    image_path: str
    description: str


@dataclass
class VoiceNote:
    timestamp_us: int
    text: str


@dataclass
class AnalysisResult:
    overview: str = ""
    steps: list[str] = field(default_factory=list)


@dataclass
class StepKeyInfo:
    objects: list[str] = field(default_factory=list)
    action: str = ""
    position: str = ""
    target_state: str = ""
    ignore: list[str] = field(default_factory=list)


_FILLER = frozenset({
    "next", "okay", "ok", "yeah", "yes", "no", "and", "then",
    "um", "uh", "hmm", "right", "sure", "alright",
})

_STEP_NUMBER_WORDS = {
    "one", "first", "two", "second", "three", "third", "four", "fourth",
    "five", "fifth", "six", "sixth", "seven", "seventh", "eight", "eighth",
    "nine", "ninth", "ten", "tenth",
}


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


def _fallback_steps_from_frames(name: str, frames: list[RecordingFrame]) -> list[str]:
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


# Verbs whose direct object is the thing the student must PLACE/manipulate this
# step (good="put a mouse next to the case" → object is "a mouse"). The noun
# phrase AFTER the relational preposition is the anchor/reference, already in
# view from an earlier step — it must become position, not the object to verify.
_PLACEMENT_VERBS = (
    "put", "place", "set", "lay", "position", "move", "drop", "rest",
    "insert", "attach", "mount", "hang", "stack", "add", "connect", "plug",
)
# Relational prepositions that introduce the anchor object in a placement step.
_ANCHOR_PREPOSITIONS = (
    "next to", "beside", "near", "on top of", "on", "onto", "in", "into",
    "inside", "under", "underneath", "behind", "in front of", "to the left of",
    "to the right of", "above", "below", "against",
)


@dataclass
class _RelationalStep:
    """A parsed "put X <prep> Y" instruction: the placed object and the anchor."""
    placed_object: str = ""
    anchor_phrase: str = ""


def _parse_relational_placement(instruction: str) -> _RelationalStep:
    """Split a placement instruction into the PLACED object and the anchor.

    For "put a mouse next to the AirPod case" the placed object is the verb's
    direct object ("mouse") and the anchor is "next to the AirPod case". The
    anchor object is already present from an earlier step, so keying step
    completion on it auto-passes without the student doing anything; this parse
    lets downstream derivation make the PLACED object primary and demote the
    anchor to position. Returns empty fields when the instruction is not a
    recognizable relational placement.
    """
    text = instruction.strip().rstrip(".").strip()
    if not text:
        return _RelationalStep()
    lower = text.lower()
    tokens = lower.split()
    if not tokens or tokens[0] not in _PLACEMENT_VERBS:
        return _RelationalStep()
    # Find the earliest relational preposition; everything before it (after the
    # verb) is the placed object, everything from it on is the anchor phrase.
    best_at = len(text)
    best_prep = ""
    for prep in _ANCHOR_PREPOSITIONS:
        m = re.search(rf"\b{re.escape(prep)}\b", lower)
        if m and m.start() < best_at:
            best_at, best_prep = m.start(), prep
    if not best_prep:
        return _RelationalStep()
    verb_end = len(tokens[0])
    placed = text[verb_end:best_at].strip()
    anchor = text[best_at:].strip()
    placed = re.sub(r"^(a|an|the)\s+", "", placed, flags=re.IGNORECASE).strip()
    if not placed or not anchor:
        return _RelationalStep()
    return _RelationalStep(placed_object=placed, anchor_phrase=anchor)


def _object_names_token(objects: list[str], placed: str) -> bool:
    """True when some entry in *objects* shares an identifying token with the
    placed object — i.e. the derivation already named the right object."""
    placed_tokens = {
        t for t in re.findall(r"[a-z0-9]+", placed.lower())
        if len(t) > 1 and t not in _EVIDENCE_STOPWORDS
    }
    if not placed_tokens:
        return True
    for obj in objects:
        obj_tokens = {t for t in re.findall(r"[a-z0-9]+", obj.lower()) if len(t) > 1}
        if placed_tokens & obj_tokens:
            return True
    return False


async def analyze_recording(
    name: str,
    started_at_us: int,
    frames: list[RecordingFrame],
    voice_notes: list[VoiceNote],
    *,
    llm: LLMService,
) -> AnalysisResult:
    if not frames:
        return AnalysisResult()

    if len(frames) > 40:
        step = len(frames) / 30.0
        idxs = sorted({0, len(frames) - 1} | {int(i * step) for i in range(1, 30)})
        shown = [frames[i] for i in idxs]
    else:
        shown = frames

    meaningful_notes = [v for v in voice_notes if not _is_filler(v.text)]
    actionable_notes = [
        v for v in meaningful_notes if not _is_step_marker(v.text)
    ]
    timeline: list[tuple[int, str]] = []
    for f in shown:
        rel = (f.timestamp_us - started_at_us) / 1_000_000
        timeline.append((
            f.timestamp_us,
            f"[Frame {f.frame_idx + 1}/{len(frames)} | +{rel:.1f}s]\n{f.description}",
        ))
    for v in meaningful_notes:
        rel = (v.timestamp_us - started_at_us) / 1_000_000
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
                f"You are analyzing a recorded demonstration: {name!r}\n\n"
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
                f"Demo: {name!r}\n\n"
                f"Timeline ({len(timeline)} entries):\n\n"
                f"{desc_block}\n\n"
                "Output the JSON."
            ),
        },
    ]
    resp = await llm.chat(
        [ChatMessage(role=m["role"], content=m["content"]) for m in messages],
        max_tokens=1024, temperature=0.0, enable_thinking=False,
    )
    raw = (resp.content or "").strip()
    content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    json_str = extract_json(content or raw)
    if not json_str:
        log.warning("analysis: JSON parse failed: %s", raw[:300])
        return AnalysisResult()
    try:
        obj = json.loads(json_str)
    except Exception:
        log.warning("analysis: invalid JSON: %s", json_str[:300])
        return AnalysisResult()
    cleaned_steps = _clean_analyzed_steps(
        [str(s).strip() for s in obj.get("steps", []) if str(s).strip()]
    )
    evidence_text = " ".join(
        [name, *(frame.description for frame in frames),
         *(note.text for note in meaningful_notes)]
    )
    filtered_steps = _filter_steps_by_evidence(cleaned_steps, evidence_text)
    if not filtered_steps and not actionable_notes:
        filtered_steps = _fallback_steps_from_frames(name, frames)
    return AnalysisResult(
        overview=str(obj.get("overview", "")).strip(),
        steps=filtered_steps,
    )


async def derive_step_requirements(
    instruction: str,
    teacher_caption: str = "",
    *,
    llm: LLMService,
) -> list[str]:
    """Turn one step instruction into a small atomic checklist.

    The selected teacher frame is the visual authority. The instruction
    names the step, and the teacher caption is a hint about what the
    frame shows.
    """
    instruction = instruction.strip()
    if not instruction:
        return []

    hint = teacher_caption.strip()
    hint_block = f"\nTEACHER FRAME CAPTION (visual authority): {hint}" if hint else ""

    messages = [
        {
            "role": "system",
            "content": (
                "You produce an atomic visual checklist for one step in a "
                "demonstration. Output 1-4 short atoms (3-7 words each). "
                "Each atom must be VISUALLY CHECKABLE from a single image — "
                "a human looking at one photo should be able to mark it "
                "true or false.\n\n"
                "Rules:\n"
                "  - For a placement step (\"put/place X next to/on Y\"), the "
                "object the student must PLACE is X (the verb's object); Y is "
                "the anchor, already present from an earlier step. The checklist "
                "MUST verify X is now present and placed relative to Y — never "
                "make the checklist solely about Y.\n"
                "  - The selected teacher frame refines the appearance of the "
                "task object. If the instruction conflicts with the teacher "
                "frame caption, prefer the caption for color/material/state — "
                "but do NOT let it swap the checklist onto the anchor object.\n"
                "  - Do NOT enumerate background details; include only the "
                "task-relevant object and end-state.\n"
                "  - Prefer concrete object + state phrases: "
                '"headset on head", "lid closed", "switch in up position".\n'
                "  - Avoid action verbs in the past tense; describe the END "
                'STATE (good: "screw inserted in hole"; bad: "insert screw").\n\n'
                "OUTPUT - a single JSON object, nothing else:\n"
                '{"requirements": ["atom 1", "atom 2", ...]}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"INSTRUCTION: {instruction}{hint_block}\n\nOutput the JSON."
            ),
        },
    ]
    try:
        resp = await llm.chat(
            [ChatMessage(role=m["role"], content=m["content"]) for m in messages],
            max_tokens=256, temperature=0.0, enable_thinking=False,
        )
        raw = (resp.content or "").strip()
    except Exception:
        log.exception("derive_step_requirements: LLM call failed")
        return []
    content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    json_str = extract_json(content or raw)
    if not json_str:
        log.warning("derive_step_requirements: JSON parse failed: %s", raw[:300])
        return []
    try:
        obj = json.loads(json_str)
    except Exception:
        log.warning("derive_step_requirements: invalid JSON: %s", json_str[:300])
        return []
    raw_reqs = obj.get("requirements", [])
    if not isinstance(raw_reqs, list):
        return []
    reqs: list[str] = []
    for entry in raw_reqs:
        text = str(entry).strip()
        if not text:
            continue
        reqs.append(text)
        if len(reqs) >= 4:
            break
    return reqs


async def derive_step_key_info(
    instruction: str,
    teacher_caption: str = "",
    requirements: list[str] | None = None,
    *,
    llm: LLMService,
) -> StepKeyInfo:
    """Distill one step's instruction + teacher caption into structured key info.

    Output fields name only what defines the step — the objects, the action,
    the spatial placement, and the end-state to verify — plus an ``ignore``
    list of details that must NOT affect the student-monitoring verdict
    (background, lighting, camera angle, …). This is what makes monitoring
    tolerant of irrelevant differences.
    """
    requirements = requirements or []
    instruction = instruction.strip()
    if not instruction:
        return StepKeyInfo()
    hint = teacher_caption.strip()
    reqs = [r.strip() for r in requirements if r and r.strip()]
    hint_block = f"\nTEACHER FRAME CAPTION (visual authority): {hint}" if hint else ""
    req_block = ("\nDERIVED CHECKS: " + "; ".join(reqs)) if reqs else ""

    messages = [
        {
            "role": "system",
            "content": (
                "You distill ONE demonstration step into the few facts that "
                "define it, so a vision model can later check a student "
                "without being distracted by irrelevant differences.\n\n"
                "Output ONLY this JSON object, nothing else:\n"
                "{\n"
                '  "objects": ["the task-relevant object(s), by name + color/shape if helpful"],\n'
                '  "action": "the single action performed, imperative (e.g. \\"place on head\\")",\n'
                '  "position": "the spatial relationship/placement that matters (e.g. \\"strap around back of head\\")",\n'
                '  "target_state": "the visible end-state that means this step is done",\n'
                '  "ignore": ["irrelevant details a checker must disregard"]\n'
                "}\n\n"
                "Rules:\n"
                "  - For a placement step (\"put/place X next to/on Y\"): the "
                "FIRST object MUST be X, the thing being placed (the verb's "
                "direct object). Y is the anchor — already placed earlier — so "
                "put \"next to Y\" in position, NOT in objects. Never make X the "
                "anchor.\n"
                "  - Prefer the teacher frame caption to refine X's "
                "color/material/state; do NOT let it replace X with the anchor.\n"
                "  - objects: 1-3 items, only task-relevant ones.\n"
                "  - target_state describes what is VISIBLE when done, not the action verb.\n"
                "  - ALWAYS include at least: background, lighting, camera angle, "
                "and the wearer's clothing in ignore.\n"
            ),
        },
        {
            "role": "user",
            "content": f"INSTRUCTION: {instruction}{hint_block}{req_block}\n\nOutput the JSON.",
        },
    ]
    try:
        resp = await llm.chat(
            [ChatMessage(role=m["role"], content=m["content"]) for m in messages],
            max_tokens=256, temperature=0.0, enable_thinking=False,
        )
        raw = (resp.content or "").strip()
    except Exception:
        log.exception("derive_step_key_info: LLM call failed")
        return StepKeyInfo()
    content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    json_str = extract_json(content or raw)
    if not json_str:
        log.warning("derive_step_key_info: JSON parse failed: %s", raw[:300])
        return StepKeyInfo()
    try:
        obj = json.loads(json_str)
    except Exception:
        log.warning("derive_step_key_info: invalid JSON: %s", json_str[:300])
        return StepKeyInfo()
    if not isinstance(obj, dict):
        return StepKeyInfo()

    def _strlist(value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        out = [str(v).strip() for v in value if str(v).strip()]
        return out[:limit]

    ignore = _strlist(obj.get("ignore"), 8)
    for default in ("background", "lighting", "camera angle"):
        if default not in [i.lower() for i in ignore]:
            ignore.append(default)

    objects = _strlist(obj.get("objects"), 3)
    position = str(obj.get("position", "")).strip()

    # Deterministic backstop for relational placement steps. The LLM is told to
    # make the placed object primary, but it (and the teacher caption it is told
    # to trust) reliably drifts onto the anchor for "put X next to Y" — yielding
    # objects=[Y] only, so live monitoring verifies the anchor that is already in
    # view and auto-passes. Force the placed object X to be the first key object
    # and keep the anchor as position, so completion grounds on X actually
    # appearing in the student's frame.
    rel = _parse_relational_placement(instruction)
    if rel.placed_object and not _object_names_token(objects, rel.placed_object):
        anchor_only = [o for o in objects if not _object_names_token([o], rel.placed_object)
                       and _object_names_token([o], rel.anchor_phrase)]
        kept = [o for o in objects if o not in anchor_only]
        objects = [rel.placed_object, *kept][:3]
        if not position:
            position = rel.anchor_phrase

    return StepKeyInfo(
        objects=objects,
        action=str(obj.get("action", "")).strip(),
        position=position,
        target_state=str(obj.get("target_state", "")).strip(),
        ignore=ignore,
    )
