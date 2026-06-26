# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable "live guidance completion check" capability.

Decides whether the student's current camera frame completes the demonstrated
step — comparing against the teacher reference image and the step's key info,
then naming a concrete correction when it does not. Framework-agnostic over an
``xr_ai_models.VLMService`` plus an injected live-frame fetcher — no nvidia-nat,
langchain, or pydantic.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from xr_ai_models import VLMService

from ._textutil import extract_json

log = logging.getLogger("xr_ai_capabilities.agent_monitor")


@dataclass
class FrameRef:
    image_path: str
    timestamp_us: int


@dataclass
class StepCheck:
    requirement: str = ""
    visible: bool = False
    evidence: str = ""


@dataclass
class GuidanceCheckResult:
    completed: bool = False
    current_observation: str = ""
    checks: list[StepCheck] = field(default_factory=list)
    missing_or_mismatched: list[str] = field(default_factory=list)
    image_path: str = ""
    teacher_image_path: str = ""
    timestamp_us: int = 0
    issue: str = ""
    raw_vlm: str = ""


def _key_info_block(
    *,
    objects: list[str],
    action: str,
    position: str,
    target_state: str,
    ignore: list[str],
) -> str:
    """Render flat key-info args into a compact prompt block, or '' if empty."""
    lines: list[str] = []
    objs = [o for o in objects if o and o.strip()]
    if objs:
        lines.append(f"  Key objects: {', '.join(objs)}")
    if action.strip():
        lines.append(f"  Action: {action.strip()}")
    if position.strip():
        lines.append(f"  Position/placement: {position.strip()}")
    if target_state.strip():
        lines.append(f"  Target end-state: {target_state.strip()}")
    if not lines:
        return ""
    ig = [i for i in ignore if i and i.strip()] or ["background", "lighting", "camera angle"]
    lines.append(f"  IGNORE (must NOT affect the verdict): {', '.join(ig)}")
    return "KEY INFO (judge ONLY these; ignore everything else):\n" + "\n".join(lines)


_TEACHER_EVIDENCE_MARKERS = ("image 1", "teacher", "reference")
_NEGATIVE_EVIDENCE_MARKERS = (
    "missing", "not visible", "no longer present", "not present", "absent",
    "without", "cannot see", "can't see", "does not show", "doesn't show",
)

# Generic container / scene / position words that don't identify a specific
# object. Excluded when grounding a key object in the student's observation so
# discrimination falls on identifying tokens (color, material, object name) —
# otherwise two different objects that share a head noun (a "blue silicone case"
# vs a "brown leather case") would both match on "case".
_GENERIC_OBJECT_WORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "on", "in", "to", "with", "at", "by",
    "is", "are", "it", "its", "this", "that", "these", "those", "there",
    "next", "near", "beside", "left", "right", "center", "centre", "side",
    "edge", "top", "bottom", "front", "back", "corner", "middle", "onto",
    "case", "cases", "object", "objects", "item", "items", "thing", "things",
    "piece", "pieces", "box", "boxes", "container", "pouch", "pouches", "unit",
    "device", "small", "large", "round", "rounded",
    "desk", "table", "surface", "floor", "frame", "image", "scene", "view",
})


def _identity_tokens(text: str) -> set[str]:
    """Identifying tokens of an object phrase (color / material / name), minus
    generic container + scene + position words."""
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 1 and t not in _GENERIC_OBJECT_WORDS
    }


def _ungrounded_key_object(key_objects: list[str], observation: str) -> str:
    """Return the first key object whose identifying tokens are entirely absent
    from *observation*, or '' if every key object is grounded.

    The completion check otherwise trusts the VLM's per-requirement ``visible``
    flags + ``evidence`` strings, which the model will happily fabricate (mark a
    requirement visible with invented evidence) even on an off-task frame. The
    ``observation`` is the model's one honest free-text description of what it
    actually sees; requiring each key object to surface there — by a
    distinguishing token, not the shared "case" head noun — catches the frame
    that shows only one of two required objects, or none at all.
    """
    obs_tokens = _identity_tokens(observation)
    for obj in key_objects:
        want = _identity_tokens(obj)
        if want and not (want & obs_tokens):
            return obj
    return ""


def _step_retargets_previous_object(
    key_objects: list[str],
    prev_key_objects: list[str],
) -> bool:
    """True when this step's key object is the SAME object as the previous step's.

    Per-step ``key_objects`` can drift onto the previous step's object — e.g. the
    teacher frame chosen for "put a mouse next to the AirPod case" still showed the
    case (the mouse was never isolated in a recorded frame), so key-info derivation
    (told to prefer the teacher caption) yields the CASE, not the mouse, for BOTH
    steps. The completion check then trivially passes: the case is still on the
    desk from the previous step, so step 2 "completes" without the student doing
    anything.

    Detect this by identity-token containment: when the previous step's
    identifying tokens (color/material/name, minus generic container/scene words)
    are non-empty and wholly contained in this step's — "blue case" ⊆ "AirPod case
    (bright blue with red strap)" — the two steps target the same object and the
    monitor cannot tell this step's completion apart from the last one. It must NOT
    auto-advance; the student hears a correction instead. A genuinely new object
    (e.g. "black wireless mouse") shares no such tokens, so the guard stays clear
    and normal grounding applies.
    """
    cur = {t for o in key_objects for t in _identity_tokens(o)}
    prev = {t for o in prev_key_objects for t in _identity_tokens(o)}
    if not cur or not prev:
        return False
    return prev <= cur


def _check_has_student_evidence(requirement: str, current_obs: str, check: dict) -> bool:
    evidence_text = str(check.get("evidence", ""))
    evidence_lower = evidence_text.lower()
    combined_lower = f"{current_obs} {evidence_text}".lower()
    if any(marker in evidence_lower for marker in _TEACHER_EVIDENCE_MARKERS):
        return False
    if any(marker in combined_lower for marker in _NEGATIVE_EVIDENCE_MARKERS):
        return False
    return bool(evidence_text.strip())


def _normalize_checks(obj: dict) -> list[dict]:
    """Pull checks out of either the flat `requirements` map or the
    legacy nested `checks` list. Each output entry has
    ``{"requirement", "visible", "evidence"}``.
    """
    out: list[dict] = []
    flat = obj.get("requirements")
    if isinstance(flat, dict):
        for req, payload in flat.items():
            if not isinstance(payload, dict):
                continue
            out.append({
                "requirement": str(req).strip(),
                "visible":     bool(payload.get("visible", False)),
                "evidence":    str(payload.get("evidence", "")).strip(),
            })
        return out
    nested = obj.get("checks", [])
    if isinstance(nested, list):
        for entry in nested:
            if not isinstance(entry, dict):
                continue
            out.append({
                "requirement": str(entry.get("requirement", "")).strip(),
                "visible":     bool(entry.get("visible", False)),
                "evidence":    str(entry.get("evidence", "")).strip(),
            })
    return out


def _parse_grounded_completion(
    raw: str,
    expected_requirements: list[str],
    *,
    require_expected: bool = True,
    key_objects: list[str] | None = None,
    prev_key_objects: list[str] | None = None,
) -> tuple[bool, str, list[dict], list[str], str]:
    """Grounded-completion parser. Accepts both the new flat shape

        {"observation": "...", "requirements": {"<req>": {"visible": .., "evidence": ".."}}}

    and the legacy nested shape

        {"current_observation": "...", "checks": [{"requirement": "..", ...}], "completed": ..}

    Returns (completed, observation, checks, missing_or_mismatched, reject_reason).
    ``completed`` is derived: a non-empty observation, no VLM-reported issue,
    every visible check backed by grounded student evidence, no check marked
    visible=false, and — when ``key_objects`` is supplied — every key object
    actually named in the observation. Anything else collapses to
    ``(False, ...)`` with a non-empty ``reject_reason`` for the trace log.
    """
    json_str = extract_json(raw)
    if not json_str:
        return False, "", [], [], "vlm returned non-json"
    try:
        obj = json.loads(json_str)
    except Exception:
        return False, "", [], [], "vlm returned non-json"
    if not isinstance(obj, dict):
        return False, "", [], [], "vlm returned non-json"

    # Accept either field name; the new flat shape uses `observation`.
    current_obs = str(obj.get("observation") or obj.get("current_observation") or "").strip()
    issue_text = str(obj.get("issue") or obj.get("correction") or "").strip()
    checks = _normalize_checks(obj)
    missing = [c["requirement"] for c in checks if not c["visible"] and c["requirement"]]

    if not current_obs:
        return False, "", checks, missing, "vlm omitted observation"
    if issue_text and issue_text.lower() not in {"none", "n/a", "na", "no issue"}:
        return False, current_obs, checks, missing, issue_text

    # Reject "visible without evidence" — the textbook bias-copy failure.
    for c in checks:
        if c["visible"] and not c["evidence"]:
            return False, current_obs, checks, missing, "visible without evidence"
        if c["visible"] and not _check_has_student_evidence(c["requirement"], current_obs, c):
            return False, current_obs, checks, missing, f"no grounded evidence for: {c['requirement']}"

    # A requirement the VLM explicitly marked visible=false is UNMET — e.g. the
    # wrong object is in place. Do NOT complete on a partial match just because a
    # sibling check (e.g. "mouse present") is visible: that is exactly how a
    # wrong object used to slip through and auto-advance with no correction.
    if missing:
        return False, current_obs, checks, missing, f"requirement not met: {missing[0]}"

    if not any(c["visible"] and c["evidence"] for c in checks):
        return False, current_obs, checks, missing, "no grounded evidence"

    # Object grounding: every key object must actually appear in the student's
    # honest observation (not just in fabricated per-requirement evidence). This
    # is what stops a "place glasses case next to the AirPod case" step from
    # completing on a frame that shows only the AirPod case — or an off-task
    # frame (e.g. a comb) — where the VLM still returned all-visible.
    ungrounded = _ungrounded_key_object(
        [o for o in (key_objects or []) if o and str(o).strip()], current_obs,
    )
    if ungrounded:
        return False, current_obs, checks, missing, f"key object not in view: {ungrounded}"

    # Retargeting guard: when this step's key object is the SAME as the previous
    # step's (its key info drifted onto the prior object, still in view), the
    # completion check is a no-op that auto-passes without the student doing this
    # step. Refuse to complete so the student is corrected instead of advanced.
    if _step_retargets_previous_object(
        [o for o in (key_objects or []) if o and str(o).strip()],
        [o for o in (prev_key_objects or []) if o and str(o).strip()],
    ):
        return False, current_obs, checks, missing, "cannot verify this step from the demo"

    return True, current_obs, checks, missing, ""


def _human_issue_from_raw(raw: str) -> str:
    json_str = extract_json(raw)
    if not json_str:
        return ""
    try:
        obj = json.loads(json_str)
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("issue") or obj.get("correction") or "").strip()


def _issue_from_failure(raw: str, reject_reason: str, missing: list[str]) -> str:
    human_issue = _human_issue_from_raw(raw)
    if human_issue:
        return human_issue
    if reject_reason:
        return reject_reason
    if missing:
        return (
            f"{missing[0]} not visible"
            if len(missing) == 1
            else f"{missing[0]} and {missing[1]} not visible"
        )
    return ""


def _key_info_correction(
    objects: list[str], action: str, position: str, target_state: str,
) -> str:
    """Build a spoken-friendly 'what's wrong' line from key info.

    Used when both the image-to-image and image-to-text checks fail and the
    VLM did not return a human-readable issue — the student still hears a
    concrete correction grounded in the step's key facts.
    """
    obj = (objects[0] if objects else "").strip()
    if target_state.strip():
        ts = target_state.strip()
        return f"{obj} should be {ts}" if obj else f"I don't see {ts} yet"
    if position.strip():
        return f"{obj or 'it'} should be {position.strip()}"
    if action.strip():
        return f"{action.strip()} isn't done yet"
    return ""


async def _diagnose_mistake(
    vlm: VLMService,
    image_path: str,
    *,
    instruction: str,
    objects: list[str],
    action: str,
    position: str,
    target_state: str,
) -> str:
    """Tier 3 — actively name what the student is doing wrong.

    Runs only after the screenshot (tier 1) and text (tier 2) checks have both
    failed WITHOUT producing a concrete, human-readable correction. A dedicated
    VLM call, told the step is not complete, names the single most important
    mistake (most often the wrong object) so the student always hears something
    specific instead of silence or a generic template.
    """
    facts: list[str] = []
    objs = [o for o in objects if o and o.strip()]
    if objs:
        facts.append(f"required object(s): {', '.join(objs)}")
    if action.strip():
        facts.append(f"action: {action.strip()}")
    if position.strip():
        facts.append(f"placement: {position.strip()}")
    if target_state.strip():
        facts.append(f"done when: {target_state.strip()}")
    facts_block = ("\n".join(f"  - {f}" for f in facts) + "\n") if facts else ""

    question = (
        f"The student is trying to do this step: {instruction}\n"
        f"{facts_block}"
        "The step is NOT complete. Look ONLY at this live image and say, in ONE "
        "short second-person sentence, the single most important thing that is "
        "wrong — most often the WRONG OBJECT (name what you actually see vs what "
        "the step needs), otherwise wrong placement or that it is not done yet. "
        "Do NOT claim it is correct.\n\n"
        'Output ONLY this JSON: {"issue": "<one short correction to the student>"}'
    )
    try:
        raw = (await vlm.ask_image(image_path, question)).content or ""
    except Exception:
        log.exception("guidance tier-3 diagnosis failed")
        return ""
    return _human_issue_from_raw(raw)


def _output_from_parse(
    *,
    completed: bool,
    current_obs: str,
    checks: list[dict],
    missing: list[str],
    issue: str,
    image_path: str,
    timestamp_us: int,
    raw: str,
    teacher_image_path: str = "",
) -> GuidanceCheckResult:
    return GuidanceCheckResult(
        completed=completed,
        current_observation=current_obs,
        checks=[StepCheck(**c) for c in checks],
        missing_or_mismatched=missing,
        image_path=image_path,
        teacher_image_path=teacher_image_path,
        timestamp_us=timestamp_us,
        issue=issue,
        raw_vlm=raw,
    )


def _parser_issue(issue: str) -> bool:
    return issue.lower().startswith((
        "vlm ",
        "missing requirement",
        "visible without",
        "no grounded",
        "vlm omitted",
        "cannot verify this step",
    ))


async def check_guidance_step_complete(
    *,
    participant_id: str,
    instruction: str,
    expected_requirements: list[str] | None = None,
    teacher_image_path: str = "",
    teacher_caption: str = "",
    min_live_timestamp_us: int = 0,
    key_objects: list[str] | None = None,
    prev_key_objects: list[str] | None = None,
    key_action: str = "",
    key_position: str = "",
    key_target_state: str = "",
    key_ignore: list[str] | None = None,
    vlm: VLMService,
    get_latest_frame: Callable[[str], Awaitable[FrameRef | None]],
) -> GuidanceCheckResult:
    expected_requirements = expected_requirements or []
    key_objects = key_objects or []
    prev_key_objects = prev_key_objects or []
    key_ignore = key_ignore or []
    frame = await get_latest_frame(participant_id)
    if frame is None or not frame.image_path:
        return GuidanceCheckResult(issue="I cannot see a current frame.")
    image_path = frame.image_path
    frame_ts = int(frame.timestamp_us or 0)
    if min_live_timestamp_us and frame_ts and frame_ts < min_live_timestamp_us:
        return GuidanceCheckResult(
            image_path=image_path,
            timestamp_us=frame_ts,
            issue="Waiting for a fresh student frame.",
        )

    key_block = _key_info_block(
        objects=key_objects or [],
        action=key_action,
        position=key_position,
        target_state=key_target_state,
        ignore=key_ignore or [],
    )
    key_lines = f"{key_block}\n\n" if key_block else ""

    expected = [r.strip() for r in (expected_requirements or []) if r and str(r).strip()]
    if expected:
        checklist_block = "\n".join(f"  - {r}" for r in expected)
        checklist_lines = (
            f"REQUIREMENTS (use as visual hints, not as stricter wording than the instruction):\n{checklist_block}\n\n"
        )
    else:
        checklist_lines = (
            "No predefined requirements were provided. Derive 1-2 short, "
            "visually checkable requirements from the INSTRUCTION and include "
            "those requirement texts in the JSON.\n\n"
        )

    teacher_path = (
        teacher_image_path
        if teacher_image_path and os.path.isfile(teacher_image_path)
        else ""
    )
    teacher_failure: GuidanceCheckResult | None = None
    if teacher_path:
        comparison_question = (
            f"INSTRUCTION: {instruction}\n"
            "Image 1 is the teacher's completed reference state for this step.\n"
            "Image 2 is the student's current state.\n"
            f"Teacher reference caption: {teacher_caption or 'not available'}\n"
            f"{key_lines}"
            f"{checklist_lines}"
            "Decide whether Image 2 matches Image 1 for the KEY INFO — the key "
            "OBJECTS (by type/identity), the action, and the spatial placement.\n"
            "OBJECT IDENTITY IS STRICT: the student must be using the SAME kind "
            "of object the step calls for. A DIFFERENT object in the right place "
            "is a MISMATCH, not a match (e.g. a phone where an AirPod case is "
            "required — fail it).\n"
            "You MAY ignore: color shade, exact brand, background, lighting, "
            "camera angle, distance, clothing, hand pose. You may NOT ignore the "
            "object's type/identity, its placement, or the action.\n\n"
            "Output ONLY this JSON, no prose, no markdown:\n"
            "{\n"
            '  "observation": "<one sentence naming EACH object you actually see in Image 2, by color/material>",\n'
            '  "requirements": {\n'
            '    "<each key object/placement, named>": {"visible": true|false, "evidence": "<Image 2 cue, or empty>"}\n'
            "  },\n"
            '  "issue": "<if any check is false, a concrete correction naming the wrong/missing object vs the expected one; else empty>"\n'
            "}\n\n"
            "Rules:\n"
            "- Add one check per key object/placement; set visible=false when the "
            "object is the wrong type, missing, or misplaced.\n"
            "- Evidence must come from Image 2, not Image 1.\n"
            "- If every key check is visible=true with Image 2 evidence, leave issue empty.\n"
            "- If any check is false, issue MUST name what is wrong (e.g. "
            '"that looks like a phone, but this step needs the AirPod case").'
        )
        # Teacher reference first (Image 1), student current frame second (Image
        # 2) — the prompt above refers to them in that order.
        compare_raw = (await vlm.ask_images([teacher_path, image_path], comparison_question)).content or ""
        if not compare_raw:
            log.warning("ask_frames guidance check failed; using instruction fallback: %s", compare_raw)
        else:
            cmp_completed, cmp_obs, cmp_checks, cmp_missing, cmp_reject = _parse_grounded_completion(
                compare_raw, expected, require_expected=False,
                key_objects=key_objects, prev_key_objects=prev_key_objects,
            )
            cmp_issue = _issue_from_failure(compare_raw, cmp_reject, cmp_missing)
            if cmp_completed:
                return _output_from_parse(
                    completed=True,
                    current_obs=cmp_obs,
                    checks=cmp_checks,
                    missing=cmp_missing,
                    issue="",
                    image_path=image_path,
                    teacher_image_path=teacher_path,
                    timestamp_us=frame_ts,
                    raw=compare_raw,
                )
            teacher_failure = _output_from_parse(
                completed=False,
                current_obs=cmp_obs,
                checks=cmp_checks,
                missing=cmp_missing,
                issue=cmp_issue,
                image_path=image_path,
                teacher_image_path=teacher_path,
                timestamp_us=frame_ts,
                raw=compare_raw,
            )

    live_question = (
        f"INSTRUCTION: {instruction}\n"
        f"TEACHER CAPTION: {teacher_caption or 'not available'}\n"
        f"{key_lines}"
        f"{checklist_lines}"
        "Look ONLY at this live student image.\n"
        "Decide whether it satisfies the KEY INFO — the key OBJECTS (by "
        "type/identity) in the target end-state/placement, performing the action.\n"
        "OBJECT IDENTITY IS STRICT: if the student is using a DIFFERENT object "
        "than the step requires (e.g. a phone instead of an AirPod case), the "
        "step is NOT satisfied — even if it is placed correctly.\n"
        "You MAY ignore color shade, exact brand, background, lighting, camera "
        "angle, and clothing; you may NOT ignore the object's type/identity, its "
        "placement, or the action.\n\n"
        "Output ONLY this JSON, no prose, no markdown:\n"
        "{\n"
        '  "observation": "<one sentence naming EACH object you actually see, by color/material>",\n'
        '  "requirements": {\n'
        '    "<each key object/placement, named>": {"visible": true|false, "evidence": "<live-image cue, or empty>"}\n'
        "  },\n"
        '  "issue": "<if any check is false, a concrete correction naming the wrong/missing object vs the expected one; else empty>"\n'
        "}\n\n"
        "Rules:\n"
        "- observation MUST be a non-empty sentence about THIS live image, naming the object(s) present.\n"
        "- Add one check per key object/placement; set visible=false when the object is the wrong type, missing, or misplaced.\n"
        "- visible=true REQUIRES non-empty evidence from THIS live image; do not invent evidence.\n"
        "- If any check is false, issue MUST name what is wrong vs what the step expects."
    )
    live_raw = (await vlm.ask_image(image_path, live_question)).content or ""
    completed, current_obs, checks, missing, reject_reason = _parse_grounded_completion(
        live_raw, expected, require_expected=not bool(teacher_path),
        key_objects=key_objects, prev_key_objects=prev_key_objects,
    )
    live_issue = _issue_from_failure(live_raw, reject_reason, missing)
    if completed:
        return _output_from_parse(
            completed=True,
            current_obs=current_obs,
            checks=checks,
            missing=missing,
            issue="",
            image_path=image_path,
            teacher_image_path=teacher_path,
            timestamp_us=frame_ts,
            raw=live_raw,
        )

    final_issue = (
        teacher_failure.issue
        if (
            teacher_failure is not None
            and teacher_failure.issue
            and not _parser_issue(teacher_failure.issue)
        )
        else live_issue
    )
    # Tiers 1 & 2 failed without a usable human correction. Tier 3: actively ask
    # the VLM what is wrong (names the wrong object / placement) so the student
    # always hears a concrete mistake. Only fires on this otherwise-silent path,
    # so it adds no cost when a correction already exists.
    if not final_issue or _parser_issue(final_issue):
        diagnosed = await _diagnose_mistake(
            vlm, image_path,
            instruction=instruction,
            objects=key_objects or [], action=key_action,
            position=key_position, target_state=key_target_state,
        )
        if diagnosed:
            final_issue = diagnosed
        else:
            ki_issue = _key_info_correction(
                key_objects or [], key_action, key_position, key_target_state,
            )
            if ki_issue:
                final_issue = ki_issue
    return _output_from_parse(
        completed=False,
        current_obs=current_obs,
        checks=checks,
        missing=missing,
        issue=final_issue,
        image_path=image_path,
        teacher_image_path=teacher_path,
        timestamp_us=frame_ts,
        raw=live_raw,
    )
