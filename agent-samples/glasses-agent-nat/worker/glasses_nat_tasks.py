# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task implementations used by registered glasses-agent NAT functions."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from glasses_nat_schemas import (
    AnalyzeRecordingInput,
    AnalyzeRecordingOutput,
    CondenseObservationsInput,
    CondenseObservationsOutput,
    DeriveStepKeyInfoInput,
    DeriveStepKeyInfoOutput,
    DeriveStepRequirementsInput,
    DeriveStepRequirementsOutput,
    GuidanceStepOutput,
    StepCheck,
)
from xr_ai_capabilities import (
    FrameRef as CapFrameRef,
)
from xr_ai_capabilities import (
    RecordingFrame as CapRecordingFrame,
)
from xr_ai_capabilities import (
    VoiceNote as CapVoiceNote,
)
from xr_ai_capabilities import (
    analyze_recording as cap_analyze_recording,
)
from xr_ai_capabilities import (
    check_guidance_step_complete as cap_check_guidance_step_complete,
)
from xr_ai_capabilities import (
    derive_step_key_info as cap_derive_step_key_info,
)
from xr_ai_capabilities import (
    derive_step_requirements as cap_derive_step_requirements,
)
from xr_ai_models import Capabilities, ChatResponse

log = logging.getLogger("glasses_agent_nat.tasks")


def extract_json(text: str) -> str | None:
    clean = text.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        if len(parts) >= 2:
            clean = parts[1].lstrip("json").strip()
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(clean):
        if ch != "{":
            continue
        try:
            _obj, end = decoder.raw_decode(clean[idx:])
            return clean[idx:idx + end]
        except json.JSONDecodeError:
            pass

    depth, start, in_string, escape = 0, -1, False, False
    for i, ch in enumerate(clean):
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
                return clean[start:i + 1]
    if start >= 0 and depth > 0 and not in_string and depth <= 3:
        return clean[start:] + ("}" * depth)
    return None


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


async def _chat(
    llm,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Invoke a NAT-provided LangChain chat model and return its text.

    Replaces the previous hand-rolled httpx POST to ``/v1/chat/completions``:
    the worker tasks now go through NAT's LLM layer (``builder.get_llm(...,
    LLMFrameworkEnum.LANGCHAIN)``), so model endpoint, retries, timeout, and
    observability are unified under the workflow YAML ``llms:`` block instead
    of duplicated here. LangChain normalizes OpenAI-style ``{"role", "content"}``
    message dicts, so the tuned prompts pass through unchanged; per-call
    ``max_tokens`` / ``temperature`` / ``extra_body`` overrides ride a
    ``.bind(...)`` so they don't mutate the shared client.
    """
    bind_kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        bind_kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        bind_kwargs["temperature"] = temperature
    if extra_body:
        bind_kwargs["extra_body"] = extra_body
    client = llm.bind(**bind_kwargs) if bind_kwargs else llm
    result = await client.ainvoke(messages)

    content = getattr(result, "content", result)
    if isinstance(content, list):
        # Some chat models return content as a list of parts/blocks.
        content = "".join(
            blk.get("text", "") if isinstance(blk, dict) else str(blk)
            for blk in content
        )
    return content.strip() if isinstance(content, str) else str(content).strip()


class _LangChainLLMService:
    """Adapt a NAT/LangChain chat model to xr_ai_models.LLMService for the
    reusable teacher-demo capability. Delegates to the module ``_chat`` helper
    (preserving the enable_thinking → chat_template_kwargs mapping)."""
    capabilities = Capabilities(tool_calls=False, vision=False)

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def chat(self, messages, *, tools=None, max_tokens=None, temperature=None,
                   enable_thinking=False, thinking_budget=None, timeout=None) -> ChatResponse:
        extra_body = None if enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}
        content = await _chat(
            self._llm,
            [{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens, temperature=temperature, extra_body=extra_body,
        )
        return ChatResponse(content=content, reasoning=None, tool_calls=None,
                            finish_reason="stop", raw={})

    def stream(self, *a, **k): raise NotImplementedError
    async def health(self) -> bool: return True
    async def close(self) -> None: return None


class _NatVLMService:
    """Adapt NAT vlm-mcp functions to xr_ai_models.VLMService for the reusable
    agent-monitor capability. ``ask_image`` → single-image vlm-mcp ask_image;
    ``ask_images`` → two-image vlm-mcp ask_frames."""
    capabilities = Capabilities(vision=True)

    def __init__(self, ask_image: Any, ask_frames: Any = None) -> None:
        self._ask_image = ask_image
        self._ask_frames = ask_frames

    async def ask_image(self, image, question, *, system_prompt="", max_tokens=None,
                        temperature=None, timeout=None) -> ChatResponse:
        raw = _str_payload(await self._ask_image.ainvoke(
            {"question": question, "image_path": image}))
        return ChatResponse(content=raw, reasoning=None, tool_calls=None,
                            finish_reason="stop", raw={})

    async def ask_images(self, images, question, *, system_prompt="", max_tokens=None,
                         temperature=None, timeout=None) -> ChatResponse:
        if self._ask_frames is None:
            return ChatResponse(content="", reasoning=None, tool_calls=None,
                                finish_reason="stop", raw={})
        raw = _str_payload(await self._ask_frames.ainvoke(
            {"question": question, "image_paths": list(images)}))
        return ChatResponse(content=raw, reasoning=None, tool_calls=None,
                            finish_reason="stop", raw={})

    async def ask_video(self, *a, **k): raise NotImplementedError
    def stream(self, *a, **k): raise NotImplementedError
    async def health(self) -> bool: return True
    async def close(self) -> None: return None


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
    agent_llm,
) -> AnalyzeRecordingOutput:
    result = await cap_analyze_recording(
        request.name, request.started_at_us,
        [CapRecordingFrame(f.frame_idx, f.timestamp_us, f.image_path, f.description) for f in request.frames],
        [CapVoiceNote(v.timestamp_us, v.text) for v in request.voice_notes],
        llm=_LangChainLLMService(agent_llm),
    )
    return AnalyzeRecordingOutput(overview=result.overview, steps=result.steps)


async def condense_observations_impl(
    request: CondenseObservationsInput,
    *,
    worker_llm,
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
    raw = await _chat(
        worker_llm,
        messages,
        max_tokens=256,
        temperature=0.1,
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


async def derive_step_requirements_impl(
    request: DeriveStepRequirementsInput,
    *,
    agent_llm,
) -> DeriveStepRequirementsOutput:
    reqs = await cap_derive_step_requirements(
        request.instruction, request.teacher_caption, llm=_LangChainLLMService(agent_llm))
    return DeriveStepRequirementsOutput(requirements=reqs)


async def derive_step_key_info_impl(
    request: DeriveStepKeyInfoInput,
    *,
    agent_llm,
) -> DeriveStepKeyInfoOutput:
    ki = await cap_derive_step_key_info(
        request.instruction, request.teacher_caption, request.requirements,
        llm=_LangChainLLMService(agent_llm))
    return DeriveStepKeyInfoOutput(objects=ki.objects, action=ki.action, position=ki.position,
                                   target_state=ki.target_state, ignore=ki.ignore)


async def check_guidance_step_complete_impl(
    *,
    participant_id: str,
    instruction: str,
    expected_requirements: list[str] | None = None,
    teacher_image_path: str = "",
    teacher_caption: str = "",
    min_live_timestamp_us: int = 0,
    key_objects: list[str] | None = None,
    key_action: str = "",
    key_position: str = "",
    key_target_state: str = "",
    key_ignore: list[str] | None = None,
    get_latest_frame,
    ask_image,
    ask_frames=None,
) -> GuidanceStepOutput:
    async def _fetch(pid: str):
        fr = _payload(await get_latest_frame.ainvoke({"participant_id": pid}))
        if not isinstance(fr, dict):
            return None
        path = fr.get("path") or ""
        if not path:
            return None
        return CapFrameRef(image_path=path, timestamp_us=int(fr.get("timestamp_us") or 0))

    res = await cap_check_guidance_step_complete(
        participant_id=participant_id, instruction=instruction,
        expected_requirements=expected_requirements, teacher_image_path=teacher_image_path,
        teacher_caption=teacher_caption, min_live_timestamp_us=min_live_timestamp_us,
        key_objects=key_objects, key_action=key_action, key_position=key_position,
        key_target_state=key_target_state, key_ignore=key_ignore,
        vlm=_NatVLMService(ask_image, ask_frames), get_latest_frame=_fetch,
    )
    return GuidanceStepOutput(
        completed=res.completed, current_observation=res.current_observation,
        checks=[StepCheck(requirement=c.requirement, visible=c.visible, evidence=c.evidence) for c in res.checks],
        missing_or_mismatched=res.missing_or_mismatched, image_path=res.image_path,
        teacher_image_path=res.teacher_image_path, timestamp_us=res.timestamp_us,
        issue=res.issue, raw_vlm=res.raw_vlm,
    )
