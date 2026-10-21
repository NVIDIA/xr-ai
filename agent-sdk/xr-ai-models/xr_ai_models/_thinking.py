# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-model control of reasoning ("thinking") on chat requests.

Models switch reasoning on and off in different ways, so the switch is part of
the adapter configuration instead of raw request extras.  Callers state intent
per call with ``enable_thinking``; the endpoint's :class:`ThinkingSpec`
supplies the default and renders the effective value in the model's dialect.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, get_args

ThinkingControl = Literal["chat_template", "system_prompt", "none"]
"""How a model receives the reasoning switch."""

_CONTROLS: tuple[str, ...] = get_args(ThinkingControl)
_SYSTEM_SWITCH = "detailed thinking"


@dataclass(frozen=True)
class ThinkingSpec:
    """How a model endpoint switches reasoning, and its default state."""

    control: ThinkingControl = "chat_template"
    """The model's switch dialect.

    ``chat_template`` sends ``chat_template_kwargs.enable_thinking`` (and
    ``thinking_budget``), as vLLM and NIM Nemotron/Qwen-style templates expect.
    ``system_prompt`` prepends ``detailed thinking on|off`` to the system
    message, as Llama-Nemotron expects.  ``none`` leaves requests unchanged
    for models without a switch or endpoints that reject template arguments.
    """

    default: bool | None = None
    """Reasoning state when a call leaves ``enable_thinking`` unset.

    ``None`` sends no switch, so the model's own template default applies.
    """

    def __post_init__(self) -> None:
        if self.control not in _CONTROLS:
            raise ValueError(
                f"thinking control must be one of {list(_CONTROLS)}, "
                f"got {self.control!r}"
            )
        if self.default is not None and not isinstance(self.default, bool):
            raise ValueError("thinking default must be a boolean or null")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ThinkingSpec":
        """Build a spec from a ``thinking`` configuration mapping.

        Raises :class:`ValueError` for unknown keys or invalid values.
        """

        unknown = set(value) - {"control", "default"}
        if unknown:
            raise ValueError(f"unknown thinking keys: {sorted(unknown)}")
        return cls(**value)


def lift_legacy_default(
    spec: ThinkingSpec,
    default_extras: Mapping[str, Any],
) -> tuple[ThinkingSpec, dict[str, Any]]:
    """Move ``chat_template_kwargs.enable_thinking`` out of request extras.

    Older configurations set the default as a raw request field, which a
    per-call value could not override.  When present, it becomes the spec's
    default and is removed from the extras; other template arguments stay.
    """

    extras = dict(default_extras)
    tpl = extras.get("chat_template_kwargs")
    if not isinstance(tpl, Mapping) or "enable_thinking" not in tpl:
        return spec, extras
    rest = {k: v for k, v in tpl.items() if k != "enable_thinking"}
    if rest:
        extras["chat_template_kwargs"] = rest
    else:
        del extras["chat_template_kwargs"]
    return replace(spec, default=bool(tpl["enable_thinking"])), extras


def apply_thinking(
    spec: ThinkingSpec,
    payload: dict[str, Any],
    *,
    enable_thinking: bool | None,
    thinking_budget: int | None,
) -> None:
    """Render the effective reasoning switch into an outgoing *payload*.

    A per-call ``enable_thinking`` always wins over the spec default.  Models
    with ``control="none"`` and the ``system_prompt`` dialect ignore
    ``thinking_budget``, which only template-driven models understand.
    """

    effective = spec.default if enable_thinking is None else enable_thinking
    if spec.control == "chat_template":
        tpl: dict[str, Any] = {}
        if effective is not None:
            tpl["enable_thinking"] = effective
        if thinking_budget is not None:
            tpl["thinking_budget"] = thinking_budget
        if tpl:
            payload["chat_template_kwargs"] = {
                **payload.get("chat_template_kwargs", {}),
                **tpl,
            }
    elif spec.control == "system_prompt" and effective is not None:
        _set_system_switch(payload["messages"], effective)


def _set_system_switch(messages: list[dict[str, Any]], on: bool) -> None:
    # Llama-Nemotron reads only the first system message; a caller that wrote
    # the phrase there already chose the mode for this request.
    line = f"{_SYSTEM_SWITCH} {'on' if on else 'off'}"
    first = messages[0] if messages else None
    if first is None or first.get("role") != "system":
        messages.insert(0, {"role": "system", "content": line})
        return
    content = first.get("content")
    if isinstance(content, str):
        if _SYSTEM_SWITCH in content:
            return
        first["content"] = f"{line}\n\n{content}" if content else line
    else:
        first["content"] = [{"type": "text", "text": line}, *(content or [])]
