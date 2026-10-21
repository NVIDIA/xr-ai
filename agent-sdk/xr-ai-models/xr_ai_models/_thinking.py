# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-model control of reasoning ("thinking") on chat requests.

Models switch reasoning on and off in different ways, so the switch is part of
the adapter configuration instead of raw request extras.  Callers state intent
per call with ``enable_thinking``; the endpoint's :class:`ThinkingSpec`
supplies the default and renders the effective value in the model's dialect.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, get_args

ThinkingControl = Literal["chat_template", "system_prompt", "none"]
"""How a model receives the reasoning switch."""

_CONTROLS: tuple[str, ...] = get_args(ThinkingControl)
_LEGACY_FIELD = "default_extras.chat_template_kwargs.enable_thinking"
# A complete Llama-Nemotron mode directive on its own line.  Prose that merely
# mentions "detailed thinking" is not a mode selection.
_DIRECTIVE = re.compile(r"^[ \t]*detailed thinking (?:on|off)[ \t]*$", re.I | re.M)


@dataclass(frozen=True)
class ThinkingSpec:
    """How a model endpoint switches reasoning, and its default state."""

    control: ThinkingControl | None = None
    """The model's switch dialect.

    ``chat_template`` sends ``chat_template_kwargs.enable_thinking`` (and
    ``thinking_budget``), as vLLM and NIM Nemotron/Qwen-style templates expect.
    ``system_prompt`` puts a ``detailed thinking on|off`` line at the start of
    the system message, as Llama-Nemotron expects.  ``none`` leaves requests
    unchanged for models without a switch or endpoints that reject template
    arguments.  ``None`` (unset) uses ``chat_template`` when ``default`` is
    set; otherwise only a call that enables thinking sends the switch, and an
    explicit off adds nothing to the request.
    """

    default: bool | None = None
    """Reasoning state when a call leaves ``enable_thinking`` unset.

    ``None`` sends no switch, so the model's own template default applies.
    """

    def __post_init__(self) -> None:
        if self.control is not None and self.control not in _CONTROLS:
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


def legacy_default(default_extras: Mapping[str, Any]) -> bool | None:
    """Return the legacy ``enable_thinking`` request default, if present.

    Raises :class:`ValueError` when the value is not a boolean.
    """

    tpl = default_extras.get("chat_template_kwargs")
    if not isinstance(tpl, Mapping) or "enable_thinking" not in tpl:
        return None
    value = tpl["enable_thinking"]
    if not isinstance(value, bool):
        raise ValueError(f"{_LEGACY_FIELD} must be a boolean")
    return value


def check_legacy_conflict(
    spec: ThinkingSpec,
    default_extras: Mapping[str, Any],
) -> None:
    """Reject one configuration that sets two different defaults.

    Raises :class:`ValueError` when ``spec.default`` and the legacy request
    default are both set and disagree on a template-driven endpoint, or when
    the legacy value is not a boolean.
    """

    if spec.control not in (None, "chat_template"):
        return
    legacy = legacy_default(default_extras)
    if legacy is not None and spec.default is not None and legacy != spec.default:
        raise ValueError(f"thinking.default conflicts with {_LEGACY_FIELD}")


def lift_legacy_default(
    spec: ThinkingSpec,
    default_extras: Mapping[str, Any],
) -> tuple[ThinkingSpec, dict[str, Any]]:
    """Move a legacy ``enable_thinking`` request default into *spec*.

    Older configurations set the default as a raw request field, which a
    per-call value could not override.  For template-driven endpoints the
    value becomes the spec default, replacing any inherited one, and other
    template arguments stay in the extras.  Endpoints declared with another
    control keep the raw field on the wire, as before.

    Raises :class:`ValueError` when a value that would be lifted is not a
    boolean.
    """

    extras = dict(default_extras)
    if spec.control not in (None, "chat_template"):
        return spec, extras
    legacy = legacy_default(extras)
    if legacy is None:
        return spec, extras
    rest = {k: v for k, v in extras["chat_template_kwargs"].items() if k != "enable_thinking"}
    if rest:
        extras["chat_template_kwargs"] = rest
    else:
        del extras["chat_template_kwargs"]
    return replace(spec, control="chat_template", default=legacy), extras


def apply_thinking(
    spec: ThinkingSpec,
    payload: dict[str, Any],
    *,
    enable_thinking: bool | None,
    thinking_budget: int | None,
) -> None:
    """Render the effective reasoning switch into an outgoing *payload*.

    A per-call ``enable_thinking`` wins over the spec default.  Models with
    ``control="none"`` and the ``system_prompt`` dialect ignore
    ``thinking_budget``, which only template-driven models understand.
    """

    effective = spec.default if enable_thinking is None else enable_thinking
    control = spec.control
    if control is None:
        if spec.default is not None:
            control = "chat_template"
        elif effective is False:
            # Undeclared dialect: an explicit off adds nothing, as before.
            effective = None
    if control in (None, "chat_template"):
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
    elif control == "system_prompt" and effective is not None:
        _set_system_switch(
            payload["messages"], effective, forced=enable_thinking is not None,
        )


def _set_system_switch(
    messages: list[dict[str, Any]],
    on: bool,
    *,
    forced: bool,
) -> None:
    # Llama-Nemotron reads only the first system message.  A directive the
    # caller wrote there stands for the endpoint default but yields to an
    # explicit per-call value.
    line = f"detailed thinking {'on' if on else 'off'}"
    first = messages[0] if messages else None
    if first is None or first.get("role") != "system":
        messages.insert(0, {"role": "system", "content": line})
        return
    content = first.get("content")
    if isinstance(content, str):
        if _DIRECTIVE.search(content):
            if forced:
                first["content"] = _DIRECTIVE.sub(line, content)
            return
        first["content"] = f"{line}\n\n{content}" if content.strip() else line
        return
    parts = list(content or [])
    for part in parts:
        text = part.get("text") if part.get("type") == "text" else None
        if isinstance(text, str) and _DIRECTIVE.search(text):
            if forced:
                part["text"] = _DIRECTIVE.sub(line, text)
            return
    first["content"] = [{"type": "text", "text": line}, *parts]
