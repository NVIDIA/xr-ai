# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass, YAML loader, and consumer-facing Protocols
for the voice gate."""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field, fields, replace
from importlib.resources import files
from typing import Protocol

import yaml

from ._conversation import _ConversationConfig
from ._speaker import _SpeakerConfig

_TRUE_BOOL_STRINGS = {"1", "true", "yes", "on"}
_FALSE_BOOL_STRINGS = {"0", "false", "no", "off"}


def _parse_config_bool(value: object, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_BOOL_STRINGS:
            return True
        if normalized in _FALSE_BOOL_STRINGS:
            return False
    accepted = sorted(_TRUE_BOOL_STRINGS | _FALSE_BOOL_STRINGS)
    raise ValueError(
        f"{key} must be a boolean or one of {accepted} (got {value!r})"
    )


@dataclass(frozen=True)
class VoiceGateConfig:
    """Voice-gate behavior settings."""

    magic_phrases:    tuple[str, ...] = ()
    """Sentence-boundary opt-in phrases.

    A match is valid at transcript start or after ``.``, ``?``, or ``!``.
    An empty tuple dispatches every non-STOP STT transcript, or every transcript
    when ``stop_commands_enabled`` is false.
    """

    followup_grace_s: float           = 5.0
    """Seconds in which the next utterance may begin without another phrase.

    The utterance may finish after the grace period.
    """

    listening_chime:  bool            = True
    """Whether phrase matches may emit a listening chime.

    The chime is available only when ``magic_phrases`` is non-empty and plays
    when the consumer calls ``VoiceGate.play_chime``. It defaults to true;
    set ``listening_chime: false`` to disable it.
    """

    stop_commands_enabled: bool = True
    """Whether STOP recognition interrupts speech or invokes the stop handler.

    Applies to early probes, streaming partials, and final transcripts,
    including conversations without a wake phrase. Disable for passive
    narration: STOP words then follow normal transcript and query gating
    without a STOP interruption or acknowledgement. Probes may remain enabled
    for wake acknowledgements.
    This does not disable other conversation controls or speech-start events.
    Scoped application commands such as "stop recording" remain queries.
    """

    _speaker: _SpeakerConfig | None = field(default=None, init=False, repr=False)
    _conversation: _ConversationConfig | None = field(default=None, init=False, repr=False)
    _mode: str = field(default="always_on", init=False, repr=False)

    def __post_init__(self) -> None:
        # Direct construction still opts in through its existing phrase field.
        object.__setattr__(self, "_mode", "wake_only" if self.magic_phrases else "always_on")


def load_voice_gate_config(path: pathlib.Path) -> VoiceGateConfig:
    """Load + parse a voice_gate YAML file into a :class:`VoiceGateConfig`.

    Schema: a top-level mapping with ``mode`` (``always_on``, ``wake_only``,
    ``conversation`` or ``hybrid``), ``magic_phrases`` (list[str] or
    bare str), ``listening_chime`` (bool), ``followup_grace_s`` (float),
    ``stop_commands_enabled`` (bool, default true).
    Optional ``conversation`` and ``speaker`` mappings configure session
    controls and speech-backend selection for the unified voice pipeline.
    Their settings remain private to that path.
    The package's annotated ``voice_gate.yaml`` supplies defaults; the selected
    file overrides individual top-level and section fields. Missing or empty
    files use the packaged ``always_on`` baseline. Mode chooses activation;
    the optional mappings tune controls and choose ASR independently. Addressed
    modes require non-empty phrases. Current mode-less configurations normalize
    once using their conversation and phrase settings; explicit mode must not be
    mixed with ``conversation.enabled``. ``magic_phrases: null`` and trailing
    whitespace are normalized before validation.
    """
    defaults = yaml.safe_load(files(__package__).joinpath("voice_gate.yaml").read_text(encoding="utf-8"))
    if not isinstance(defaults, dict):
        raise ValueError("packaged voice_gate.yaml: top-level must be a mapping")
    raw = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping")

    values = {**defaults, **raw}
    for section in ("conversation", "speaker"):
        if section not in raw or not isinstance(raw[section], dict):
            continue
        section_defaults = dict(defaults[section])
        if section == "conversation":
            # Legacy mapping presence enables controls unless explicitly disabled.
            section_defaults.pop("enabled", None)
        else:
            # Legacy speaker-only files also carry the conversation controls.
            section_defaults = {
                **{k: v for k, v in defaults["conversation"].items() if k != "enabled"},
                **section_defaults,
            }
        values[section] = {**section_defaults, **raw[section]}

    phrases_raw = values.get("magic_phrases") or []
    if isinstance(phrases_raw, str):
        phrases_raw = [phrases_raw]
    phrases = tuple(p for p in (s.strip() for s in phrases_raw) if p)

    cfg = VoiceGateConfig(
        magic_phrases    = phrases,
        followup_grace_s = float(values["followup_grace_s"]),
        listening_chime  = _parse_config_bool(
            values["listening_chime"], "listening_chime"
        ),
        stop_commands_enabled = _parse_config_bool(
            values["stop_commands_enabled"], "stop_commands_enabled"
        ),
    )
    if (
        isinstance(values["followup_grace_s"], bool)
        or not math.isfinite(cfg.followup_grace_s) or cfg.followup_grace_s < 0
    ):
        raise ValueError("followup_grace_s must be finite and non-negative")
    policy_keys = {"mode", "magic_phrases", "followup_grace_s", "stop_commands_enabled"}
    if isinstance(raw.get("speaker"), dict) and raw["speaker"].keys() & policy_keys:
        raise ValueError("interaction policy settings belong at the top level, not under speaker")
    explicit_mode = "mode" in raw
    control_keys = {f.name for f in fields(_ConversationConfig)}
    if explicit_mode and isinstance(raw.get("speaker"), dict) and raw["speaker"].keys() & control_keys:
        raise ValueError("with explicit mode, move conversation controls from speaker to conversation")
    if explicit_mode and isinstance(raw.get("conversation"), dict) and "enabled" in raw["conversation"]:
        raise ValueError("mode replaces conversation.enabled; remove enabled from the conversation mapping")
    speaker = _SpeakerConfig._from_yaml(values.get("speaker"))
    conversation_values = values.get("conversation")
    if isinstance(conversation_values, dict) and explicit_mode:
        conversation_values = {k: v for k, v in conversation_values.items() if k != "enabled"}
    conversation = _ConversationConfig._from_yaml(conversation_values)
    if not explicit_mode and "conversation" not in raw and speaker is not None:
        # Preserve existing speaker-only YAML while separating the two concerns
        # for newly configured workers.
        conversation = _ConversationConfig(**{f.name: getattr(speaker, f.name) for f in fields(_ConversationConfig)})
    if explicit_mode:
        mode = raw["mode"]
    elif "conversation" in raw or ("speaker" in raw and speaker is not None):
        mode = "conversation" if conversation is not None else ("wake_only" if phrases else "always_on")
    elif "magic_phrases" in raw:
        mode = "wake_only" if phrases else "always_on"
    else:
        mode = defaults["mode"]
    if not isinstance(mode, str) or mode not in {"always_on", "wake_only", "conversation", "hybrid"}:
        raise ValueError("mode must be always_on, wake_only, conversation, or hybrid")
    if mode in {"wake_only", "hybrid"} and not phrases:
        raise ValueError(f"mode {mode} requires non-empty magic_phrases; use always_on for ungated speech")
    if mode in {"conversation", "hybrid"}:
        if conversation is None:
            if explicit_mode and "conversation" in raw:
                raise ValueError("conversation settings must be a mapping for conversation or hybrid mode")
            conversation = _ConversationConfig()
        if conversation.require_wake_phrase and not phrases:
            raise ValueError("conversation.require_wake_phrase requires non-empty magic_phrases")
    else:
        conversation = None
    if speaker is not None:
        controls = conversation or _ConversationConfig()
        speaker = replace(speaker, **{f.name: getattr(controls, f.name) for f in fields(_ConversationConfig)})
        speaker = replace(
            speaker, mode=mode, magic_phrases=phrases, followup_grace_s=cfg.followup_grace_s,
            stop_commands_enabled=cfg.stop_commands_enabled,
        )
        speaker._validate_interaction()
    object.__setattr__(cfg, "_mode", mode)
    object.__setattr__(cfg, "_speaker", speaker)
    object.__setattr__(cfg, "_conversation", conversation)
    return cfg


class AudioSink(Protocol):
    """Consumer-supplied return-audio writer."""

    async def play_wav(self, pid: str, wav_bytes: bytes) -> None:
        """Play WAV bytes for participant *pid*."""
        ...


class TTSLike(Protocol):
    """Duck-typed text-to-speech client used for ``say_stop_ack``."""

    async def synthesize(self, text: str) -> bytes:
        """Synthesize *text* and return WAV bytes."""
        ...
