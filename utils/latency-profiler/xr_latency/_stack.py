# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build typed model services from a models profile, exactly as workers do."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xr_ai_models import (
    DecisionService,
    LLMService,
    ModelsConfig,
    TTSService,
    VLMService,
    load_models_config_from_dict,
    make_decision,
    make_llm,
    make_tts,
    make_vlm,
)

STACKS = Path(__file__).resolve().parents[1] / "stacks"


@dataclass
class Stack:
    """The model services available to a benchmark, keyed by profile role."""

    config: ModelsConfig
    raw: dict[str, Any]

    @classmethod
    def load(cls, name_or_path: str, host: str | None = None) -> Stack:
        """Load ``stacks/<name>.json`` or an explicit profile path.

        ``host`` replaces ``localhost`` / ``127.0.0.1`` in every URL, so a profile written for the machine that
        serves the models can be profiled from another machine on the network.
        """
        path = Path(name_or_path)
        if not path.exists():
            path = STACKS / f"{name_or_path}.json"
        text = path.read_text()
        if host:
            text = re.sub(r"//(localhost|127\.0\.0\.1)\b", f"//{host}", text)
        raw = json.loads(text)
        return cls(load_models_config_from_dict(raw["models"]), raw)

    def service_url(self, name: str) -> str | None:
        """Base URL of an auxiliary service declared under the profile's top-level ``services`` (e.g. speaker STT)."""
        return (self.raw.get("services") or {}).get(name)

    def endpoints(self) -> dict[str, str]:
        """Base URL of each OpenAI-compatible model role (``llm``, ``vlm``), for metrics and fingerprints."""
        out = {}
        for role in ("llm", "vlm"):
            url = self.raw["models"].get(role, {}).get("endpoint", {}).get("base_url")
            if url:
                out[role] = url
        return out

    def roles(self) -> list[str]:
        """Return the roles declared in the profile."""
        return list(self.raw["models"])

    def llm(self, role: str = "llm") -> LLMService:
        """Return the chat model service for ``role``."""
        return make_llm(self.config, role)

    def vlm(self, role: str = "vlm") -> VLMService:
        """Return the vision-language model service for ``role``."""
        return make_vlm(self.config, role)

    def decision(self, role: str = "router") -> DecisionService:
        """Return the decision (router) service for ``role``."""
        return make_decision(self.config, role)

    def tts(self, role: str = "tts") -> TTSService:
        """Return the text-to-speech service for ``role``."""
        return make_tts(self.config, role)
