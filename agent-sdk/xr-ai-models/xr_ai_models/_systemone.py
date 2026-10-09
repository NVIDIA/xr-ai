# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP client for choice-only SystemOne decision endpoints."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

import httpx

from ._openai_compat import _auth_headers, _http_health, _warn_if_cleartext_key
from ._protocols import DecisionAnswer, DecisionQuestion, DecisionResponse


class _SystemOneDecision:
    """HTTP adapter behind the public :class:`DecisionService` protocol."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key_env: str | None,
        timeout: float,
        health_check: bool,
        health_path: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        self.url = base + "/v1/systemone"
        self.health_url = base + health_path
        self.model_name = model_name
        self.timeout = timeout
        self._api_key = os.environ.get(api_key_env) if api_key_env else None
        _warn_if_cleartext_key(base_url, self._api_key)
        self._health_check = health_check
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=False)
        self._owns_client = client is None

    async def decide(
        self,
        state: str | Mapping[str, Any],
        questions: Mapping[str, DecisionQuestion],
        *,
        timeout: float | None = None,
    ) -> DecisionResponse:
        """Submit choice questions and validate every answer before returning."""

        if not isinstance(state, (str, Mapping)):
            raise TypeError("decision state must be text or a mapping")
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("at least one decision question is required")
        payload_questions: dict[str, dict[str, Any]] = {}
        for question_id, question in questions.items():
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError("decision question IDs must be non-empty strings")
            if not isinstance(question, DecisionQuestion):
                raise TypeError(f"question {question_id!r} must be a DecisionQuestion")
            if not isinstance(question.instructions, str):
                raise TypeError(f"question {question_id!r} instructions must be text")
            if not isinstance(question.criteria, Mapping) or not question.criteria:
                raise ValueError(f"question {question_id!r} criteria must be a non-empty mapping")
            criteria: dict[str, str] = {}
            for label, criterion in question.criteria.items():
                if not isinstance(label, str) or not label.strip():
                    raise ValueError(f"question {question_id!r} choice labels must be non-empty strings")
                if not isinstance(criterion, str) or not criterion.strip():
                    raise ValueError(f"question {question_id!r} criteria must be non-empty strings")
                criteria[label] = criterion
            if len(criteria) < 2:
                raise ValueError(f"question {question_id!r} must define at least two choices")
            payload_questions[question_id] = {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": criteria,
            }

        effective_timeout = self._bounded_timeout(timeout)
        state_payload = dict(state) if isinstance(state, Mapping) else state
        response = await self._client.post(
            self.url,
            json={"model": self.model_name, "state": state_payload, "questions": payload_questions},
            headers=_auth_headers(self._api_key),
            timeout=effective_timeout,
        )
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, dict):
            raise ValueError("SystemOne response must be a JSON object")
        if raw.get("model") != self.model_name:
            raise ValueError("SystemOne response model does not match the configured model")
        raw_answers = raw.get("answers")
        if not isinstance(raw_answers, dict) or raw_answers.keys() != payload_questions.keys():
            raise ValueError("SystemOne response must contain exactly one answer for every question")
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("SystemOne response usage must be an object")

        answers: dict[str, DecisionAnswer] = {}
        for question_id, question_payload in payload_questions.items():
            item = raw_answers[question_id]
            if not isinstance(item, dict) or item.get("type") != "choice":
                raise ValueError(f"SystemOne answer {question_id!r} must have type 'choice'")
            criteria = question_payload["criteria"]
            choice = item.get("choice")
            if not isinstance(choice, str) or choice not in criteria:
                raise ValueError(f"SystemOne answer {question_id!r} selected an unknown choice")
            confidence = self._probability(item.get("confidence"), f"{question_id!r} confidence")
            raw_probabilities = item.get("probabilities")
            if not isinstance(raw_probabilities, dict) or raw_probabilities.keys() != criteria.keys():
                raise ValueError(f"SystemOne answer {question_id!r} probabilities must match all choices")
            probabilities = {
                label: self._probability(value, f"{question_id!r} probability for {label!r}")
                for label, value in raw_probabilities.items()
            }
            if not math.isclose(math.fsum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-3):
                raise ValueError(f"SystemOne answer {question_id!r} probabilities must sum to one")
            if not math.isclose(confidence, probabilities[choice], rel_tol=0.0, abs_tol=1e-3):
                raise ValueError(f"SystemOne answer {question_id!r} confidence must match its selected choice")
            answers[question_id] = DecisionAnswer(
                choice=choice,
                confidence=confidence,
                probabilities=probabilities,
            )
        return DecisionResponse(
            answers=answers,
            model=self.model_name,
            usage=usage,
            raw=raw,
        )

    def _bounded_timeout(self, timeout: float | None) -> float:
        if timeout is None:
            return self.timeout
        if isinstance(timeout, bool):
            raise ValueError("decision timeout must be a positive finite number")
        try:
            requested = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("decision timeout must be a positive finite number") from exc
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError("decision timeout must be a positive finite number")
        return min(requested, self.timeout)

    @staticmethod
    def _probability(value: Any, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"SystemOne {field} must be a finite number between zero and one")
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"SystemOne {field} must be a finite number between zero and one")
        return probability

    async def health(self) -> bool:
        """Probe the configured endpoint readiness URL."""

        return await _http_health(self._client, self.health_url, self._health_check)

    async def close(self) -> None:
        """Close the internally created HTTP client, if the adapter owns it."""

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> _SystemOneDecision:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
