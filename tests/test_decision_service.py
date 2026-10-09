# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the typed SystemOne decision client and model config."""

from __future__ import annotations

from types import MappingProxyType

import httpx
import pytest
from xr_ai_models import (
    DecisionQuestion,
    DecisionSpec,
    _systemone,
    load_models_config_from_dict,
    make_decision,
)


def _config():
    return load_models_config_from_dict(
        {
            "clef": {
                "category": "decision",
                "kind": "systemone",
                "model_name": "Cloudflare/Clef",
                "base_url": "http://clef.test:8003",
                "api_key_env": "CLEF_TEST_TOKEN",
                "timeout": 9.0,
                "health_path": "/health",
            }
        }
    )


@pytest.mark.asyncio
async def test_make_decision_resolves_config_and_validates_systemone_wire_contract(monkeypatch) -> None:
    config = _config()
    assert isinstance(config.decision("clef"), DecisionSpec)
    assert config.decision("clef").model_name == "Cloudflare/Clef"
    monkeypatch.setenv("CLEF_TEST_TOKEN", "test-token")
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/health":
            return httpx.Response(200)
        assert request.url.path == "/v1/systemone"
        assert request.headers["authorization"] == "Bearer test-token"
        return httpx.Response(
            200,
            json={
                "model": "Cloudflare/Clef",
                "answers": {
                    "scope": {
                        "type": "choice",
                        "choice": "electronics",
                        "confidence": 0.8125,
                        "probabilities": {"general": 0.1875, "electronics": 0.8125},
                    }
                },
                "usage": {"input_tokens": 27, "output_tokens": 0},
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        _systemone.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    service = make_decision(config, "clef")
    try:
        assert await service.health()
        result = await service.decide(
            MappingProxyType({"request": "Which domain owns this?"}),
            {
                "scope": DecisionQuestion(
                    instructions="Select the best destination.",
                    criteria={
                        "general": "Ordinary broad conversation.",
                        "electronics": "A focused question about consumer electronics.",
                    },
                )
            },
            timeout=1.5,
        )
    finally:
        await service.close()

    assert result.answers["scope"].choice == "electronics"
    assert result.answers["scope"].confidence == 0.8125
    assert result.answers["scope"].probabilities == {"general": 0.1875, "electronics": 0.8125}
    assert result.model == "Cloudflare/Clef"
    assert result.usage == {"input_tokens": 27, "output_tokens": 0}
    request = next(item for item in seen if item.url.path == "/v1/systemone")
    assert request.read() == (
        b'{"model":"Cloudflare/Clef","state":{"request":"Which domain owns this?"},'
        b'"questions":{"scope":{"type":"choice","instructions":"Select the best destination.",'
        b'"criteria":{"general":"Ordinary broad conversation.",'
        b'"electronics":"A focused question about consumer electronics."}}}}'
    )
    assert request.extensions["timeout"]["read"] == 1.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        {"type": "choice", "choice": "unknown", "confidence": 0.5, "probabilities": {"yes": 0.5, "no": 0.5}},
        {"type": "choice", "choice": "yes", "confidence": 0.5, "probabilities": {"yes": 0.5}},
        {"type": "choice", "choice": "yes", "confidence": 0.5, "probabilities": {"yes": float("nan"), "no": 0.5}},
        {"type": "choice", "choice": "yes", "confidence": 0.7, "probabilities": {"yes": 0.7, "no": 0.1}},
        {"type": "score", "choice": "yes", "confidence": 0.5, "probabilities": {"yes": 0.5, "no": 0.5}},
    ],
)
async def test_decision_rejects_malformed_or_unexpected_answer_data(answer) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"model": "Cloudflare/Clef", "answers": {"q": answer}, "usage": {}},
            )
        )
    )
    service = _systemone._SystemOneDecision(
        base_url="http://clef.test:8003",
        model_name="Cloudflare/Clef",
        api_key_env=None,
        timeout=5,
        health_check=False,
        health_path="/health",
        client=client,
    )
    try:
        with pytest.raises(ValueError):
            await service.decide(
                "state",
                {"q": DecisionQuestion(instructions="choose", criteria={"yes": "yes", "no": "no"})},
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_decision_rejects_partial_question_set() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"model": "Cloudflare/Clef", "answers": {}, "usage": {}},
            )
        )
    )
    service = _systemone._SystemOneDecision(
        base_url="http://clef.test:8003",
        model_name="Cloudflare/Clef",
        api_key_env=None,
        timeout=5,
        health_check=False,
        health_path="/health",
        client=client,
    )
    try:
        with pytest.raises(ValueError, match="exactly one answer"):
            await service.decide(
                "state",
                {"q": DecisionQuestion(instructions="choose", criteria={"yes": "yes", "no": "no"})},
            )
    finally:
        await client.aclose()


def test_systemone_kind_is_restricted_to_decision_category() -> None:
    with pytest.raises(ValueError, match="decision kind"):
        load_models_config_from_dict({"llm": {"category": "llm", "kind": "systemone", "base_url": "http://clef.test"}})
    with pytest.raises(ValueError, match="requires the systemone"):
        load_models_config_from_dict(
            {"decision": {"category": "decision", "kind": "openai_compat", "base_url": "http://clef.test"}}
        )
