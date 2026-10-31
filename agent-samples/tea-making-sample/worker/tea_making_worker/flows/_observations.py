# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate YAML observation policies and normalize model evidence.

Policies select bounded semantic choices, typed extraction or native-result
matching. Model output is evidence, never a state patch: ``_observe`` returns an
Observation and ``_resolve`` expands caller-approved bindings. Unit conversion,
output schemas and type checks stay code-owned; wording and criteria stay in YAML.
"""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from xr_ai_models import DecisionQuestion, DecisionService, LLMService
from xr_ai_tools import ToolSet


class _Unit(BaseModel):
    """Declare one finite linear conversion into a field's canonical unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    scale: float = 1
    offset: float = 0

    @model_validator(mode="after")
    def validate_scale(self) -> _Unit:
        """Reject conversions that could yield nonfinite or inverted measurements."""
        if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.offset):
            raise ValueError("unit scale must be finite and positive; offset must be finite")
        return self


class _Value(BaseModel):
    """Describe one extractable field and its accepted values or observed units."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "integer", "boolean"]
    description: str = Field(min_length=1)
    choices: list[str] | None = None
    unavailable_values: list[str] = Field(default_factory=list)
    minimum: float | None = None
    units: dict[str, _Unit] | None = None

    @model_validator(mode="after")
    def validate_units(self) -> _Value:
        """Require a usable numeric unit mapping and a finite optional lower bound."""
        if self.units is not None and (self.type not in {"number", "integer"} or not self.units):
            raise ValueError("units require a numeric field and finite positive scales")
        if self.minimum is not None and not math.isfinite(self.minimum):
            raise ValueError("minimum must be finite")
        return self

    def json_schema(self) -> dict[str, Any]:
        """Build the code-owned nullable extraction schema for this field."""
        if self.units is not None:
            return {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["value", "unit"],
                        "properties": {
                            "value": {"type": "number"},
                            "unit": {"type": "string", "enum": list(self.units)},
                        },
                    },
                ],
                "description": self.description,
            }
        schema: dict[str, Any] = {"type": [self.type, "null"], "description": self.description}
        if self.choices is not None:
            schema["enum"] = [*self.choices, None]
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        return schema

    def decode(self, value: Any) -> Any:
        """Normalize an observed value and reject unsupported units or field values.

        Configured unavailable strings become None. Numeric unit conversion runs in code;
        the model reports its observed unit rather than computing a replacement reading.
        """
        if isinstance(value, str) and value.strip().casefold() in {
            item.casefold() for item in self.unavailable_values
        }:
            return None
        if self.units is not None:
            if not isinstance(value, dict) or set(value) != {"value", "unit"}:
                raise ValueError("unit-bearing extraction needs value and unit")
            reading, unit = value["value"], value["unit"]
            if not isinstance(reading, (float, int)) or isinstance(reading, bool) or not math.isfinite(reading):
                raise ValueError("unit-bearing value must be finite numeric data")
            if not isinstance(unit, str) or unit not in self.units:
                raise ValueError("unsupported extracted unit")
            conversion = self.units[unit]
            value = (reading + conversion.offset) * conversion.scale
            if self.type == "integer" and math.isfinite(value) and float(value).is_integer():
                value = int(value)
        if not self.accepts(value):
            raise ValueError("extracted value does not satisfy the declared field")
        return value

    def accepts(self, value: Any) -> bool:
        """Check scalar type, finiteness, lower bound and any configured choices."""
        if self.type == "boolean":
            valid = isinstance(value, bool)
        elif self.type == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif self.type == "number":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            valid = valid and math.isfinite(value)
        if self.minimum is not None:
            valid = valid and isinstance(value, (float, int)) and value >= self.minimum
        return valid and (self.choices is None or value in self.choices)


class _Evaluation(BaseModel):
    """Bind extracted values to one configured native tool and exact result predicate."""

    model_config = ConfigDict(extra="forbid")
    function: str
    arguments: dict[str, Any]
    match: dict[str, Any]


class _Lookup(BaseModel):
    """Declare identity-based retrieval, missing-field fallback and provenance bindings."""

    model_config = ConfigDict(extra="forbid")
    identity_field: str
    query: str = Field(min_length=1)
    fields: list[str] = Field(min_length=1)
    instructions: str = Field(min_length=1)
    updates: dict[str, Any] = Field(default_factory=dict)
    fallback_instructions: str = ""
    source_fields: dict[str, str] = Field(default_factory=dict)


class ObservationPolicy(BaseModel):
    """External instructions, evidence choices and typed normalization bindings.

    Parse the YAML mapping with ``model_validate``. Choice policies require all
    three outcomes; extraction schemas and unit conversion remain code-owned.
    Update bindings describe proposals for the caller, never state writes here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["choice", "extract", "result"]
    instructions: str = ""
    structured: bool = False
    criteria: dict[str, str] = Field(default_factory=dict)
    fields: dict[str, _Value] = Field(default_factory=dict)
    visual_fields: list[str] | None = None
    retain: list[str] = Field(default_factory=list)
    match: dict[str, Any] = Field(default_factory=dict)
    evaluation: _Evaluation | None = None
    lookup: _Lookup | None = None
    updates: dict[str, Any]

    @model_validator(mode="after")
    def validate_policy(self) -> ObservationPolicy:
        """Validate choice labels, declared field references and identity-preservation rules."""
        if self.kind == "choice":
            if set(self.criteria) != {"accepted", "rejected", "unknown"}:
                raise ValueError("choice criteria must be accepted, rejected, and unknown")
            if not self.instructions.strip() or any(not criterion.strip() for criterion in self.criteria.values()):
                raise ValueError("choice instructions and criteria must not be blank")
        elif self.kind == "extract" and not self.fields:
            raise ValueError("extract observation needs fields")
        elif self.kind == "result" and not self.match:
            raise ValueError("result observation needs a match predicate")
        if self.visual_fields is not None and (
            not self.visual_fields or not set(self.visual_fields) <= self.fields.keys()
        ):
            raise ValueError("visual_fields must select declared fields")
        if not set(self.retain) <= self.fields.keys():
            raise ValueError("retained fields must be declared extraction fields")
        if self.lookup is not None:
            if self.kind != "extract" or self.lookup.identity_field not in self.fields:
                raise ValueError("lookup needs an extracted identity field")
            if not set(self.lookup.fields) <= self.fields.keys():
                raise ValueError("lookup fields must be declared extraction fields")
            if not set(self.lookup.source_fields) <= set(self.lookup.fields):
                raise ValueError("provenance must describe lookup fields")
            if self.lookup.identity_field in self.lookup.fields:
                raise ValueError("lookup cannot replace the observed identity")
        return self

    def extraction_question(self, question: str) -> str:
        """Combine caller wording with a code-owned evidence response contract.

        The output schema constrains observable facts, not workflow state or completion.
        Unanswerable views must remain unknown, while clearly visible negative evidence
        can still be assessed by the later choice policy.
        """
        if self.kind == "choice" and self.structured:
            return (
                question + '\nReturn only JSON: {"observable": boolean, "description": string}. '
                "Observable means the requested condition can be assessed. "
                "A clearly observed negative is still observable. "
                "If the requested view is missing, obscured, or unanswerable, set observable to false. "
                "Describe only visible evidence; never infer a relationship from co-presence."
            )
        if self.kind != "extract":
            return question
        fields = {name: self.fields[name] for name in (self.visual_fields or self.fields)}
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": list(fields),
            "properties": {name: field.json_schema() for name, field in fields.items()},
        }
        return (
            f"{question}\n{self.instructions}\n"
            "Return only a JSON object with these fields. Use null for an unavailable value. "
            "For a field declaring units, return {value: number, unit: unit-label} using the observed unit; "
            "code performs conversion. Never infer missing values. Do not output Markdown or workflow state.\n"
            + json.dumps(schema, ensure_ascii=False)
        )


@dataclass(frozen=True)
class Observation:
    """Evidence, normalized fields, provenance overrides and optional native result."""

    evidence: Literal["accepted", "rejected", "unknown"]
    values: dict[str, Any]
    overrides: dict[str, Any]
    result: dict[str, Any] | None = None


def _resolve(value: Any, state: dict[str, Any], observation: dict[str, Any], *, now_us: int | None = None) -> Any:
    """Expand configured value bindings against caller-supplied native evidence.

    State and observation references are reads. The clock binding is allowed only when
    the guarded commit caller supplies now_us, so uncertain sightings cannot start a
    timer early.
    """
    if isinstance(value, dict):
        return {key: _resolve(item, state, observation, now_us=now_us) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, state, observation, now_us=now_us) for item in value]
    if isinstance(value, str):
        if value.startswith("$state."):
            return state[value.removeprefix("$state.")]
        if value.startswith("$observation."):
            return observation[value.removeprefix("$observation.")]
        if value == "$clock.now_us":
            if now_us is None:
                raise ValueError("clock binding is only valid during a confirmed commit")
            return now_us
    return value


async def _observe(
    policy: ObservationPolicy,
    observation: Any,
    state: dict[str, Any],
    *,
    decisions: DecisionService,
    llm: LLMService,
    tools: ToolSet,
    timeout_s: float,
    on_event: Callable[..., Awaitable[None]] | None = None,
) -> Observation:
    """Evaluate a configured semantic choice, extraction or native-result predicate.

    Args:
        policy: Validated external evidence instructions and field bindings.
        observation: Visible evidence or a configured native trigger result.
        state: Established values exposed to this evaluation by the caller.
        decisions: Bounded semantic-choice service.
        llm: Language service for configured identity/settings resolution.
        tools: Restricted native evaluator and retrieval catalog.
        timeout_s: Timeout for model requests made by this evaluation.
        on_event: Optional asynchronous provenance recording callback.

    Returns:
        Evidence, normalized values and proposed overrides; no execution state is written.
    """
    overrides: dict[str, Any] = {}
    if policy.kind == "choice":
        response = await decisions.decide(
            {"observation": observation},
            {"evidence": DecisionQuestion(policy.instructions, policy.criteria)},
            timeout=timeout_s,
        )
        evidence = response.answers["evidence"].choice
        if evidence not in policy.criteria:
            raise ValueError(f"invalid evidence choice: {evidence!r}")
        return Observation(evidence, {}, {})
    if policy.kind == "result":
        values = observation
    else:
        values = _decode_values(policy, observation)
        values.update({name: state[name] for name in policy.retain if name in state})
        lookup = policy.lookup
        if lookup is not None:
            values.update({name: state[name] for name in lookup.fields if name in state})
            from ._identity import IdentityResolutionStep

            resolved = await IdentityResolutionStep(
                policy, llm=llm, tools=tools, timeout_s=timeout_s, on_event=on_event,
            ).check(values)
            return resolved

        if set(values) != policy.fields.keys():
            return Observation("unknown", values, {})
    if not isinstance(values, dict):
        raise ValueError("observation result must be a mapping")
    evaluated = None
    if policy.evaluation is not None:
        evaluation = policy.evaluation
        evaluation_tool = tools.get(evaluation.function)
        if evaluation_tool is None:
            raise ValueError(f"unknown evaluation tool: {evaluation.function}")
        evaluation_result = await evaluation_tool.execute(
            evaluation_tool.request_model.model_validate(_resolve(evaluation.arguments, state, values))
        )
        evaluated = evaluation_result.model_dump()
        accepted = all(
            type(evaluated.get(key)) is type(expected) and evaluated.get(key) == expected
            for key, expected in evaluation.match.items()
        )
    else:
        accepted = all(
            type(values.get(key)) is type(expected) and values.get(key) == expected
            for key, expected in policy.match.items()
        )
    return Observation("accepted" if accepted else "rejected", values, overrides, evaluated)


def _decode_values(policy: ObservationPolicy, observation: Any) -> dict[str, Any]:
    """Decode only declared visual fields, dropping explicitly unavailable values."""
    values = json.loads(observation) if isinstance(observation, str) else observation
    allowed = set(policy.visual_fields or policy.fields)
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError("visual extraction must contain only declared visual fields")
    decoded = {
        field_name: policy.fields[field_name].decode(value)
        for field_name, value in values.items()
        if value is not None
    }
    return {field_name: value for field_name, value in decoded.items() if value is not None}
