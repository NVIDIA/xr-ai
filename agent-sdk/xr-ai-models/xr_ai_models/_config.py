# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed model roles, endpoints, deployment metadata, and profile loading."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

from . import presets as _presets
from ._utils import merge_dicts


Category = Literal["llm", "vlm", "stt", "tts", "embedding"]
ModelKind = Literal["openai_compat"]
Readiness = Literal["health", "none"]
Ownership = Literal["managed", "reused", "external"]

KIND_OPENAI_COMPAT: ModelKind = "openai_compat"


@dataclass(frozen=True)
class AdapterSpec:
    """API dialect and model-specific request/response behavior."""

    kind: ModelKind = KIND_OPENAI_COMPAT
    model_name: str = ""
    reasoning_field: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    default_extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EndpointSpec:
    """Connectivity, authentication, timeout, and readiness for an adapter."""

    base_url: str = ""
    api_key_env: str | None = None
    timeout: float = 60.0
    readiness: Readiness = "health"

    @property
    def health_check(self) -> bool:
        return self.readiness == "health"


@dataclass(frozen=True)
class DeploymentSpec:
    """Process ownership for the endpoint that serves a model role."""

    ownership: Ownership = "external"
    service: str | None = None


class _RoleSpec:
    """Read-only compatibility aliases for flat role-spec attributes."""

    adapter: AdapterSpec
    endpoint: EndpointSpec

    @property
    def kind(self) -> ModelKind:
        return self.adapter.kind

    @property
    def model_name(self) -> str:
        return self.adapter.model_name

    @property
    def reasoning_field(self) -> str | None:
        return self.adapter.reasoning_field

    @property
    def capabilities(self) -> dict[str, Any]:
        return self.adapter.capabilities

    @property
    def default_extras(self) -> dict[str, Any]:
        return self.adapter.default_extras

    @property
    def base_url(self) -> str:
        return self.endpoint.base_url

    @property
    def api_key_env(self) -> str | None:
        return self.endpoint.api_key_env

    @property
    def timeout(self) -> float:
        return self.endpoint.timeout

    @property
    def health_check(self) -> bool:
        return self.endpoint.health_check

    def _set_specs(
        self,
        adapter: AdapterSpec,
        endpoint: EndpointSpec,
        deployment: DeploymentSpec,
    ) -> None:
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "deployment", deployment)


def _structured_specs(
    adapter: AdapterSpec | None,
    endpoint: EndpointSpec | None,
    deployment: DeploymentSpec | None,
    *,
    default_timeout: float,
) -> tuple[AdapterSpec, EndpointSpec, DeploymentSpec] | None:
    if adapter is None and endpoint is None:
        return None
    return (
        adapter or AdapterSpec(),
        endpoint or EndpointSpec(timeout=default_timeout),
        deployment or DeploymentSpec(),
    )


def _reject_mixed_construction(**legacy_nondefault: bool) -> None:
    mixed = [name for name, nondefault in legacy_nondefault.items() if nondefault]
    if mixed:
        raise TypeError(
            "cannot mix structured model specs with legacy fields: "
            + ", ".join(mixed)
        )


@dataclass(frozen=True, init=False)
class LLMSpec(_RoleSpec):
    adapter: AdapterSpec = field(default_factory=AdapterSpec)
    endpoint: EndpointSpec = field(default_factory=EndpointSpec)
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)

    def __init__(
        self,
        kind: ModelKind = KIND_OPENAI_COMPAT,
        base_url: str = "",
        model_name: str = "",
        api_key_env: str | None = None,
        reasoning_field: str | None = None,
        capabilities: dict[str, Any] | None = None,
        default_extras: dict[str, Any] | None = None,
        timeout: float = 60.0,
        health_check: bool = True,
        deployment: DeploymentSpec | None = None,
        *,
        adapter: AdapterSpec | None = None,
        endpoint: EndpointSpec | None = None,
    ) -> None:
        structured = _structured_specs(
            adapter,
            endpoint,
            deployment,
            default_timeout=60.0,
        )
        if structured is not None:
            _reject_mixed_construction(
                kind=kind != KIND_OPENAI_COMPAT,
                base_url=bool(base_url),
                model_name=bool(model_name),
                api_key_env=api_key_env is not None,
                reasoning_field=reasoning_field is not None,
                capabilities=capabilities is not None,
                default_extras=default_extras is not None,
                timeout=timeout != 60.0,
                health_check=health_check is not True,
            )
            self._set_specs(*structured)
            return
        self._set_specs(
            AdapterSpec(
                kind=kind,
                model_name=model_name,
                reasoning_field=reasoning_field,
                capabilities=capabilities or {},
                default_extras=default_extras or {},
            ),
            EndpointSpec(
                base_url=base_url,
                api_key_env=api_key_env,
                timeout=timeout,
                readiness="health" if health_check else "none",
            ),
            deployment or DeploymentSpec(),
        )


@dataclass(frozen=True, init=False)
class VLMSpec(_RoleSpec):
    adapter: AdapterSpec = field(default_factory=AdapterSpec)
    endpoint: EndpointSpec = field(default_factory=EndpointSpec)
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)

    def __init__(
        self,
        kind: ModelKind = KIND_OPENAI_COMPAT,
        base_url: str = "",
        model_name: str = "",
        api_key_env: str | None = None,
        capabilities: dict[str, Any] | None = None,
        default_extras: dict[str, Any] | None = None,
        timeout: float = 60.0,
        health_check: bool = True,
        deployment: DeploymentSpec | None = None,
        *,
        adapter: AdapterSpec | None = None,
        endpoint: EndpointSpec | None = None,
    ) -> None:
        structured = _structured_specs(
            adapter,
            endpoint,
            deployment,
            default_timeout=60.0,
        )
        if structured is not None:
            _reject_mixed_construction(
                kind=kind != KIND_OPENAI_COMPAT,
                base_url=bool(base_url),
                model_name=bool(model_name),
                api_key_env=api_key_env is not None,
                capabilities=capabilities is not None,
                default_extras=default_extras is not None,
                timeout=timeout != 60.0,
                health_check=health_check is not True,
            )
            self._set_specs(*structured)
            return
        self._set_specs(
            AdapterSpec(
                kind=kind,
                model_name=model_name,
                capabilities=capabilities or {},
                default_extras=default_extras or {},
            ),
            EndpointSpec(
                base_url=base_url,
                api_key_env=api_key_env,
                timeout=timeout,
                readiness="health" if health_check else "none",
            ),
            deployment or DeploymentSpec(),
        )


@dataclass(frozen=True, init=False)
class STTSpec(_RoleSpec):
    adapter: AdapterSpec = field(default_factory=AdapterSpec)
    endpoint: EndpointSpec = field(
        default_factory=lambda: EndpointSpec(timeout=30.0)
    )
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)

    def __init__(
        self,
        kind: ModelKind = KIND_OPENAI_COMPAT,
        base_url: str = "",
        api_key_env: str | None = None,
        timeout: float = 30.0,
        health_check: bool = True,
        deployment: DeploymentSpec | None = None,
        *,
        adapter: AdapterSpec | None = None,
        endpoint: EndpointSpec | None = None,
    ) -> None:
        structured = _structured_specs(
            adapter,
            endpoint,
            deployment,
            default_timeout=30.0,
        )
        if structured is not None:
            _reject_mixed_construction(
                kind=kind != KIND_OPENAI_COMPAT,
                base_url=bool(base_url),
                api_key_env=api_key_env is not None,
                timeout=timeout != 30.0,
                health_check=health_check is not True,
            )
            self._set_specs(*structured)
            return
        self._set_specs(
            AdapterSpec(kind=kind),
            EndpointSpec(
                base_url=base_url,
                api_key_env=api_key_env,
                timeout=timeout,
                readiness="health" if health_check else "none",
            ),
            deployment or DeploymentSpec(),
        )
@dataclass(frozen=True, init=False)
class TTSSpec(_RoleSpec):
    adapter: AdapterSpec = field(default_factory=AdapterSpec)
    endpoint: EndpointSpec = field(
        default_factory=lambda: EndpointSpec(timeout=30.0)
    )
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)

    def __init__(
        self,
        kind: ModelKind = KIND_OPENAI_COMPAT,
        base_url: str = "",
        api_key_env: str | None = None,
        timeout: float = 30.0,
        health_check: bool = True,
        deployment: DeploymentSpec | None = None,
        *,
        adapter: AdapterSpec | None = None,
        endpoint: EndpointSpec | None = None,
    ) -> None:
        structured = _structured_specs(
            adapter,
            endpoint,
            deployment,
            default_timeout=30.0,
        )
        if structured is not None:
            _reject_mixed_construction(
                kind=kind != KIND_OPENAI_COMPAT,
                base_url=bool(base_url),
                api_key_env=api_key_env is not None,
                timeout=timeout != 30.0,
                health_check=health_check is not True,
            )
            self._set_specs(*structured)
            return
        self._set_specs(
            AdapterSpec(kind=kind),
            EndpointSpec(
                base_url=base_url,
                api_key_env=api_key_env,
                timeout=timeout,
                readiness="health" if health_check else "none",
            ),
            deployment or DeploymentSpec(),
        )


@dataclass(frozen=True, init=False)
class EmbeddingSpec(_RoleSpec):
    adapter: AdapterSpec = field(default_factory=AdapterSpec)
    endpoint: EndpointSpec = field(default_factory=EndpointSpec)
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)

    def __init__(
        self,
        kind: ModelKind = KIND_OPENAI_COMPAT,
        base_url: str = "",
        model_name: str = "",
        api_key_env: str | None = None,
        timeout: float = 60.0,
        health_check: bool = True,
        deployment: DeploymentSpec | None = None,
        *,
        adapter: AdapterSpec | None = None,
        endpoint: EndpointSpec | None = None,
    ) -> None:
        structured = _structured_specs(
            adapter,
            endpoint,
            deployment,
            default_timeout=60.0,
        )
        if structured is not None:
            _reject_mixed_construction(
                kind=kind != KIND_OPENAI_COMPAT,
                base_url=bool(base_url),
                model_name=bool(model_name),
                api_key_env=api_key_env is not None,
                timeout=timeout != 60.0,
                health_check=health_check is not True,
            )
            self._set_specs(*structured)
            return
        self._set_specs(
            AdapterSpec(kind=kind, model_name=model_name),
            EndpointSpec(
                base_url=base_url,
                api_key_env=api_key_env,
                timeout=timeout,
                readiness="health" if health_check else "none",
            ),
            deployment or DeploymentSpec(),
        )


Spec = LLMSpec | VLMSpec | STTSpec | TTSSpec | EmbeddingSpec
T = TypeVar("T", LLMSpec, VLMSpec, STTSpec, TTSSpec, EmbeddingSpec)


@dataclass(frozen=True)
class ModelsConfig:
    """Logical-name to typed model-role specifications."""

    entries: dict[str, Spec]

    def llm(self, name: str) -> LLMSpec:
        return _typed(self.entries, name, LLMSpec)

    def vlm(self, name: str) -> VLMSpec:
        return _typed(self.entries, name, VLMSpec)

    def stt(self, name: str) -> STTSpec:
        return _typed(self.entries, name, STTSpec)

    def tts(self, name: str) -> TTSSpec:
        return _typed(self.entries, name, TTSSpec)

    def embedding(self, name: str) -> EmbeddingSpec:
        return _typed(self.entries, name, EmbeddingSpec)

    @property
    def required_credentials(self) -> tuple[str, ...]:
        return tuple(sorted({
            spec.endpoint.api_key_env
            for spec in self.entries.values()
            if spec.endpoint.api_key_env
        }))


def _typed(entries: dict[str, Spec], name: str, cls: type[T]) -> T:
    try:
        spec = entries[name]
    except KeyError as exc:
        raise KeyError(f"no spec named {name!r} in models config") from exc
    if not isinstance(spec, cls):
        raise TypeError(
            f"spec {name!r} is {type(spec).__name__}, expected {cls.__name__}"
        )
    return spec


def load_models_config(path: Path | str) -> ModelsConfig:
    """Load a JSON or YAML model profile and resolve adapter presets."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    return load_models_config_from_dict(raw, source=str(path))


def load_models_config_from_dict(
    raw: dict[str, Any], *, source: str = "<dict>"
) -> ModelsConfig:
    """Build a model configuration from a parsed direct or ``models`` mapping."""

    if not isinstance(raw, dict):
        raise ValueError(f"{source}: top-level must be a mapping")
    models = raw.get("models", raw)
    if not isinstance(models, dict):
        raise ValueError(f"{source}: 'models' must be a mapping")

    entries: dict[str, Spec] = {}
    for name, body in models.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{source}: model role names must be non-empty strings")
        if not isinstance(body, dict):
            raise ValueError(f"{source}: entry {name!r} must be a mapping")
        try:
            entries[name] = _build_spec(_flatten_entry(body))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{source}: entry {name!r}: {exc}") from exc
    return ModelsConfig(entries=entries)


def _flatten_entry(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize nested and transitional mixed entries to the legacy shape."""

    if not any(key in body for key in ("adapter", "endpoint", "deployment")):
        return dict(body)
    if "api_key_env" in body:
        raise ValueError(
            "structured profiles must declare api_key_env in endpoint"
        )

    sections: dict[str, dict[str, Any]] = {}
    for label in ("adapter", "endpoint", "deployment"):
        value = body.get(label, {})
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a mapping")
        sections[label] = dict(value)

    adapter = sections["adapter"]
    if "preset" in adapter:
        preset = adapter.pop("preset")
        if not isinstance(preset, str) or not preset:
            raise ValueError("adapter preset must be a non-empty string")
        if "kind" in adapter:
            raise ValueError("adapter cannot declare both preset and kind")
        adapter["kind"] = f"preset:{preset}"

    flattened = {
        key: value
        for key, value in body.items()
        if key not in {"adapter", "endpoint", "deployment"}
    }
    for label in ("adapter", "endpoint"):
        overlap = flattened.keys() & sections[label].keys()
        if overlap:
            fields = ", ".join(sorted(overlap))
            raise ValueError(
                f"{label} fields also declared at role level: {fields}"
            )
        flattened.update(sections[label])
    flattened["deployment"] = sections["deployment"]
    return flattened


def _build_spec(body: dict[str, Any]) -> Spec:
    resolved, preset_category = _resolve_preset(body)
    explicit_category = resolved.get("category")
    if (
        preset_category is not None
        and explicit_category is not None
        and explicit_category != preset_category
    ):
        raise ValueError(
            f"category mismatch: preset gave {preset_category!r}, "
            f"entry gave {explicit_category!r}"
        )
    category = preset_category or explicit_category
    if category not in {"llm", "vlm", "stt", "tts", "embedding"}:
        raise ValueError(
            f"missing or unknown category {category!r}; "
            "set category when not using a preset"
        )
    return _construct(category, resolved)


def _resolve_preset(body: dict[str, Any]) -> tuple[dict[str, Any], Category | None]:
    kind = body.get("kind", KIND_OPENAI_COMPAT)
    if not isinstance(kind, str):
        raise ValueError("adapter kind must be a string")
    if not kind.startswith("preset:"):
        return dict(body), None
    preset_name = kind.split(":", 1)[1]
    if not preset_name:
        raise ValueError("preset name must be non-empty")
    preset = _presets.get_preset(preset_name)
    merged = merge_dicts(preset, body, skip_keys=("kind",))
    merged["kind"] = preset.get("kind", KIND_OPENAI_COMPAT)
    return merged, preset["category"]


def _construct(category: Category, body: dict[str, Any]) -> Spec:
    kind = body.get("kind", KIND_OPENAI_COMPAT)
    if kind != KIND_OPENAI_COMPAT:
        raise ValueError(f"unsupported adapter kind: {kind!r}")

    endpoint = EndpointSpec(
        base_url=_require_str(body, "base_url"),
        api_key_env=_optional_str(body, "api_key_env"),
        timeout=_timeout(body, category),
        readiness=_readiness(body),
    )
    adapter = AdapterSpec(
        kind=KIND_OPENAI_COMPAT,
        model_name=(
            _require_str(body, "model_name")
            if category in ("llm", "vlm", "embedding")
            else ""
        ),
        reasoning_field=_optional_str(body, "reasoning_field"),
        capabilities=_mapping(body, "capabilities"),
        default_extras=_mapping(body, "default_extras"),
    )
    deployment = _deployment(body.get("deployment", {}))

    if category == "llm":
        return LLMSpec(adapter=adapter, endpoint=endpoint, deployment=deployment)
    if category == "vlm":
        return VLMSpec(adapter=adapter, endpoint=endpoint, deployment=deployment)
    if category == "stt":
        return STTSpec(adapter=adapter, endpoint=endpoint, deployment=deployment)
    if category == "tts":
        return TTSSpec(adapter=adapter, endpoint=endpoint, deployment=deployment)
    if category == "embedding":
        return EmbeddingSpec(
            adapter=adapter,
            endpoint=endpoint,
            deployment=deployment,
        )
    raise AssertionError(category)


def _readiness(body: dict[str, Any]) -> Readiness:
    legacy = body.get("health_check")
    if "health_check" in body and not isinstance(legacy, bool):
        raise ValueError("health_check must be a boolean")

    readiness = body.get("readiness")
    if readiness is None:
        return "health" if legacy is not False else "none"
    if readiness not in {"health", "none"}:
        raise ValueError(f"unsupported readiness policy: {readiness!r}")
    if "health_check" in body and (readiness == "health") != legacy:
        raise ValueError("readiness conflicts with health_check")
    return readiness


def _deployment(value: Any) -> DeploymentSpec:
    if not isinstance(value, dict):
        raise ValueError("deployment must be a mapping")
    ownership = value.get("ownership", "external")
    if ownership not in {"managed", "reused", "external"}:
        raise ValueError(f"unsupported deployment ownership: {ownership!r}")
    service = _optional_str(value, "service")
    if ownership != "external" and service is None:
        raise ValueError(f"{ownership} deployments require a service name")
    return DeploymentSpec(ownership=ownership, service=service)


def _timeout(body: dict[str, Any], category: Category) -> float:
    default = 60.0 if category in ("llm", "vlm", "embedding") else 30.0
    value = body.get("timeout", default)
    if isinstance(value, bool):
        raise ValueError("timeout must be a positive number")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be a positive number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive number")
    return timeout


def _mapping(body: dict[str, Any], key: str) -> dict[str, Any]:
    value = body.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return dict(value)


def _optional_str(body: dict[str, Any], key: str) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_str(body: dict[str, Any], key: str) -> str:
    value = _optional_str(body, key)
    if value is None:
        raise ValueError(f"missing required string field {key!r}")
    return value
