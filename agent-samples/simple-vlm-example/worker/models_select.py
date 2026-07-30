# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-role model backend selection for simple-vlm-example.

``model_backend`` in the worker YAML routes each model role (stt, tts, vlm)
to a models file: ``local`` reads the file named by ``models_yaml`` (default
models.yaml), ``nim`` reads models.nim.yaml, ``nim_local`` reads
models.nim_local.yaml. A scalar value applies to every role, except that
scalar ``nim`` keeps stt/tts local (hosted NIM speech is not OpenAI
/v1/audio-compatible); the map form sets roles individually, with an
optional ``default`` key for unlisted roles (``local`` when absent). Kept in
sync with the same module in xr-render-demo/worker (deliberate per-sample
duplication).
"""
from __future__ import annotations

import pathlib

import yaml
from loguru import logger
from xr_ai_models import ModelsConfig, load_models_config_from_dict

_ROLES = ("stt", "tts", "vlm")
# Model entries each role selects from its backend's models file.
_ROLE_ENTRIES: dict[str, tuple[str, ...]] = {role: (role,) for role in _ROLES}
_BACKENDS = ("local", "nim", "nim_local")

# A role name at the top level of the worker YAML almost always means the
# model_backend map lost its indentation.
_MISINDENT_KEYS = ("stt", "tts", "llm", "vlm")


def model_backends(cfg: dict) -> dict[str, str]:
    """Resolve ``model_backend`` in ``cfg`` to a role → backend map."""
    misplaced = [k for k in _MISINDENT_KEYS if k in cfg]
    if misplaced:
        logger.warning(
            "worker config has top-level key(s) {}: this looks like a "
            "mis-indented model_backend map; indent them under model_backend:",
            misplaced,
        )
    raw = cfg.get("model_backend", "local")
    if isinstance(raw, str):
        backends = _expand_scalar(raw.lower())
    elif isinstance(raw, dict):
        default = str(raw.get("default", "local")).lower()
        backends = {role: str(raw.get(role, default)).lower() for role in _ROLES}
    else:
        raise ValueError(
            f"model_backend must be a string or a mapping, got {type(raw).__name__}"
        )
    for role, backend in backends.items():
        if backend not in _BACKENDS:
            raise ValueError(
                f"model_backend for role {role!r} is {backend!r}; "
                f"expected one of {', '.join(_BACKENDS)}"
            )
    return backends


def _expand_scalar(value: str) -> dict[str, str]:
    # Scalar `nim` keeps speech local; write the map form with an explicit
    # `default: nim` to host every role.
    if value == "nim":
        return {role: "local" if role in ("stt", "tts") else "nim"
                for role in _ROLES}
    return {role: value for role in _ROLES}


def compose_models_config(
    cfg: dict, config_path: pathlib.Path | None,
) -> ModelsConfig:
    """Build the effective models config, taking each role's entries from
    the file its backend selects. Each file is read at most once and only
    when some role uses it."""
    backends = model_backends(cfg)
    files = {
        "local": str(cfg.get("models_yaml", "models.yaml")),
        "nim": "models.nim.yaml",
        "nim_local": "models.nim_local.yaml",
    }
    loaded: dict[pathlib.Path, dict] = {}
    entries: dict[str, dict] = {}
    for role in _ROLES:
        backend = backends[role]
        path = _resolve(config_path, files[backend])
        if path not in loaded:
            try:
                raw = yaml.safe_load(path.read_text()) or {}
            except OSError as exc:
                raise ValueError(
                    f"model_backend {backend!r} for role {role!r} needs {path}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}: top-level must be a mapping")
            loaded[path] = raw
        for entry in _ROLE_ENTRIES[role]:
            if entry not in loaded[path]:
                raise ValueError(
                    f"{path} has no {entry!r} entry "
                    f"(needed by role {role!r} with backend {backend!r})"
                )
            entries[entry] = loaded[path][entry]
    return load_models_config_from_dict(entries, source="model_backend composition")


def _resolve(config_path: pathlib.Path | None, raw: str) -> pathlib.Path:
    p = pathlib.Path(raw)
    if config_path and not p.is_absolute():
        p = config_path.parent / p
    return p
