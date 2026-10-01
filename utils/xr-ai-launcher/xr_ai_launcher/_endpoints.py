# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded checks for configured model endpoints."""

from __future__ import annotations

import os
import socket
from collections.abc import Collection
from dataclasses import astuple
from http.client import HTTPException
from ipaddress import ip_address
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import SplitResult, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from ._checks import _row
from ._models import _load_model_endpoints, _ModelEndpoint

_TIMEOUT = 3.0
_START_MODELS = (
    "From the repository root, start the stack with `uv run --project "
    "model-server-samples/model-servers "
    "model_servers` or `uv run --project model-server-samples/model-servers-nim "
    "model_servers_nim`, or fix the configured endpoint."
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = build_opener(ProxyHandler({}), _NoRedirect)


def _is_loopback(parts: SplitResult) -> bool:
    hostname = parts.hostname
    try:
        return hostname == "localhost" or bool(hostname and ip_address(hostname).is_loopback)
    except ValueError:
        return False


def _http_parts(endpoint: _ModelEndpoint, profile: Path) -> tuple[SplitResult | None, dict[str, object] | None]:
    try:
        parts = urlsplit(endpoint.base_url)
        valid_port = parts.port is None or 1 <= parts.port <= 65535
    except ValueError:
        parts = None
        valid_port = False
    health_path = endpoint.health_path
    if (
        parts is None
        or parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or not valid_port
        or (
            health_path is not None
            and (not health_path.startswith("/") or "?" in health_path or "#" in health_path)
        )
    ):
        return None, _row(
            f"endpoint:{endpoint.role}", False, "invalid HTTP endpoint URL or health path",
            "valid HTTP endpoint URL and absolute health path",
            f"Fix base_url or health_path for {endpoint.role} in {profile}.",
        )
    return parts, None


def _remediation(endpoint: _ModelEndpoint, profile: Path, parts: SplitResult) -> str:
    if _is_loopback(parts):
        return _START_MODELS
    return f"Fix the configured endpoint for {endpoint.role} in {profile}."


def _probe_health(endpoint: _ModelEndpoint, profile: Path, parts: SplitResult) -> dict[str, object]:
    health_path = endpoint.health_path or "/health"
    url = endpoint.base_url.rstrip("/") + health_path
    remediation = _remediation(endpoint, profile, parts)
    token = os.environ.get(endpoint.api_key_env, "") if endpoint.api_key_env else ""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        request = Request(url, headers=headers, method="GET")
        with _OPENER.open(request, timeout=_TIMEOUT) as response:
            status = response.status
    except HTTPError as exc:
        status = exc.code
    except (HTTPException, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        return _row(
            f"endpoint:{endpoint.role}", False,
            f"health endpoint unreachable ({type(reason).__name__})",
            "configured health endpoint reachable", remediation,
        )
    return _row(
        f"endpoint:{endpoint.role}", 200 <= status < 300, f"HTTP {status} from {url}",
        "2xx response from configured health endpoint", "" if 200 <= status < 300 else remediation,
    )


def _probe_reachability(endpoint: _ModelEndpoint, profile: Path, parts: SplitResult) -> dict[str, object]:
    hostname = parts.hostname
    assert hostname is not None
    port = parts.port or (443 if parts.scheme == "https" else 80)
    address = f"{hostname}:{port}"
    try:
        with socket.create_connection((hostname, port), timeout=_TIMEOUT):
            pass
    except OSError as exc:
        return _row(
            f"endpoint:{endpoint.role}", False,
            f"unreachable at {address} ({type(exc).__name__})", "configured endpoint reachable",
            _remediation(endpoint, profile, parts),
        )
    return _row(
        f"endpoint:{endpoint.role}", None, f"TCP reachable at {address}; health unverified",
        "known health endpoint for health verification", "",
    )


def endpoint_checks(
    profile: Path | None,
    owned_services: Collection[str] = (),
) -> list[dict[str, object]]:
    """Check literal endpoint connectivity fields in the selected model profile."""
    if profile is None:
        return []
    path = Path(profile).resolve()
    if path.suffix.lower() != ".json" and path.is_file():
        return [
            _row(
                f"endpoint-profile:{path.name}", None,
                "not checked: launcher endpoint checks require a JSON model profile",
                "JSON endpoint profile", "",
            )
        ]
    try:
        endpoints = _load_model_endpoints(path)
    except (AttributeError, TypeError, ValueError):
        return [
            _row(
                f"endpoint-profile:{path.name}", False, "missing or invalid model profile",
                "readable model profile with literal endpoint fields", f"Fix {path}.",
            )
        ]

    rows: list[dict[str, object]] = []
    probes: dict[tuple[object, ...], dict[str, object]] = {}
    for endpoint in endpoints:
        credential = endpoint.api_key_env
        name = f"endpoint:{endpoint.role}"
        if credential and not os.environ.get(credential):
            rows.append(_row(name, False, f"credential {credential} is not set",
                             f"credential {credential} set", f"Set {credential}."))
            continue
        if endpoint.managed_service in owned_services:
            rows.append(_row(name, None, "not checked before the configured service is launched",
                             "service startup performs its own readiness check", ""))
            continue
        if endpoint.readiness == "none":
            rows.append(_row(name, None, "not checked: health probing is disabled by the profile",
                             "endpoint available to the worker", ""))
            continue
        if endpoint.kind == "riva_grpc":
            rows.append(_row(name, None, "not checked: launcher does not probe gRPC endpoints",
                             "gRPC endpoint available to the worker", ""))
            continue

        parts, error = _http_parts(endpoint, path)
        if error is not None:
            rows.append(error)
            continue
        assert parts is not None
        if endpoint.health_path is not None or endpoint.readiness == "health":
            probe = _probe_health
        elif _is_loopback(parts):
            probe = _probe_reachability
        else:
            rows.append(_row(name, None, "not checked: remote endpoint has no configured health route",
                             "remote endpoint available to the worker", ""))
            continue
        key = astuple(endpoint)[1:]
        result = probes.get(key)
        if result is None:
            result = probe(endpoint, path, parts)
            probes[key] = result
        rows.append({
            **result,
            "name": name,
            "remediation": (
                _remediation(endpoint, path, parts)
                if result["status"] == "failed" else ""
            ),
        })
    return rows
