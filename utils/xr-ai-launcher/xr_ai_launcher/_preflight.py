# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-contract validation and launcher preflight probes."""
from __future__ import annotations

import ctypes.util
import ipaddress
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence
from urllib.parse import SplitResult, urlsplit, urlunsplit

from ._config import read_config_scalar
from ._credentials import load_credentials
from ._models import EndpointProbe, ModelDeployment, Ownership, load_model_deployment
from ._stack import Parallel, Process, _effective_process_env

Tier = Literal["cheap", "expensive"]
"""Preflight probe cost classification exposed in reports."""

CheckStatus = Literal["ok", "warning", "failed", "skipped", "blocked", "deferred"]
"""Stable machine-readable state for one dependency check."""

_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
_PortInspectionState = Literal["clear", "occupied", "uninspectable"]
_PortInspectionErrorKind = Literal[
    "bind_host", "missing", "timeout", "command", "malformed",
]

_TOP_LEVEL_FIELDS = {
    "$schema",
    "os",
    "arch",
    "python",
    "nvidia_driver",
    "cuda",
    "docker",
    "nvidia_container_toolkit",
    "vulkan",
    "nvenc",
    "node",
    "disk_gb_free",
    "ports",
    "env",
    "commands",
}
_VERSION_CLAUSE = re.compile(r"^(>=|<=|==|!=|>|<)?\s*([0-9]+(?:\.[0-9]+)*)$")
_VERSION_TEXT = re.compile(r"[0-9]+(?:\.[0-9]+)+|[0-9]+")
_REPORT_VERSION = 2
_DOCS = "docs/source/guides/troubleshooting.md"
_EPHEMERAL_PORT_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")
_RESERVED_PORTS = Path("/proc/sys/net/ipv4/ip_local_reserved_ports")
_GPU_PROBE_CLEANUP_TIMEOUT = 2.0
_GPU_PROBE_CLEANUP_RETRY_INTERVAL = 0.05
_GPU_PROBE_CLEANUP_COMMAND_TIMEOUT = 0.5
_PORT_INSPECTION_TIMEOUT = 2.0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


_HEALTH_OPENER = urllib.request.build_opener(_NoRedirect())
_LOCAL_HEALTH_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    _NoRedirect(),
)


class ContractError(ValueError):
    """A dependency contract is unreadable or violates its schema."""


class OwnershipProbeMismatch(RuntimeError):
    """A listener is a known managed service with incompatible identity."""

    def __init__(self, detected: str, remediation: str) -> None:
        super().__init__(detected)
        self.detected = detected
        self.remediation = remediation


class _ProbeSignalInterrupt(BaseException):
    def __init__(self, signum: int, frame: object) -> None:
        super().__init__(signum)
        self.signum = signum
        self.frame = frame


@dataclass(frozen=True)
class _PortInspection:
    state: _PortInspectionState
    evidence: str = ""
    remediation: str = ""
    error_kind: _PortInspectionErrorKind | None = None

    @property
    def detected(self) -> str:
        if self.state == "clear":
            return "no visible socket conflict"
        if self.state == "occupied":
            return f"occupied ({self.evidence})"
        return f"could not inspect ({self.evidence})"


@dataclass(frozen=True)
class _VisibleSocketAddress:
    address: _IPAddress
    dual_stack_wildcard: bool = False


@dataclass(frozen=True)
class CheckResult:
    """One dependency check with machine-readable evidence and remediation."""

    name: str
    """Stable identifier unique within one report."""

    ok: bool | None
    """Whether the requirement passed; ``None`` means it could not be evaluated."""

    detected: str
    """Redacted evidence collected by the probe."""

    required: str
    """Expected version, state, or capability."""

    remediation: str
    """Operator action for a failed check."""

    tier: Tier = "cheap"
    """Probe cost classification exposed in machine-readable reports."""

    skipped: bool = False
    """Whether the probe was inapplicable or deferred."""

    status: CheckStatus | None = None
    """Stable result state; inferred from ``ok`` and ``skipped`` when omitted."""

    def __post_init__(self) -> None:
        if self.status is None:
            object.__setattr__(
                self,
                "status",
                "skipped"
                if self.skipped
                else "deferred"
                if self.ok is None
                else "ok"
                if self.ok
                else "failed",
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class PreparationSpace:
    """Known remaining artifact bytes at one effective service cache path."""

    process: str
    """Owned process whose preparation inventory supplied this entry."""

    cache_path: str | Path
    """Effective cache target after applying service configuration and environment."""

    remaining_bytes: int
    """Bytes still needed for selected artifacts; zero means fully prepared."""


@dataclass(frozen=True)
class ResolvedService:
    """A configured service after process and deployment ownership resolution."""

    name: str
    """Process, managed service, or external model role."""

    ownership: Ownership
    """Lifecycle relationship between the launcher and service."""

    port: int | None = None
    """Resolved port used for bind and ownership checks, when applicable."""

    proto: Literal["tcp", "udp"] = "tcp"
    """Transport used by the resolved port."""

    endpoint: str | None = None
    """Configured endpoint with credentials and query parameters removed."""

    health: str | None = None
    """Readiness endpoint with credentials and query parameters removed."""

    bind_host: str = "0.0.0.0"
    """Configured local address whose port availability is checked."""

    verified_running: bool = False
    """Whether an ownership probe confirmed the expected service is ready."""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class PreflightResult:
    """Complete preflight report. Callers decide how to render or exit."""

    contract_path: Path
    """Requested base contract, whether or not the file exists."""

    contract_files: tuple[Path, ...]
    """Base contract and existing overlays applied to it."""

    profiles: tuple[str, ...]
    """Contract overlay names applied in order."""

    checks: tuple[CheckResult, ...]
    """Dependency and endpoint results in evaluation order."""

    services: tuple[ResolvedService, ...]
    """Resolved lifecycle, bind, and endpoint metadata."""

    @property
    def ok(self) -> bool:
        """Whether every applicable requirement passed."""
        return all(check.ok is not False for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-serializable check report."""
        return {
            "version": _REPORT_VERSION,
            "ok": self.ok,
            "contract": str(self.contract_path),
            "contract_files": [str(path) for path in self.contract_files],
            "profiles": list(self.profiles),
            "checks": [check.to_dict() for check in self.checks],
            "services": [service.to_dict() for service in self.services],
        }


def configuration_error(
    contract_path: str | Path,
    error: BaseException | str,
    *,
    required: str = "valid requirements and deployment profiles",
    remediation: str = "Correct the selected configuration and rerun preflight.",
) -> PreflightResult:
    """Build the standard machine-readable report for configuration failure."""

    return PreflightResult(
        contract_path=Path(contract_path).resolve(),
        contract_files=(),
        profiles=(),
        checks=(CheckResult(
            name="configuration",
            ok=False,
            detected=str(error),
            required=required,
            remediation=remediation,
        ),),
        services=(),
    )


def report_preflight(
    result: PreflightResult,
    *,
    json_output: bool = False,
    verbose: bool = False,
) -> None:
    """Render one preflight report and exit with status 1 when it failed."""

    if json_output:
        print(json.dumps(result.to_dict()))
    else:
        for check in result.checks:
            if verbose or check.status in {"warning", "failed", "blocked", "deferred"}:
                status = "FAIL" if check.status == "failed" else check.status
                print(
                    f"[{status}] {check.name}: detected {check.detected}; "
                    f"required {check.required}",
                    file=sys.stderr,
                )
                if check.status == "failed":
                    print(f"  Fix: {check.remediation}", file=sys.stderr)
                elif check.status == "warning" and check.remediation:
                    print(f"  Guidance: {check.remediation}", file=sys.stderr)
    if not result.ok:
        raise SystemExit(1)


def _error(path: Path, field: str, message: str) -> ContractError:
    location = f"{path}:{field}" if field else str(path)
    return ContractError(f"{location}: {message}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load dependency contract {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(path, "", "contract must be a JSON object")
    return value


def _validate_string_or_strings(path: Path, field: str, value: object) -> None:
    values = [value] if isinstance(value, str) else value
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(item, str) or not item for item in values)
    ):
        raise _error(path, field, "must be a non-empty string or array of strings")


def _validate_constraint(path: Path, field: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise _error(path, field, "must be a non-empty version string")
    for clause in value.split(","):
        if not _VERSION_CLAUSE.fullmatch(clause.strip()):
            raise _error(path, field, f"invalid version clause {clause!r}")


def _validate_version_requirement(path: Path, field: str, value: object) -> None:
    if not isinstance(value, dict):
        _validate_constraint(path, field, value)
        return
    unknown = set(value) - {"version", "unless_env_truthy"}
    if unknown:
        raise _error(path, field, f"unknown field(s): {', '.join(sorted(unknown))}")
    if "version" not in value:
        raise _error(path, field, "must define version")
    _validate_constraint(path, f"{field}.version", value["version"])
    unless_env = value.get("unless_env_truthy")
    if unless_env is not None and (
        not isinstance(unless_env, str) or not unless_env
    ):
        raise _error(path, f"{field}.unless_env_truthy", "must be a non-empty string")


def _validate_contract(path: Path, contract: Mapping[str, object]) -> None:
    unknown = sorted(set(contract) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise _error(path, "", f"unknown field(s): {', '.join(unknown)}")

    if "os" in contract:
        _validate_string_or_strings(path, "os", contract["os"])
    if "arch" in contract:
        arch = contract["arch"]
        if isinstance(arch, dict):
            unknown_arch = set(arch) - {
                "allowed", "unless_env_present", "unless_env_truthy", "unless_config",
            }
            if unknown_arch:
                raise _error(path, "arch", f"unknown field(s): {', '.join(sorted(unknown_arch))}")
            _validate_string_or_strings(path, "arch.allowed", arch.get("allowed"))
            for key in ("unless_env_present", "unless_env_truthy"):
                unless_env = arch.get(key)
                if unless_env is not None and (
                    not isinstance(unless_env, str) or not unless_env
                ):
                    raise _error(path, f"arch.{key}", "must be a non-empty string")
            if "unless_env_present" in arch and "unless_env_truthy" in arch:
                raise _error(
                    path,
                    "arch",
                    "must not define both unless_env_present and unless_env_truthy",
                )
            unless_config = arch.get("unless_config")
            if unless_config is not None and (
                not isinstance(unless_config, str) or not unless_config
            ):
                raise _error(path, "arch.unless_config", "must be a non-empty string")
        else:
            _validate_string_or_strings(path, "arch", arch)

    for field in ("python", "nvidia_driver", "docker", "node"):
        if field in contract:
            _validate_version_requirement(path, field, contract[field])
    if "cuda" in contract:
        cuda = contract["cuda"]
        if not isinstance(cuda, bool):
            _validate_constraint(path, "cuda", cuda)
    for field in ("nvidia_container_toolkit", "vulkan", "nvenc"):
        if field in contract and not isinstance(contract[field], bool):
            raise _error(path, field, "must be a boolean")

    disk = contract.get("disk_gb_free")
    if disk is not None:
        field = "disk_gb_free"
        if not isinstance(disk, dict):
            raise _error(path, field, "must be an object")
        unknown_disk = set(disk) - {
            "config_keys", "runtime_gb", "preparation",
        }
        if unknown_disk:
            raise _error(
                path, field, f"unknown field(s): {', '.join(sorted(unknown_disk))}",
            )
        config_keys = disk.get("config_keys")
        if (
            not isinstance(config_keys, list)
            or not config_keys
            or any(not isinstance(key, str) or not key for key in config_keys)
            or len(set(config_keys)) != len(config_keys)
        ):
            raise _error(
                path,
                f"{field}.config_keys",
                "must be a non-empty array of unique strings",
            )
        runtime_gb = disk.get("runtime_gb")
        if (
            isinstance(runtime_gb, bool)
            or not isinstance(runtime_gb, (int, float))
            or runtime_gb < 0
        ):
            raise _error(
                path, f"{field}.runtime_gb", "must be a non-negative number",
            )
        if "preparation" in disk and not isinstance(disk["preparation"], bool):
            raise _error(path, f"{field}.preparation", "must be a boolean")

    ports = contract.get("ports")
    if ports is not None:
        if not isinstance(ports, list):
            raise _error(path, "ports", "must be an array")
        seen_ports: set[tuple[int, str, object]] = set()
        for index, port in enumerate(ports):
            field = f"ports[{index}]"
            if not isinstance(port, dict):
                raise _error(path, field, "must be an object")
            unknown_port = set(port) - {
                "name", "port", "proto", "health", "config_key",
                "enabled_config_key", "unless_env_truthy", "bind_host", "bind_config_key",
            }
            if unknown_port:
                raise _error(path, field, f"unknown field(s): {', '.join(sorted(unknown_port))}")
            number = port.get("port")
            if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= 65535:
                raise _error(path, f"{field}.port", "must be an integer from 1 to 65535")
            proto = port.get("proto", "tcp")
            if proto not in {"tcp", "udp"}:
                raise _error(path, f"{field}.proto", "must be 'tcp' or 'udp'")
            key = (number, proto, port.get("name"))
            if key in seen_ports:
                raise _error(path, field, f"duplicates {proto} port {number}")
            seen_ports.add(key)
            for key_name in (
                "name", "health", "config_key", "enabled_config_key", "unless_env_truthy",
                "bind_host", "bind_config_key",
            ):
                value = port.get(key_name)
                if value is not None and (not isinstance(value, str) or not value):
                    raise _error(path, f"{field}.{key_name}", "must be a non-empty string")

    env = contract.get("env")
    if env is not None:
        if not isinstance(env, list):
            raise _error(path, "env", "must be an array")
        seen_env: set[str] = set()
        for index, item in enumerate(env):
            field = f"env[{index}]"
            if not isinstance(item, dict):
                raise _error(path, field, "must be an object")
            unknown_env = set(item) - {"name", "required", "docs"}
            if unknown_env:
                raise _error(path, field, f"unknown field(s): {', '.join(sorted(unknown_env))}")
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise _error(path, f"{field}.name", "must be a non-empty string")
            if name in seen_env:
                raise _error(path, field, f"duplicates environment variable {name}")
            seen_env.add(name)
            if not isinstance(item.get("required"), bool):
                raise _error(path, f"{field}.required", "must be a boolean")
            docs = item.get("docs")
            if docs is not None and (not isinstance(docs, str) or not docs):
                raise _error(path, f"{field}.docs", "must be a non-empty string")

    commands = contract.get("commands")
    if commands is not None:
        if not isinstance(commands, list):
            raise _error(path, "commands", "must be an array")
        seen_commands: set[str] = set()
        for index, command in enumerate(commands):
            field = f"commands[{index}]"
            if isinstance(command, str):
                name = command
            elif isinstance(command, dict):
                unknown_command = set(command) - {"name", "unless_env_truthy"}
                if unknown_command:
                    raise _error(
                        path, field,
                        f"unknown field(s): {', '.join(sorted(unknown_command))}",
                    )
                name = command.get("name")
                unless_env = command.get("unless_env_truthy")
                if unless_env is not None and (
                    not isinstance(unless_env, str) or not unless_env
                ):
                    raise _error(
                        path, f"{field}.unless_env_truthy", "must be a non-empty string",
                    )
            else:
                name = None
            if not isinstance(name, str) or not name:
                raise _error(path, field, "must name a non-empty command")
            if name in seen_commands:
                raise _error(path, field, f"duplicates command {name!r}")
            seen_commands.add(name)


def _merge_contract(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_contract(current, value)
        else:
            merged[key] = value
    return merged


def _profile_names(profile: str | Sequence[str] | None) -> tuple[str, ...]:
    if profile is None:
        return ()
    values = (profile,) if isinstance(profile, str) else tuple(profile)
    if any(not isinstance(value, str) or not value for value in values):
        raise ContractError("preflight profile names must be non-empty strings")
    return tuple(dict.fromkeys(values))


def load_contract(
    contract_path: str | Path,
    *,
    profile: str | Sequence[str] | None = None,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Load, overlay, and validate a dependency contract."""

    path = Path(contract_path).resolve()
    if not path.is_file():
        return {}, ()
    contract = _read_json_object(path)
    _validate_contract(path, contract)
    files = [path]
    for name in _profile_names(profile):
        overlay_path = path.with_name(f"{path.stem}.{name}{path.suffix}")
        if not overlay_path.is_file():
            continue
        overlay = _read_json_object(overlay_path)
        _validate_contract(overlay_path, overlay)
        contract = _merge_contract(contract, overlay)
        files.append(overlay_path)
    _validate_contract(path, contract)
    return contract, tuple(files)


def _flatten_processes(processes: Iterable[Process | Parallel]) -> tuple[Process, ...]:
    flat: list[Process] = []
    for item in processes:
        flat.extend(item.processes if isinstance(item, Parallel) else (item,))
    return tuple(flat)


def _discover_deployments(
    processes: Sequence[Process], base: Path,
) -> tuple[ModelDeployment, ...]:
    deployments: dict[Path, ModelDeployment] = {}
    for process in processes:
        config = process.config
        if config is None:
            continue
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = base / config_path
        if not config_path.is_file():
            continue
        text = config_path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"^models_config\s*:", text, re.MULTILINE):
            if not (config_path.parent / "models.json").is_file():
                continue
        deployment = load_model_deployment(config_path)
        deployments.setdefault(deployment.profile_path.resolve(), deployment)
    return tuple(deployments.values())


def _deployment_profiles(
    deployments: Sequence[ModelDeployment],
) -> tuple[str, ...]:
    candidates: list[str] = []
    for deployment in deployments:
        stem = deployment.profile_path.stem
        if stem.startswith("models."):
            candidates.append(stem.removeprefix("models."))
    return tuple(dict.fromkeys(candidates))


def _normalize_os(value: str) -> str:
    aliases = {"win32": "windows", "cygwin": "windows", "darwin": "macos"}
    return aliases.get(value.lower(), value.lower())


def _normalize_arch(value: str) -> str:
    aliases = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}
    return aliases.get(value.lower(), value.lower())


def _requirement_text(value: object, *, minimum: bool = False) -> str:
    text = str(value)
    if minimum and not re.match(r"^[<>=!]", text):
        return f">={text}"
    return text


def _version_requirement(value: object) -> tuple[object, str | None]:
    if isinstance(value, dict):
        return value["version"], value.get("unless_env_truthy")
    return value, None


def _version_tuple(value: str) -> tuple[int, ...] | None:
    match = _VERSION_TEXT.search(value)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(0).split("."))


def _pad_versions(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)), right + (0,) * (width - len(right))


def _version_satisfies(detected: str, constraint: object, *, minimum: bool = False) -> bool:
    detected_version = _version_tuple(detected)
    if detected_version is None:
        return False
    text = _requirement_text(constraint, minimum=minimum)
    for raw_clause in text.split(","):
        match = _VERSION_CLAUSE.fullmatch(raw_clause.strip())
        if match is None:
            return False
        operator = match.group(1) or "=="
        required = tuple(int(part) for part in match.group(2).split("."))
        left, right = _pad_versions(detected_version, required)
        comparisons = {
            "==": left == right,
            "!=": left != right,
            ">=": left >= right,
            "<=": left <= right,
            ">": left > right,
            "<": left < right,
        }
        if not comparisons[operator]:
            return False
    return True


def _run(command: Sequence[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _command_output(command: Sequence[str], *, timeout: float = 10.0) -> tuple[bool, str]:
    try:
        completed = _run(command, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        error = (completed.stderr or "").strip()
        output = (completed.stdout or "").strip()
        return False, error or output or f"exit status {completed.returncode}"
    return True, (completed.stdout or completed.stderr).strip()


def _version_check(
    name: str,
    detected: str | None,
    required: object,
    remediation: str,
    *,
    minimum: bool = False,
    failure_detail: str | None = None,
) -> CheckResult:
    requirement = _requirement_text(required, minimum=minimum)
    shown = detected or failure_detail or "not found"
    return CheckResult(
        name=name,
        ok=detected is not None and _version_satisfies(detected, requirement),
        detected=shown,
        required=requirement,
        remediation=remediation,
    )


def _driver_version() -> tuple[str | None, str | None]:
    ok, output = _command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    )
    if not ok:
        return None, output
    versions = {line.strip() for line in output.splitlines() if line.strip()}
    if not versions:
        return None, "nvidia-smi returned no driver version"
    return min(versions, key=lambda item: _version_tuple(item) or ()), None


def _cuda_version() -> str | None:
    ok, output = _command_output(["nvidia-smi"])
    if not ok:
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else None


def _docker_version() -> tuple[str | None, str | None]:
    ok, output = _command_output(["docker", "version", "--format", "{{.Server.Version}}"])
    if not ok:
        return None, output
    if not output.strip():
        return None, "docker returned no server version"
    return output.splitlines()[0].strip(), None


def _node_version() -> str | None:
    ok, output = _command_output(["node", "--version"])
    return output.lstrip("v").strip() if ok and output.strip() else None


def _redact_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        parsed_port = parsed.port
    except ValueError:
        return "configured endpoint"
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed_port is not None:
        host = f"{host}:{parsed_port}"
    return urlunsplit(SplitResult(parsed.scheme, host, parsed.path, "", ""))


def _probe_endpoint(probe: EndpointProbe) -> CheckResult:
    check_name = probe.role or probe.name
    required = "HTTP 2xx readiness response"
    safe_url = _redact_url(probe.health_url) if probe.health_url else "configured endpoint"
    try:
        parsed = urlsplit(probe.health_url or "")
        parsed_port = parsed.port
    except ValueError:
        parsed = urlsplit("")
        parsed_port = None
    local_endpoint = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    remediation = (
        f"Start the shared model service {check_name!r} that provides {safe_url} "
        "with the selected deployment profile, or correct its endpoint. "
        f"See {_DOCS}#preflight-reused-service."
        if probe.ownership == "reuse" or local_endpoint
        else f"Fix the hosted endpoint or network. See {_DOCS}#preflight-external-service."
    )
    if probe.readiness == "none":
        return CheckResult(
            name=f"endpoint:{check_name}",
            ok=True,
            detected="readiness disabled",
            required="readiness: none",
            remediation="",
            skipped=True,
        )
    if not probe.health_url:
        return CheckResult(
            name=f"endpoint:{check_name}",
            ok=False,
            detected="no health URL",
            required=required,
            remediation=remediation,
        )
    if parsed.scheme == "tcp" and parsed.hostname and parsed_port:
        try:
            with socket.create_connection(
                (parsed.hostname, parsed_port), timeout=probe.timeout
            ):
                pass
            return CheckResult(
                name=f"endpoint:{check_name}",
                ok=True,
                detected=f"{safe_url} accepted a TCP connection",
                required="configured endpoint reachable",
                remediation=remediation,
            )
        except OSError as exc:
            return CheckResult(
                name=f"endpoint:{check_name}",
                ok=False,
                detected=f"{safe_url} unavailable ({type(exc).__name__})",
                required="configured endpoint reachable",
                remediation=remediation,
            )
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return CheckResult(
            name=f"endpoint:{check_name}",
            ok=False,
            detected=f"{safe_url} is not a usable health endpoint",
            required="an http(s) URL or tcp://host:port",
            remediation=remediation,
        )
    headers: dict[str, str] = {}
    if probe.api_key_env and os.environ.get(probe.api_key_env):
        headers["Authorization"] = f"Bearer {os.environ[probe.api_key_env]}"
    request = urllib.request.Request(probe.health_url, headers=headers)
    try:
        opener = (
            _LOCAL_HEALTH_OPENER
            if parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            else _HEALTH_OPENER
        )
        with opener.open(request, timeout=probe.timeout) as response:
            status = response.status
        ok = 200 <= status < 300
        detected = f"{safe_url} returned HTTP {status}"
    except urllib.error.HTTPError as exc:
        ok = False
        detected = f"{safe_url} returned HTTP {exc.code}"
    except Exception as exc:
        ok = False
        detected = f"{safe_url} unavailable ({type(exc).__name__})"
    return CheckResult(
        name=f"endpoint:{check_name}",
        ok=ok,
        detected=detected,
        required=required,
        remediation=remediation,
    )


def _port_is_free(
    port: int,
    proto: str,
    host: str = "0.0.0.0",
) -> _PortInspection:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    targets, error = _resolve_bind_addresses(host, family)
    if error is not None:
        return error

    families = (socket.AF_INET6,) if family == socket.AF_INET6 else (
        socket.AF_INET,
        socket.AF_INET6,
    )
    visible: list[tuple[int, _VisibleSocketAddress]] = []
    for visible_family in families:
        addresses, error = _visible_socket_addresses(port, proto, visible_family)
        if error is not None:
            return error
        visible.extend((visible_family, address) for address in addresses)

    for visible_family, socket_address in visible:
        if _socket_conflicts(targets, family, socket_address, visible_family):
            address = socket_address.address
            if socket_address.dual_stack_wildcard:
                rendered = "*"
            else:
                rendered = f"[{address}]" if address.version == 6 else str(address)
            return _PortInspection(
                state="occupied",
                evidence=f"visible {proto} socket on {rendered}:{port}",
                remediation=(
                    "Stop the process using this port or configure a different port. "
                    f"See {_DOCS}#preflight-port."
                ),
            )
    return _PortInspection(state="clear")


def _resolve_bind_addresses(
    host: str,
    family: int,
) -> tuple[set[_IPAddress], _PortInspection | None]:
    normalized = host.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    literal = normalized.rsplit("%", 1)[0]
    try:
        address = ipaddress.ip_address(literal)
    except ValueError:
        try:
            infos = socket.getaddrinfo(normalized, None, family)
        except socket.gaierror as exc:
            description = exc.strerror or type(exc).__name__
            return set(), _PortInspection(
                state="uninspectable",
                error_kind="bind_host",
                evidence=f"could not resolve bind host {host!r}: {description}",
                remediation=(
                    f"Correct bind host {host!r} in the service configuration, then "
                    f"rerun preflight. See {_DOCS}#preflight-port."
                ),
            )
        addresses = {
            ipaddress.ip_address(info[4][0].rsplit("%", 1)[0])
            for info in infos
        }
    else:
        addresses = {address}
    return addresses, None


def _visible_socket_addresses(
    port: int,
    proto: str,
    family: int,
) -> tuple[set[_VisibleSocketAddress], _PortInspection | None]:
    command = [
        "ss",
        "-H",
        "-n",
        "-6" if family == socket.AF_INET6 else "-4",
        "-l" if proto == "tcp" else "-a",
        "-t" if proto == "tcp" else "-u",
        f"sport = :{port}",
    ]
    diagnostic = shlex.join(command)
    command_remediation = (
        f"Run `{diagnostic}` in the same login session and correct the reported "
        f"error. See {_DOCS}#preflight-port."
    )
    try:
        completed = _run(command, timeout=_PORT_INSPECTION_TIMEOUT)
    except FileNotFoundError:
        return set(), _PortInspection(
            state="uninspectable",
            error_kind="missing",
            evidence="ss not found",
            remediation=(
                "Install the iproute2 package so `ss` is available on PATH, then "
                f"rerun preflight. See {_DOCS}#preflight-port."
            ),
        )
    except subprocess.TimeoutExpired:
        return set(), _PortInspection(
            state="uninspectable",
            error_kind="timeout",
            evidence=f"ss timed out after {_PORT_INSPECTION_TIMEOUT:g} seconds",
            remediation=(
                f"Run `{diagnostic}` directly to diagnose the timeout, then rerun "
                f"preflight. See {_DOCS}#preflight-port."
            ),
        )
    except OSError as exc:
        description = exc.strerror or type(exc).__name__
        return set(), _PortInspection(
            state="uninspectable",
            error_kind="command",
            evidence=f"ss failed: {description}",
            remediation=command_remediation,
        )
    detail = _bounded_inspection_text(completed.stderr)
    if completed.returncode != 0:
        suffix = f": {detail}" if detail else ""
        return set(), _PortInspection(
            state="uninspectable",
            error_kind="command",
            evidence=f"ss exited with status {completed.returncode}{suffix}",
            remediation=command_remediation,
        )
    if detail:
        return set(), _PortInspection(
            state="uninspectable",
            error_kind="command",
            evidence=f"ss reported: {detail}",
            remediation=command_remediation,
        )

    addresses: set[_VisibleSocketAddress] = set()
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if (
            len(fields) != 5
            or not fields[1].isdigit()
            or not fields[2].isdigit()
            or (proto == "tcp" and fields[0] != "LISTEN")
            or (proto == "udp" and fields[0] not in {"UNCONN", "ESTAB"})
        ):
            return set(), _malformed_socket_inspection(line, diagnostic)
        address = _parse_ss_local_address(fields[3], port, family)
        if address is None:
            return set(), _malformed_socket_inspection(line, diagnostic)
        addresses.add(address)
    return addresses, None


def _bounded_inspection_text(text: str) -> str:
    detail = " ".join(text.split())
    return f"{detail[:197]}..." if len(detail) > 200 else detail


def _malformed_socket_inspection(row: str, diagnostic: str) -> _PortInspection:
    return _PortInspection(
        state="uninspectable",
        error_kind="malformed",
        evidence=f"malformed ss output: {_bounded_inspection_text(row)!r}",
        remediation=(
            f"Report the parsing gap with output from `ss -V` and `{diagnostic}`. "
            f"See {_DOCS}#preflight-port."
        ),
    )


def _parse_ss_local_address(
    endpoint: str,
    port: int,
    family: int,
) -> _VisibleSocketAddress | None:
    bracketed = endpoint.startswith("[")
    if bracketed:
        closing = endpoint.find("]")
        if closing < 0:
            return None
        host = endpoint[1:closing]
        scope, delimiter, port_text = endpoint[closing + 1:].rpartition(":")
        if not delimiter or (scope and not scope.startswith("%")):
            return None
    else:
        host, delimiter, port_text = endpoint.rpartition(":")
        if not delimiter:
            return None
    if not port_text.isdigit() or int(port_text) != port:
        return None
    host = host.rsplit("%", 1)[0]
    wildcard = host == "*"
    if wildcard:
        host = "::" if family == socket.AF_INET6 else "0.0.0.0"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    expected_version = 6 if family == socket.AF_INET6 else 4
    if address.version != expected_version:
        return None
    # ss renders a dual-stack wildcard as * and an IPv6-only wildcard as [::].
    return _VisibleSocketAddress(
        address=address,
        dual_stack_wildcard=(
            family == socket.AF_INET6
            and address.is_unspecified
            and (wildcard or not bracketed)
        ),
    )


def _socket_conflicts(
    targets: set[_IPAddress],
    target_family: int,
    visible: _VisibleSocketAddress,
    visible_family: int,
) -> bool:
    address = visible.address
    if target_family == visible_family:
        return any(
            target.is_unspecified or address.is_unspecified or target == address
            for target in targets
        )
    if target_family != socket.AF_INET or visible_family != socket.AF_INET6:
        return False
    if visible.dual_stack_wildcard:
        return True
    mapped = address.ipv4_mapped
    return mapped is not None and any(
        target.is_unspecified or target == mapped
        for target in targets
    )


def _bind_hosts_overlap(left: str, right: str) -> bool:
    left = left.strip().lower().strip("[]")
    right = right.strip().lower().strip("[]")
    if left == right:
        return True
    left_is_ipv6 = ":" in left
    right_is_ipv6 = ":" in right
    if left_is_ipv6 != right_is_ipv6:
        return False
    wildcard = "::" if left_is_ipv6 else "0.0.0.0"
    return left == wildcard or right == wildcard


def _owned_bind_conflict_checks(
    services: Sequence[ResolvedService],
) -> list[CheckResult]:
    owned = [
        (index, service)
        for index, service in enumerate(services)
        if service.ownership == "own" and service.port is not None
    ]
    checks: list[CheckResult] = []
    for position, (left_index, left) in enumerate(owned):
        for right_index, right in owned[position + 1:]:
            if (
                left.proto != right.proto
                or left.port != right.port
                or not _bind_hosts_overlap(left.bind_host, right.bind_host)
            ):
                continue
            checks.append(CheckResult(
                name=(
                    f"port_conflict:{left.proto}:{left.port}:"
                    f"{left_index}:{right_index}"
                ),
                ok=False,
                detected=(
                    f"{left.name} binds {left.bind_host}:{left.port} and "
                    f"{right.name} binds {right.bind_host}:{right.port}"
                ),
                required="unique owned service bind addresses",
                remediation=(
                    f"Configure {left.name} and {right.name} to use different "
                    f"{left.proto} ports or non-overlapping bind addresses."
                ),
            ))
    return checks


def _read_ephemeral_port_policy(
    port_range_path: Path,
    reserved_ports_path: Path,
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    range_fields = port_range_path.read_text(encoding="utf-8").split()
    if len(range_fields) != 2:
        raise ValueError(f"{port_range_path} must contain two integers")
    try:
        first, last = (int(value) for value in range_fields)
    except ValueError as exc:
        raise ValueError(f"{port_range_path} must contain two integers") from exc
    if not 1 <= first <= last <= 65535:
        raise ValueError(f"{port_range_path} contains an invalid port range")

    reservations: list[tuple[int, int]] = []
    reserved_text = reserved_ports_path.read_text(encoding="utf-8").strip()
    if reserved_text:
        for entry in reserved_text.split(","):
            fields = entry.strip().split("-")
            if len(fields) not in {1, 2}:
                raise ValueError(
                    f"{reserved_ports_path} contains invalid entry {entry!r}"
                )
            try:
                start = int(fields[0])
                end = int(fields[-1])
            except ValueError as exc:
                raise ValueError(
                    f"{reserved_ports_path} contains invalid entry {entry!r}"
                ) from exc
            if not 1 <= start <= end <= 65535:
                raise ValueError(
                    f"{reserved_ports_path} contains invalid entry {entry!r}"
                )
            reservations.append((start, end))
    return (first, last), tuple(reservations)


def _ephemeral_port_checks(
    services: Sequence[ResolvedService],
    *,
    port_range_path: Path = _EPHEMERAL_PORT_RANGE,
    reserved_ports_path: Path = _RESERVED_PORTS,
) -> list[CheckResult]:
    owned = [
        service
        for service in services
        if service.ownership == "own" and service.port is not None
    ]
    if not owned:
        return []
    try:
        port_range, reservations = _read_ephemeral_port_policy(
            port_range_path, reserved_ports_path,
        )
    except (OSError, ValueError) as exc:
        return [CheckResult(
            name="ephemeral_ports",
            ok=True,
            detected=f"could not inspect host ephemeral port policy ({exc})",
            required="host ephemeral port policy available for owned ports",
            remediation=(
                f"Read `{port_range_path}` and `{reserved_ports_path}`, or run "
                "`sysctl net.ipv4.ip_local_port_range "
                "net.ipv4.ip_local_reserved_ports`, then correct the access or "
                f"value and rerun preflight. Refer to {_DOCS}#preflight-port."
            ),
            status="warning",
        )]

    first, last = port_range
    checks: list[CheckResult] = []
    for service in owned:
        port = service.port
        if not first <= port <= last or any(
            start <= port <= end for start, end in reservations
        ):
            continue
        checks.append(CheckResult(
            name=f"ephemeral_port:{service.name}:{service.proto}:{port}",
            ok=True,
            detected=(
                f"{service.proto} port {port} for {service.name} is inside the host "
                f"ephemeral range {first}-{last} and is not reserved"
            ),
            required="owned service port excluded from automatic ephemeral allocation",
            remediation=(
                "Preserve every existing `net.ipv4.ip_local_reserved_ports` entry "
                f"and add port {port}; writing this setting replaces the complete "
                "list. The reservation prevents future automatic allocation but "
                "does not release an existing connection. "
                f"Refer to {_DOCS}#preflight-port."
            ),
            status="warning",
        ))
    return checks


def _resolve_services(
    contract: Mapping[str, Any],
    processes: Sequence[Process],
    probes: Sequence[EndpointProbe],
    base: Path,
    *,
    suppress_unprofiled_reuse: bool,
) -> tuple[ResolvedService, ...]:
    process_by_name = {
        process.name: process
        for process in processes
        if process.name
    }
    probe_by_name = {probe.name: probe for probe in probes}

    services: list[ResolvedService] = []
    represented: set[tuple[str, Ownership]] = set()
    represented_processes: set[str] = set()
    has_resolution_context = bool(processes)
    for spec in contract.get("ports", []):
        declared_port = int(spec["port"])
        proto = spec.get("proto", "tcp")
        unless_env = spec.get("unless_env_truthy")
        if unless_env and _env_truthy(unless_env):
            continue
        declared_name = spec.get("name")
        process = process_by_name.get(declared_name)
        probe = probe_by_name.get(declared_name)
        if declared_name and has_resolution_context and process is None and probe is None:
            raise ContractError(
                f"port requirement {declared_name!r} has no matching process or endpoint"
            )
        if process is not None and not _configured_port_enabled(process, spec, base):
            continue
        name = str(
            spec.get("name")
            or (process.name if process is not None else f"port-{declared_port}")
        )
        port = _configured_port(process, spec, base) if process is not None else declared_port
        bind_host = (
            _configured_bind_host(process, spec, base)
            if process is not None else str(spec.get("bind_host", "0.0.0.0"))
        )
        if probe is not None:
            ownership: Ownership = probe.ownership
        else:
            launch_mode = process.launch_mode if process is not None else "own"
            ownership = "reuse" if launch_mode == "reuse" else "own"
        health = probe.health_url if probe is not None else None
        if health is None and spec.get("health"):
            health = f"http://127.0.0.1:{port}{spec['health']}"
        services.append(
            ResolvedService(
                name=name,
                ownership=ownership,
                port=port,
                proto=proto,
                bind_host=bind_host,
                endpoint=(
                    _redact_url(probe.endpoint_url)
                    if probe and probe.endpoint_url else None
                ),
                health=_redact_url(health) if health else None,
            )
        )
        represented.add((name, ownership))
        if process is not None:
            represented_processes.add(name)

    for process in processes:
        name = process.name
        if not name or name in represented_processes:
            continue
        launch_mode = process.launch_mode
        if launch_mode == "reuse" and suppress_unprofiled_reuse and name not in probe_by_name:
            continue
        port = process.port
        if not isinstance(port, int):
            continue
        ownership: Ownership = "reuse" if launch_mode == "reuse" else "own"
        probe = probe_by_name.get(name)
        services.append(ResolvedService(
            name=name,
            ownership=probe.ownership if probe else ownership,
            port=port,
            endpoint=(
                _redact_url(probe.endpoint_url)
                if probe and probe.endpoint_url else None
            ),
            health=_redact_url(probe.health_url) if probe and probe.health_url else None,
        ))
        represented.add((name, probe.ownership if probe else ownership))
        represented_processes.add(name)

    for probe in probes:
        key = (probe.name, probe.ownership)
        if key in represented:
            continue
        port: int | None = None
        if probe.health_url:
            try:
                parsed = urlsplit(probe.health_url)
                port = parsed.port or ({"http": 80, "https": 443}.get(parsed.scheme))
            except ValueError:
                pass
        services.append(
            ResolvedService(
                name=probe.name,
                ownership=probe.ownership,
                port=port,
                endpoint=_redact_url(probe.endpoint_url) if probe.endpoint_url else None,
                health=_redact_url(probe.health_url) if probe.health_url else None,
            )
        )
        represented.add(key)
    return tuple(services)


def _configured_port(process: Process, spec: Mapping[str, Any], base: Path) -> int:
    process_port = process.port
    if isinstance(process_port, int):
        return process_port
    declared = int(spec["port"])
    config = process.config
    if config is None:
        return declared
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = base / config_path
    key = spec.get("config_key")
    if key:
        raw_port = read_config_scalar(config_path, key)
        if raw_port:
            try:
                resolved = int(raw_port)
            except ValueError as exc:
                raise ContractError(
                    f"{config_path}: {key} must be an integer"
                ) from exc
            if not 1 <= resolved <= 65535:
                raise ContractError(f"{config_path}: {key} must be from 1 to 65535")
            return resolved
    endpoint = read_config_scalar(config_path, "endpoint")
    if endpoint:
        try:
            endpoint_port = urlsplit(endpoint).port
        except ValueError as exc:
            raise ContractError(f"{config_path}: endpoint has an invalid port") from exc
        if endpoint_port is not None:
            return endpoint_port
    return declared


def _configured_bind_host(
    process: Process,
    spec: Mapping[str, Any],
    base: Path,
) -> str:
    declared = str(spec.get("bind_host", "0.0.0.0"))
    key = spec.get("bind_config_key")
    config = process.config
    if key is None or config is None:
        return declared
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = base / config_path
    return read_config_scalar(config_path, key, declared)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _configured_port_enabled(
    process: Process,
    spec: Mapping[str, Any],
    base: Path,
) -> bool:
    enable_key = spec.get("enabled_config_key")
    if enable_key is None:
        return True
    config = process.config
    if config is None:
        return False
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = base / config_path
    return (read_config_scalar(config_path, enable_key) or "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _configured_value(processes: Sequence[Process], base: Path, key: str) -> str:
    for process in processes:
        config = process.config
        if config is None:
            continue
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = base / config_path
        value = read_config_scalar(config_path, key)
        if value:
            return value
    return ""


def _process_config_path(process: Process, base: Path) -> Path | None:
    if process.config is None:
        return None
    path = Path(process.config)
    return path if path.is_absolute() else base / path


def _configured_cache_path(
    process: Process,
    base: Path,
    config_keys: Sequence[str],
) -> Path:
    config_path = _process_config_path(process, base)
    if config_path is None:
        raise ContractError(
            f"process {process.name!r} needs a config with one of "
            f"{', '.join(config_keys)} for its disk requirement"
        )
    for key in config_keys:
        raw = read_config_scalar(config_path, key)
        if not raw:
            continue
        expanded = Path(os.path.expandvars(raw)).expanduser()
        return (expanded if expanded.is_absolute() else config_path.parent / expanded).resolve()
    raise ContractError(
        f"{config_path}: process {process.name!r} must configure one of "
        f"{', '.join(config_keys)} for its disk requirement"
    )


def _effective_cache_path(cache_path: str | Path, base: Path) -> Path:
    expanded = Path(os.path.expandvars(str(cache_path))).expanduser()
    return (expanded if expanded.is_absolute() else base / expanded).resolve()


def _filesystem_usage(path: Path) -> tuple[int, int]:
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    stat = existing.stat()
    return stat.st_dev, shutil.disk_usage(existing).free


def _disk_checks(
    policy: Mapping[str, Any] | None,
    processes: Sequence[Process],
    services: Sequence[ResolvedService],
    base: Path,
    preparation_space: Sequence[PreparationSpace],
) -> list[CheckResult]:
    if policy is None:
        if preparation_space:
            raise ContractError("preparation-space inventory requires a disk policy")
        return []
    if not processes:
        if preparation_space:
            raise ContractError("preparation-space inventory requires local processes")
        return [CheckResult(
            name="disk:resolution:unknown",
            ok=None,
            detected="no process configuration was supplied to resolve cache paths",
            required="effective cache paths and runtime headroom",
            remediation=(
                "Pass the launcher processes and base directory, then rerun preflight."
            ),
            status="deferred",
            skipped=True,
        )]
    if preparation_space and not policy.get("preparation"):
        raise ContractError(
            "preparation-space inventory requires a preparation-enabled disk policy"
        )

    process_by_name = {process.name: process for process in processes}
    ownership = {service.name: service.ownership for service in services}

    def is_owned(process: Process) -> bool:
        return ownership.get(
            process.name,
            "reuse" if process.launch_mode == "reuse" else "own",
        ) == "own"

    inventory: dict[str, list[tuple[Path, int]]] = {}
    seen_inventory: set[tuple[str, Path]] = set()
    for item in preparation_space:
        if not isinstance(item, PreparationSpace):
            raise ContractError("preparation_space entries must be PreparationSpace values")
        if not isinstance(item.process, str) or not item.process:
            raise ContractError("preparation-space process names must be non-empty strings")
        process = process_by_name.get(item.process)
        if process is None or not is_owned(process):
            raise ContractError(
                f"preparation-space process {item.process!r} is not launcher-owned"
            )
        if (
            isinstance(item.remaining_bytes, bool)
            or not isinstance(item.remaining_bytes, int)
            or item.remaining_bytes < 0
        ):
            raise ContractError(
                f"preparation-space bytes for {item.process!r} must be a non-negative integer"
            )
        if (
            not isinstance(item.cache_path, (str, Path))
            or not str(item.cache_path)
        ):
            raise ContractError(
                f"preparation-space cache path for {item.process!r} must not be empty"
            )
        cache_path = _effective_cache_path(item.cache_path, base)
        key = (item.process, cache_path)
        if key in seen_inventory:
            raise ContractError(
                f"duplicate preparation-space path for {item.process!r}: {cache_path}"
            )
        seen_inventory.add(key)
        inventory.setdefault(item.process, []).append((cache_path, item.remaining_bytes))

    runtime_by_device: dict[int, dict[str, Any]] = {}
    preparation_by_device: dict[int, int] = {}
    deferred: list[CheckResult] = []

    config_keys = tuple(policy["config_keys"])
    runtime_gb = float(policy["runtime_gb"])
    for process in processes:
        if not is_owned(process):
            continue
        supplied = inventory.get(process.name)
        if supplied is not None:
            paths = supplied
        else:
            configured = _configured_cache_path(process, base, config_keys)
            paths = [(configured, 0)]
            if policy.get("preparation"):
                deferred.append(CheckResult(
                    name=f"disk:preparation:{process.name}:unknown",
                    ok=None,
                    detected=(
                        f"remaining preparation size is unknown for {configured}"
                    ),
                    required="caller-supplied selected-artifact preparation inventory",
                    remediation=(
                        f"Collect the effective cache paths and remaining bytes for "
                        f"{process.name}, then rerun preflight."
                    ),
                    status="deferred",
                    skipped=True,
                ))

        for path_index, (cache_path, remaining_bytes) in enumerate(paths):
            try:
                device, free_bytes = _filesystem_usage(cache_path)
            except OSError as exc:
                deferred.append(CheckResult(
                    name=f"disk:runtime:{process.name}:{path_index}:unavailable",
                    ok=False,
                    detected=f"{cache_path} unavailable ({type(exc).__name__}: {exc})",
                    required=f">={runtime_gb:g} GB free runtime headroom",
                    remediation=(
                        f"Make the configured cache path accessible or update one of "
                        f"{', '.join(config_keys)}."
                    ),
                ))
                continue
            entry = runtime_by_device.setdefault(device, {
                "free": free_bytes,
                "paths": set(),
            })
            entry["free"] = min(entry["free"], free_bytes)
            entry["paths"].add(cache_path)
            if supplied is not None and policy.get("preparation"):
                preparation_by_device[device] = (
                    preparation_by_device.get(device, 0) + remaining_bytes
                )

    checks: list[CheckResult] = []
    for device, entry in sorted(runtime_by_device.items()):
        free_bytes = int(entry["free"])
        runtime_bytes = int(runtime_gb * 1_000_000_000)
        paths = ", ".join(str(path) for path in sorted(entry["paths"]))
        keys = ", ".join(config_keys)
        checks.append(CheckResult(
            name=f"disk:runtime:{device}",
            ok=free_bytes >= runtime_bytes,
            detected=f"{free_bytes / 1_000_000_000:.1f} GB free for {paths}",
            required=f">={runtime_gb:g} GB free runtime headroom",
            remediation=(
                f"Free space or move the cache by updating one of {keys}. "
                f"See {_DOCS}#preflight-disk."
            ),
        ))
        if device in preparation_by_device:
            remaining = preparation_by_device[device]
            required = remaining + runtime_bytes
            checks.append(CheckResult(
                name=f"disk:preparation:{device}",
                ok=free_bytes >= required,
                detected=(
                    f"{free_bytes / 1_000_000_000:.1f} GB free for {paths}; "
                    f"{remaining / 1_000_000_000:.1f} GB of selected artifacts remain"
                ),
                required=(
                    f">={required / 1_000_000_000:.1f} GB free for remaining artifacts "
                    "and runtime headroom"
                ),
                remediation=(
                    f"Free enough space for the selected remaining artifacts and runtime "
                    f"headroom, or move the cache by updating one of {keys}. "
                    f"See {_DOCS}#preflight-disk."
                ),
            ))
    return [*checks, *deferred]


def _local_services_require_docker(processes: Sequence[Process]) -> bool:
    return any(
        process.launch_mode != "reuse"
        and process.needs_docker is True
        for process in processes
    )


def _cleanup_gpu_probe_container(container_name: str) -> None:
    deadline = time.monotonic() + _GPU_PROBE_CLEANUP_TIMEOUT
    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK,
        {signal.SIGINT, signal.SIGTERM},
    )
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                completed = subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=min(_GPU_PROBE_CLEANUP_COMMAND_TIMEOUT, remaining),
                )
            except (OSError, subprocess.TimeoutExpired):
                completed = None
            if completed is not None and completed.returncode == 0:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(_GPU_PROBE_CLEANUP_RETRY_INTERVAL, remaining))
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _run_gpu_probe(command: Sequence[str], container_name: str) -> tuple[bool, str]:
    cleanup_late_create = False
    probe_started = False
    interrupted: _ProbeSignalInterrupt | None = None
    previous_handlers: dict[int, Any] = {}
    managed_signals = threading.current_thread() is threading.main_thread()
    watched_signals = {signal.SIGINT, signal.SIGTERM}
    previous_mask: set[signal.Signals] | None = None
    signals_blocked = False

    def defer_signal(signum: int, frame: object) -> None:
        raise _ProbeSignalInterrupt(signum, frame)

    if managed_signals:
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched_signals)
        signals_blocked = True

    try:
        if managed_signals:
            for signum in watched_signals:
                previous = signal.getsignal(signum)
                if previous != signal.SIG_IGN:
                    previous_handlers[signum] = previous
                    signal.signal(signum, defer_signal)
        try:
            if managed_signals:
                signals_blocked = False
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            probe_started = True
            try:
                completed = _run(command, timeout=45.0)
            finally:
                if managed_signals:
                    signal.pthread_sigmask(signal.SIG_BLOCK, watched_signals)
                    signals_blocked = True
        except subprocess.TimeoutExpired as exc:
            cleanup_late_create = True
            result = False, f"{type(exc).__name__}: {exc}"
        except _ProbeSignalInterrupt as exc:
            cleanup_late_create = probe_started
            interrupted = exc
            if managed_signals and not signals_blocked:
                signal.pthread_sigmask(signal.SIG_BLOCK, watched_signals)
                signals_blocked = True
            result = False, f"interrupted by signal {exc.signum}"
        except OSError as exc:
            result = False, f"{type(exc).__name__}: {exc}"
        else:
            output = (completed.stdout or completed.stderr).strip()
            result = (
                (False, output or f"exit status {completed.returncode}")
                if completed.returncode != 0
                else (True, output)
            )
    except BaseException:
        cleanup_late_create = probe_started
        raise
    finally:
        try:
            if cleanup_late_create:
                _cleanup_gpu_probe_container(container_name)
        finally:
            for signum, previous in reversed(previous_handlers.items()):
                signal.signal(signum, previous)
            if managed_signals and signals_blocked:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    if interrupted is not None:
        previous = previous_handlers[interrupted.signum]
        if previous == signal.SIG_DFL:
            os.kill(os.getpid(), interrupted.signum)
        elif callable(previous):
            previous(interrupted.signum, interrupted.frame)
    return result


def _container_gpu_check(*, strict: bool) -> CheckResult:
    ok, images_output = _command_output(
        ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
    )
    images = [
        image for image in images_output.splitlines()
        if ok and image and "<none>" not in image
    ]
    preferred = next((
        image for image in images
        if image.startswith((
            "nvcr.io/nim/", "nvcr.io/nvidia/", "nvidia/cuda:", "ubuntu:",
        ))
    ), None)
    if preferred is None:
        return CheckResult(
            name="container_gpu",
            ok=None if not strict else False,
            detected=(
                "not run: no local container image; deferred until after prepare"
                if not strict else "not run: no local container image"
            ),
            required="GPU device visible inside a container",
            remediation=(
                "Prepare the service artifacts, then rerun preflight. "
                f"See {_DOCS}#preflight-container-toolkit."
            ),
            tier="expensive",
            skipped=not strict,
            status="deferred" if not strict else "failed",
        )
    container_name = f"xr-ai-preflight-gpu-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    command = [
        "docker", "run", "--rm", "--name", container_name, "--pull=never",
        "--runtime", "nvidia",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "--entrypoint", "nvidia-smi",
        preferred, "-L",
    ]
    success, output = _run_gpu_probe(command, container_name)
    return CheckResult(
        name="container_gpu",
        ok=success,
        detected=f"GPU visible using {preferred}" if success else (output or "GPU not visible"),
        required="GPU device visible inside a container",
        remediation=f"See {_DOCS}#preflight-container-toolkit.",
        tier="expensive",
    )


def _vulkan_device_check() -> CheckResult:
    executable = shutil.which("vulkaninfo")
    if executable is None:
        return CheckResult(
            name="vulkan_device",
            ok=False,
            detected="not run: vulkaninfo is not installed",
            required="usable Vulkan device",
            remediation="Install vulkan-tools and rerun preflight.",
            tier="expensive",
        )
    success, output = _command_output([executable, "--summary"], timeout=30.0)
    summary = output.lower()
    hardware_device = (
        "vendorid" in summary
        and (
            "0x10de" in summary
            or "physical_device_type_discrete_gpu" in summary
            or "physical_device_type_integrated_gpu" in summary
        )
    )
    success = success and hardware_device
    return CheckResult(
        name="vulkan_device",
        ok=success,
        detected=(
            "hardware Vulkan device found"
            if success
            else "software-only Vulkan device found"
            if "llvmpipe" in summary or "physical_device_type_cpu" in summary
            else (output or "no usable device")
        ),
        required="usable Vulkan device",
        remediation=f"See {_DOCS}#preflight-vulkan.",
        tier="expensive",
    )


def _nvenc_check() -> CheckResult:
    executable = shutil.which("ffmpeg")
    if executable is None:
        return CheckResult(
            name="nvenc",
            ok=False,
            detected="not run: ffmpeg is not installed",
            required="functional NVENC session",
            remediation="Install ffmpeg and rerun preflight for an encode probe.",
            tier="expensive",
        )
    command = [
        executable, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "color=size=256x256:rate=1", "-frames:v", "1", "-c:v",
        "h264_nvenc", "-f", "null", "-",
    ]
    success, output = _command_output(command, timeout=30.0)
    return CheckResult(
        name="nvenc",
        ok=success,
        detected="functional encode completed" if success else (output or "encode failed"),
        required="functional NVENC session",
        remediation=f"See {_DOCS}#preflight-nvenc.",
        tier="expensive",
    )


def _expensive_checks(
    names: Sequence[str],
    *,
    force: bool,
) -> list[CheckResult]:
    runners = {
        "container_gpu": lambda: _container_gpu_check(strict=force),
        "vulkan_device": _vulkan_device_check,
        "nvenc": _nvenc_check,
    }
    return [runners[name]() for name in names]


def rerun_deferred_checks(result: PreflightResult) -> PreflightResult:
    """Rerun deferred post-prepare probes without repeating preflight."""

    deferred = {
        check.name for check in result.checks
        if check.status == "deferred" and check.name == "container_gpu"
    }
    if not deferred:
        return result
    replacements = {
        check.name: check
        for check in _expensive_checks(
            sorted(deferred), force=True,
        )
    }
    return replace(
        result,
        checks=tuple(replacements.get(check.name, check) for check in result.checks),
    )


def preflight(
    contract_path: str | Path,
    *,
    processes: Sequence[Process | Parallel] = (),
    base: str | Path | None = None,
    force_expensive: bool = False,
    profile: str | Sequence[str] | None = None,
    endpoints: Sequence[EndpointProbe] = (),
    deployment: ModelDeployment | None = None,
    required_credentials: Sequence[str] = (),
    preparation_space: Sequence[PreparationSpace] = (),
) -> PreflightResult:
    """Validate a contract and return all applicable checks without printing.

    ``base`` resolves relative process paths. ``force_expensive`` makes an
    expensive probe fail when it cannot run instead of deferring it; it never
    bypasses failed cheap checks. ``preparation_space`` must describe every
    effective cache target for each process it inventories; omitted inventories
    remain explicitly deferred. Credential loading mutates ``os.environ`` before
    process launch environments and ownership probes are evaluated. Invalid
    dependency contracts and resolution inputs raise ``ContractError``; invalid
    model profiles raise ``ValueError``. Process configuration discovery may
    propagate ``OSError`` when a configuration file cannot be read.
    """

    load_credentials()
    path = Path(contract_path).resolve()
    sample_base = Path(base).resolve() if base is not None else path.parent
    flat_processes = _flatten_processes(processes)
    discovered = list(_discover_deployments(flat_processes, sample_base))
    if deployment is not None:
        discovered.append(deployment)
    deployments = tuple(
        {item.profile_path.resolve(): item for item in discovered}.values()
    )
    explicit_profiles = _profile_names(profile)
    deployment_profiles = _deployment_profiles(deployments)
    profiles = tuple(dict.fromkeys((*explicit_profiles, *deployment_profiles)))
    contract, contract_files = load_contract(path, profile=profiles)

    all_probes: dict[tuple[str, Ownership], EndpointProbe] = {}
    for item in (*endpoints, *(probe for item in deployments for probe in item.endpoint_probes)):
        key = (item.role or item.name, item.ownership)
        previous = all_probes.setdefault(key, item)
        if previous != item:
            raise ContractError(
                f"conflicting resolved endpoints for role {item.role or item.name!r}"
            )
    probes = tuple(all_probes.values())
    services = _resolve_services(
        contract,
        flat_processes,
        probes,
        sample_base,
        suppress_unprofiled_reuse=bool(deployments),
    )

    checks: list[CheckResult] = []
    declared_processes = {
        process.name: process
        for process in flat_processes
        if process.name
    }
    for service_name, ownership in {
        name: mode for item in deployments for name, mode in item.services.items()
    }.items():
        process = declared_processes.get(service_name)
        if ownership == "own" and process is None:
            checks.append(CheckResult(
                f"ownership:{service_name}",
                False,
                "selected profile declares a managed service with no process",
                "a launcher-owned process definition",
                "Add the managed process definition or select a reused/external profile.",
            ))
        elif (
            ownership == "own"
            and process is not None
            and process.launch_mode == "reuse"
        ):
            checks.append(CheckResult(
                f"ownership:{service_name}",
                False,
                "process is declared reuse but the selected profile is managed",
                "a launcher-owned process definition",
                "Select a reused/external model profile or add a managed process definition.",
            ))
    probed_services = {probe.name for probe in probes}
    for process in flat_processes:
        name = process.name
        if (
            name
            and process.launch_mode == "reuse"
            and name not in probed_services
        ):
            checks.append(CheckResult(
                f"endpoint:{name}",
                False,
                "reuse has no deployment endpoint",
                "a deployment profile with a readiness endpoint",
                "Declare the reused service in the selected model deployment profile.",
            ))
    if "os" in contract:
        required_os = contract["os"]
        allowed = [required_os] if isinstance(required_os, str) else required_os
        detected_os = _normalize_os(sys.platform)
        normalized_allowed = [_normalize_os(item) for item in allowed]
        checks.append(CheckResult(
            "os", detected_os in normalized_allowed, detected_os,
            ", ".join(normalized_allowed),
            f"See {_DOCS}#preflight-platform.",
        ))
    if "arch" in contract:
        arch = contract["arch"]
        unless_env_present = (
            arch.get("unless_env_present") if isinstance(arch, dict) else None
        )
        unless_env_truthy = (
            arch.get("unless_env_truthy") if isinstance(arch, dict) else None
        )
        unless_config = arch.get("unless_config") if isinstance(arch, dict) else None
        allowed_value = arch["allowed"] if isinstance(arch, dict) else arch
        allowed_arch = [allowed_value] if isinstance(allowed_value, str) else allowed_value
        detected_arch = _normalize_arch(platform.machine())
        normalized_allowed = [_normalize_arch(item) for item in allowed_arch]
        escaped_by_env = bool(
            (unless_env_present and os.environ.get(unless_env_present))
            or (unless_env_truthy and _env_truthy(unless_env_truthy))
        )
        escaped_by_config = bool(
            unless_config
            and _configured_value(flat_processes, sample_base, unless_config)
        )
        escaped = escaped_by_env or escaped_by_config
        escape_source = (
            f"{unless_env_present or unless_env_truthy} set"
            if escaped_by_env
            else f"{unless_config} configured"
        )
        checks.append(CheckResult(
            "arch",
            detected_arch in normalized_allowed or escaped,
            f"{detected_arch} ({escape_source})" if escaped else detected_arch,
            ", ".join(normalized_allowed),
            f"Set {unless_env_present or unless_env_truthy} or configure "
            f"{unless_config}. See {_DOCS}#preflight-platform."
            if (unless_env_present or unless_env_truthy) and unless_config
            else f"Set {unless_env_present or unless_env_truthy}. See {_DOCS}#preflight-platform."
            if unless_env_present or unless_env_truthy
            else f"Configure {unless_config}. See {_DOCS}#preflight-platform."
            if unless_config
            else f"See {_DOCS}#preflight-platform.",
            skipped=escaped,
        ))
    if "python" in contract:
        python_requirement, python_unless_env = _version_requirement(contract["python"])
        checks.append(_version_check(
            "python", platform.python_version(), python_requirement,
            f"See {_DOCS}#preflight-python.",
        ) if not (python_unless_env and _env_truthy(python_unless_env)) else CheckResult(
            "python", True, f"skipped because {python_unless_env} is set",
            str(python_requirement), f"See {_DOCS}#preflight-python.", skipped=True,
        ))

    has_local = any(service.ownership == "own" for service in services)
    if not services:
        has_local = bool(flat_processes) or not probes
    docker_applicable = has_local and (
        not flat_processes or _local_services_require_docker(flat_processes)
    )
    if has_local and "nvidia_driver" in contract:
        driver_requirement, driver_unless_env = _version_requirement(
            contract["nvidia_driver"]
        )
        driver_version, driver_error = _driver_version()
        checks.append(_version_check(
            "nvidia_driver", driver_version, driver_requirement,
            f"Upgrade the NVIDIA driver. See {_DOCS}#preflight-nvidia-driver.",
            minimum=True,
            failure_detail=driver_error,
        ) if not (driver_unless_env and _env_truthy(driver_unless_env)) else CheckResult(
            "nvidia_driver", True, f"skipped because {driver_unless_env} is set",
            str(driver_requirement), f"See {_DOCS}#preflight-nvidia-driver.", skipped=True,
        ))
    if has_local and contract.get("cuda") is not None:
        required_cuda = contract["cuda"]
        cuda_version = _cuda_version()
        if required_cuda is True:
            checks.append(CheckResult(
                "cuda", cuda_version is not None, cuda_version or "not found", "available",
                f"Install a compatible NVIDIA driver. See {_DOCS}#preflight-nvidia-driver.",
            ))
        elif required_cuda is not False:
            checks.append(_version_check(
                "cuda", cuda_version, required_cuda,
                f"Install a compatible NVIDIA driver. See {_DOCS}#preflight-nvidia-driver.",
            ))
    if docker_applicable and "docker" in contract:
        docker_requirement, docker_unless_env = _version_requirement(contract["docker"])
        docker_version, docker_error = _docker_version()
        checks.append(_version_check(
            "docker", docker_version, docker_requirement,
            f"Install or start Docker. See {_DOCS}#preflight-docker.",
            minimum=True,
            failure_detail=docker_error,
        ) if not (docker_unless_env and _env_truthy(docker_unless_env)) else CheckResult(
            "docker", True, f"skipped because {docker_unless_env} is set",
            str(docker_requirement), f"See {_DOCS}#preflight-docker.", skipped=True,
        ))
    if docker_applicable and contract.get("nvidia_container_toolkit"):
        success, output = _command_output(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
        )
        runtime_found = success and re.search(r'"nvidia"\s*:', output) is not None
        checks.append(CheckResult(
            "nvidia_container_toolkit",
            runtime_found,
            "Docker nvidia runtime found"
            if runtime_found else (output or "Docker nvidia runtime not found"),
            "NVIDIA Container Toolkit Docker runtime",
            f"Configure the Docker nvidia runtime. See {_DOCS}#preflight-container-toolkit.",
        ))
    if has_local and contract.get("vulkan"):
        loader = ctypes.util.find_library("vulkan")
        checks.append(CheckResult(
            "vulkan_loader", loader is not None, loader or "not found", "Vulkan loader",
            f"Install the Vulkan loader. See {_DOCS}#preflight-vulkan.",
        ))
    if has_local and contract.get("nvenc"):
        nvenc = ctypes.util.find_library("nvidia-encode")
        checks.append(CheckResult(
            "nvenc_library", nvenc is not None, nvenc or "not found", "NVENC driver library",
            f"See {_DOCS}#preflight-nvenc.",
        ))
    if "node" in contract:
        node_requirement, node_unless_env = _version_requirement(contract["node"])
        if node_unless_env and _env_truthy(node_unless_env):
            checks.append(CheckResult(
                "node", True, f"skipped because {node_unless_env} is set",
                str(node_requirement), f"See {_DOCS}#preflight-node.", skipped=True,
            ))
        else:
            checks.append(_version_check(
                "node", _node_version(), node_requirement,
                f"Install a supported Node.js release. See {_DOCS}#preflight-node.",
            ))
    for command_spec in contract.get("commands", []):
        if isinstance(command_spec, str):
            command = command_spec
            unless_env = None
        else:
            command = command_spec["name"]
            unless_env = command_spec.get("unless_env_truthy")
        if unless_env and _env_truthy(unless_env):
            checks.append(CheckResult(
                f"command:{command}", True,
                f"skipped because {unless_env} is set", command,
                f"See {_DOCS}#preflight-command.", skipped=True,
            ))
            continue
        executable = shutil.which(command)
        checks.append(CheckResult(
            f"command:{command}", executable is not None, executable or "not found", command,
            f"Install `{command}` and ensure it is available on PATH. See {_DOCS}#preflight-command.",
        ))

    if has_local:
        checks.extend(_disk_checks(
            contract.get("disk_gb_free"),
            flat_processes,
            services,
            sample_base,
            preparation_space,
        ))
    elif preparation_space:
        raise ContractError("preparation-space inventory requires launcher-owned services")

    process_by_name = {
        process.name: process
        for process in flat_processes
        if process.name
    }
    verified_services: set[str] = set()
    checks.extend(_ephemeral_port_checks(services))
    checks.extend(_owned_bind_conflict_checks(services))
    for service in services:
        if service.ownership != "own" or service.port is None:
            continue
        inspection = _port_is_free(
            service.port, service.proto, service.bind_host,
        )
        detected = inspection.detected
        verified = False
        inspection_warning = False
        process = process_by_name.get(service.name)
        ownership_probe = process.ownership_probe if process is not None else None
        remediation = inspection.remediation
        is_device_io_hub = (
            process is not None and process.command == "device_io_hub"
        )
        if (
            inspection.state == "occupied"
            and service.name == "hub"
            and (process is None or is_device_io_hub)
        ):
            remediation = (
                "If this is a LiveKit listener, stop or restart the "
                "`xr-ai-livekit-server` container. Otherwise, stop the process using "
                f"the port or configure a different port for the hub. See {_DOCS}#preflight-port."
            )
        probe_inspection_failure = (
            inspection.state == "uninspectable"
            and inspection.error_kind in {"missing", "command", "malformed"}
        )
        should_probe_ownership = (
            inspection.state == "occupied" or probe_inspection_failure
        )
        ownership_probe_attempted = False
        targeted_mismatch = False
        if should_probe_ownership and callable(ownership_probe):
            ownership_probe_attempted = True
            try:
                verified = ownership_probe(_effective_process_env(process)) is True
            except OwnershipProbeMismatch as exc:
                targeted_mismatch = True
                detected += f"; {exc.detected}"
                remediation = exc.remediation
            except Exception as exc:
                detected += f"; ownership probe failed ({type(exc).__name__})"
            if verified:
                if inspection.state == "occupied":
                    detected = "occupied by the expected managed service"
                else:
                    detected = (
                        "verified as the expected managed service; socket inspection "
                        f"failed ({inspection.evidence})"
                    )
                    inspection_warning = True
                verified_services.add(service.name)
        if (
            inspection.state == "uninspectable"
            and not verified
            and not targeted_mismatch
        ):
            detected += (
                "; ownership unverified"
                if ownership_probe_attempted
                else "; ownership not probed"
            )
        if inspection.state == "occupied" and service.health:
            if not verified and not targeted_mismatch:
                detected += "; ownership unverified"
        checks.append(CheckResult(
            f"port:{service.name}:{service.proto}:{service.port}",
            inspection.state == "clear" or verified, detected,
            f"no conflicting socket on {service.proto} port {service.port}",
            remediation,
            status="warning" if inspection_warning else None,
        ))
    if verified_services:
        services = tuple(
            replace(service, verified_running=True)
            if service.name in verified_services else service
            for service in services
        )

    declared_env: dict[str, tuple[bool, str]] = {
        item["name"]: (item["required"], item.get("docs") or "")
        for item in contract.get("env", [])
    }
    for probe in probes:
        if probe.api_key_env:
            _, docs = declared_env.get(probe.api_key_env, (False, ""))
            declared_env[probe.api_key_env] = (True, docs)
        if probe.ownership == "own":
            for credential in probe.required_credentials:
                _, docs = declared_env.get(credential, (False, ""))
                declared_env[credential] = (True, docs)
    for credential in required_credentials:
        if not isinstance(credential, str) or not credential:
            raise ContractError("required credential names must be non-empty strings")
        _, docs = declared_env.get(credential, (False, ""))
        declared_env[credential] = (True, docs)
    for name, (required, docs) in declared_env.items():
        present = bool(os.environ.get(name))
        warning = not present and not required
        checks.append(CheckResult(
            f"env:{name}", present or not required, "set" if present else "not set",
            "set" if required else "optional",
            docs or f"See {_DOCS}#preflight-credential.",
            status="warning" if warning else None,
        ))

    for probe in probes:
        if probe.ownership in {"reuse", "external"}:
            checks.append(_probe_endpoint(probe))

    expensive: list[str] = []
    if docker_applicable and contract.get("nvidia_container_toolkit"):
        expensive.append("container_gpu")
    if has_local and contract.get("vulkan"):
        expensive.append("vulkan_device")
    if has_local and contract.get("nvenc"):
        expensive.append("nvenc")
    failed_cheap = [check.name for check in checks if check.ok is False]
    if expensive and not failed_cheap:
        checks.extend(_expensive_checks(expensive, force=force_expensive))
    elif expensive:
        requirements = {
            "container_gpu": "GPU device visible inside a container",
            "vulkan_device": "usable Vulkan device",
            "nvenc": "functional NVENC session",
        }
        blocked_by = ", ".join(failed_cheap)
        checks.extend(
            CheckResult(
                name=name,
                ok=None,
                detected=f"blocked by failed cheap checks: {blocked_by}",
                required=requirements[name],
                remediation="Fix the failed cheap checks, then rerun preflight.",
                tier="expensive",
                skipped=True,
                status="blocked",
            )
            for name in expensive
        )

    return PreflightResult(
        contract_path=path,
        contract_files=contract_files,
        profiles=profiles,
        checks=tuple(checks),
        services=services,
    )
