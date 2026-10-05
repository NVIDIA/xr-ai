# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort failure reports for launcher-managed processes."""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import sys
import tempfile
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import gettempdir as _gettempdir
from typing import Any, Iterable

from ._processes import _configured_log_dir

_MAX_EVIDENCE_BYTES = 32 * 1024
_MAX_LINE_BYTES = 4096
_MAX_EVIDENCE_LINES = 256
_DRAIN_TIMEOUT = 0.25

_CREDENTIAL_ENV_NAMES = frozenset(
    {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NGC_API_KEY",
        "NVIDIA_API_KEY",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    }
)
_AUTHORIZATION_RE = re.compile(
    r"(?i)(\bauthorization[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|digest[^\r\n]*|(?:basic|bearer)\s+[^\s,;]+|[^\s,;]+)"
)
_LABELED_SECRET_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])((?:[A-Z0-9]+[_-])*"
    r"(?:access[_-]?token|api[_-]?key|password|passwd|secret|token)"
    r"[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+")
_CLI_SECRET_RE = re.compile(
    r"(?i)(--(?:[a-z0-9]+-)*(?:access-token|api-key|password|secret|token)"
    r"(?:=|\s+))(?:\"[^\"]*\"|'[^']*'|[^\s]+)"
)
_TOKEN_SHAPE_RE = re.compile(
    r"\b(?:hf_[A-Za-z0-9]{10,}|nvapi-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9_-]{10,}|"
    r"gh[opusr]_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,}|"
    r"AKIA[A-Z0-9]{12,})\b"
)
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]{8,})?\b"
)
_URL_USERINFO_RE = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://)([^/@\s]+)@")
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:access_token|api_key|apikey|auth|authorization|password|secret|token)=)"
    r"([^&#\s]+)"
)


def _known_secrets() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                value
                for key, value in os.environ.items()
                if key.upper() in _CREDENTIAL_ENV_NAMES and len(value) >= 4
            },
            key=len,
            reverse=True,
        )
    )


def _redact(text: str, secrets: Iterable[str]) -> str:
    redacted = text
    for secret in secrets:
        redacted = redacted.replace(secret, "<redacted>")
    redacted = _URL_USERINFO_RE.sub(r"\1<redacted>@", redacted)
    redacted = _QUERY_SECRET_RE.sub(r"\1<redacted>", redacted)
    redacted = _AUTHORIZATION_RE.sub(r"\1<redacted>", redacted)
    redacted = _LABELED_SECRET_RE.sub(r"\1<redacted>", redacted)
    redacted = _BEARER_RE.sub(r"\1<redacted>", redacted)
    redacted = _CLI_SECRET_RE.sub(r"\1<redacted>", redacted)
    redacted = _TOKEN_SHAPE_RE.sub("<redacted>", redacted)
    return _JWT_RE.sub("<redacted>", redacted)


@dataclass
class FailureContext:
    """Launch metadata and a size-bounded tail of already-forwarded output."""

    service: str
    command: list[str]
    project_path: Path
    config_path: Path | None
    _lines: deque[tuple[str, str, int]] = field(default_factory=deque)
    _bytes: int = 0
    _truncated: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _forwarders: list[threading.Thread] = field(default_factory=list)
    _secrets: tuple[str, ...] = field(default_factory=_known_secrets)

    def redact(self, text: str) -> str:
        return _redact(text, self._secrets)

    def add_line(self, stream: str, line: str) -> None:
        raw_size = len(line.encode("utf-8", errors="replace"))
        line_truncated = raw_size > _MAX_LINE_BYTES
        if line_truncated:
            line = f"[line omitted: exceeded {_MAX_LINE_BYTES} bytes]"
        size = len(line.encode("utf-8", errors="replace"))
        with self._lock:
            if line_truncated:
                self._truncated = True
            while self._lines and self._bytes + size > _MAX_EVIDENCE_BYTES:
                _, _, removed = self._lines.popleft()
                self._bytes -= removed
                self._truncated = True
            self._lines.append((stream, line, size))
            self._bytes += size
            while len(self._lines) > _MAX_EVIDENCE_LINES:
                _, _, removed = self._lines.popleft()
                self._bytes -= removed
                self._truncated = True

    def add_forwarder(self, thread: threading.Thread) -> None:
        self._forwarders.append(thread)

    def evidence(self) -> dict[str, Any]:
        for thread in self._forwarders:
            thread.join(timeout=_DRAIN_TIMEOUT)
        with self._lock:
            snapshot = list(self._lines)
            truncated = self._truncated

        lines: deque[dict[str, str]] = deque()
        bytes_kept = 0
        for stream, line, _ in snapshot:
            safe = self.redact(line)
            size = len(safe.encode("utf-8", errors="replace"))
            # Redaction can expand a short match, so enforce the byte cap again.
            while lines and bytes_kept + size > _MAX_EVIDENCE_BYTES:
                removed = lines.popleft()
                bytes_kept -= len(removed["text"].encode("utf-8", errors="replace"))
                truncated = True
            if size > _MAX_EVIDENCE_BYTES:
                safe = f"[line omitted: exceeded {_MAX_EVIDENCE_BYTES} bytes after redaction]"
                size = len(safe.encode("utf-8", errors="replace"))
                truncated = True
            lines.append({"stream": stream, "text": safe})
            bytes_kept += size
        return {
            "tail": list(lines),
            "bytes_kept": bytes_kept,
            "truncated": truncated,
        }


def _write_json(directory: Path, report: dict[str, Any], prefix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=directory,
        prefix=prefix,
        suffix=".json",
        text=True,
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def _write_report(
    report: dict[str, Any], preferred_dir: Path | None
) -> tuple[Path | None, str | None]:
    errors: list[str] = []
    if preferred_dir is not None:
        try:
            return _write_json(preferred_dir, report, "failure-report-"), None
        except OSError as exc:
            errors.append(f"{preferred_dir}: {exc}")

    fallback_dir = Path(_gettempdir())
    try:
        return _write_json(fallback_dir, report, "xr-ai-failure-report-"), None
    except OSError as exc:
        errors.append(f"{fallback_dir}: {exc}")
    return None, "; ".join(errors) or "no writable report directory"


def emit_failure_report(
    context: FailureContext,
    phase: str,
    *,
    returncode: int | None = None,
    error: BaseException | None = None,
) -> Path | None:
    """Print and persist a failure report without propagating reporting errors."""
    try:
        evidence = context.evidence()
        safe_command = [context.redact(arg) for arg in context.command]
        error_text = context.redact(f"{type(error).__name__}: {error}") if error else None
        signal_number = -returncode if returncode is not None and returncode < 0 else None
        signal_name = None
        if signal_number is not None:
            try:
                signal_name = signal.Signals(signal_number).name
            except ValueError:
                signal_name = "UNKNOWN"
        configured_dir = _configured_log_dir()
        report = {
            "schema_version": 1,
            "service": context.redact(context.service),
            "phase": phase,
            "command": safe_command,
            "project_path": context.redact(str(context.project_path)),
            "config_path": (
                context.redact(str(context.config_path))
                if context.config_path is not None
                else None
            ),
            "log_dir": (
                context.redact(str(configured_dir))
                if configured_dir is not None
                else None
            ),
            "outcome": {
                "exit_code": returncode if returncode is not None and returncode >= 0 else None,
                "signal": (
                    {"number": signal_number, "name": signal_name}
                    if signal_number is not None
                    else None
                ),
                "error": error_text,
            },
            "evidence": evidence,
        }
        path, write_error = _write_report(report, configured_dir)

        lines = [
            "─" * 78,
            f"Process failure: {report['service']} ({phase})",
            f"Command: {shlex.join(safe_command)}",
            f"Project: {report['project_path']}",
        ]
        if report["config_path"] is not None:
            lines.append(f"Config: {report['config_path']}")
        if error_text:
            lines.append(f"Error: {error_text}")
        elif signal_number is not None:
            lines.append(f"Terminated by signal: {signal_name} ({signal_number})")
        else:
            lines.append(f"Exit code: {returncode}")
        if evidence["tail"]:
            lines.append("Captured output tail:")
            lines.extend(
                f"  [{entry['stream']}] {entry['text'][-500:]}"
                for entry in evidence["tail"][-6:]
            )
        if (
            configured_dir is not None
            and path is not None
            and path.parent == configured_dir
        ):
            lines.append(
                "Suggested log inspection: "
                f"tail -n 200 {shlex.quote(report['log_dir'])}/*.log"
            )
        elif evidence["tail"]:
            lines.append("Use the captured output above to investigate the failure.")
        if path is not None:
            lines.append(f"Failure report: {context.redact(str(path))}")
        else:
            lines.append(
                "Failure report could not be written: "
                f"{context.redact(write_error or '')}"
            )
        lines.extend(("─" * 78, ""))
        print("\n".join(lines), file=sys.stderr, flush=True)
        return path
    except Exception as exc:
        try:
            print(
                f"Failure reporting failed: {context.redact(str(exc))}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        return None


def failure_exit_status(returncode: int | None) -> int:
    """Return a nonzero shell status, mapping zero to 1 and signals to 128+N."""
    if returncode is None or returncode == 0:
        return 1
    if returncode < 0:
        return 128 + min(-returncode, 127)
    return returncode
