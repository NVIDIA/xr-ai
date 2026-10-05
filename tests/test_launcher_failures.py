# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure-reporting tests for the stack launcher."""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import pytest
import xr_ai_launcher._failure as _failure
import xr_ai_launcher._stack as _stack


def _set_log_dir(monkeypatch, root: Path) -> Path:
    monkeypatch.setenv("XR_AI_LOG_ROOT", str(root))
    monkeypatch.setenv("XR_AI_LOG_NAMESPACE", "failure-test")
    monkeypatch.setenv("XR_AI_LOG_TIMESTAMP", "2026-10-05_12-00-00")
    return root / "log_failure-test_2026-10-05_12-00-00"


def _reports(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("failure-report-*.json"))


def _only_report(log_dir: Path) -> Path:
    reports = _reports(log_dir)
    assert len(reports) == 1
    return reports[0]


def _write_module(tmp_path: Path, name: str, source: str) -> None:
    (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")


def _run_python_module(monkeypatch, tmp_path: Path, module: str) -> None:
    monkeypatch.setattr(_stack, "load_credentials", lambda: None)
    monkeypatch.setattr(_stack.shutil, "which", lambda _command: None)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    _stack.run_stack([_stack.Process("test-service", tmp_path, module)], tmp_path)


@pytest.mark.integration
def test_real_readiness_failure_records_unknown_stdout_and_stderr(
    monkeypatch, tmp_path, capsys
):
    log_dir = _set_log_dir(monkeypatch, tmp_path / "logs")
    _write_module(
        tmp_path,
        "unknown_failure",
        "import sys\n"
        "print('opaque stdout detail', flush=True)\n"
        "print('opaque stderr detail', file=sys.stderr, flush=True)\n"
        "raise SystemExit(7)\n",
    )

    with pytest.raises(SystemExit) as excinfo:
        _run_python_module(monkeypatch, tmp_path, "unknown_failure")

    assert excinfo.value.code == 7
    path = _only_report(log_dir)
    report = json.loads(path.read_text())
    assert report["service"] == "test-service"
    assert report["phase"] == "readiness"
    assert "unknown_failure" in report["command"]
    assert report["project_path"] == str(tmp_path)
    assert report["log_dir"] == str(log_dir)
    assert report["outcome"]["exit_code"] == 7
    assert "diagnosis" not in report
    assert {entry["stream"] for entry in report["evidence"]["tail"]} == {
        "stdout",
        "stderr",
    }
    evidence = "\n".join(entry["text"] for entry in report["evidence"]["tail"])
    assert "opaque stdout detail" in evidence
    assert "opaque stderr detail" in evidence
    terminal = capsys.readouterr().err
    assert "Process failure: test-service (readiness)" in terminal
    assert "Command:" in terminal
    assert f"Project: {tmp_path}" in terminal
    assert "Exit code: 7" in terminal
    assert "Captured output tail:" in terminal
    assert "opaque stderr detail" in terminal
    assert f"Failure report: {path}" in terminal
    assert f"Suggested log inspection: tail -n 200 {log_dir}/*.log" in terminal
    assert "Next step:" not in terminal


@pytest.mark.integration
def test_real_foreground_runtime_exit_reports_and_stays_nonzero(
    monkeypatch, tmp_path
):
    log_dir = _set_log_dir(monkeypatch, tmp_path / "logs")
    _write_module(
        tmp_path,
        "runtime_failure",
        "import sys\n"
        "from pathlib import Path\n"
        "ready = Path(sys.argv[sys.argv.index('--ready-file') + 1])\n"
        "ready.touch()\n"
        "print('runtime stopped unexpectedly', file=sys.stderr, flush=True)\n"
        "raise SystemExit(9)\n",
    )

    with pytest.raises(SystemExit) as excinfo:
        _run_python_module(monkeypatch, tmp_path, "runtime_failure")

    assert excinfo.value.code == 9
    report = json.loads(_only_report(log_dir).read_text())
    assert report["phase"] == "runtime"
    assert report["outcome"]["exit_code"] == 9


def test_evidence_is_bounded_and_credentials_are_redacted(
    monkeypatch, tmp_path, capsys
):
    env_secret = "z9Q!"
    values = {
        "hf": "unlisted-hf-credential-value",
        "ngc": "nvapi-abcdefghijklmnopqrstuv",
        "livekit": "unlisted-livekit-secret-value",
        "cli": "unlisted-cli-credential-value",
        "shape": "sk-abcdefghijklmnopqrstuv",
        "json": "opaque-quoted-credential",
        "jwt": "eyJabcdefghijk.abcdefghijklmnop.abcdefghijklmnop",
        "basic": "dXNlcjpwYXNz",
        "digest": "digest-secret",
    }
    monkeypatch.setenv("HF_TOKEN", env_secret)
    log_dir = _set_log_dir(monkeypatch, tmp_path / "logs")
    context = _failure.FailureContext(
        service="worker",
        command=[
            "worker",
            f"--hf-token={values['cli']}",
            f"https://person:{env_secret}@example.test/run?token={values['hf']}",
        ],
        project_path=tmp_path,
        config_path=tmp_path / "service.yaml",
    )
    for number in range(1000):
        context.add_line("stdout", f"line {number}: {'x' * 200}")
    context.add_line("stderr", f"HF_TOKEN={values['hf']}")
    context.add_line("stderr", f"NGC_API_KEY={values['ngc']}")
    context.add_line("stderr", f"export LIVEKIT_API_SECRET={values['livekit']}")
    context.add_line("stderr", f"session credential is {values['jwt']}")
    context.add_line("stderr", f"provider rejected {values['shape']}")
    context.add_line("stderr", f"Authorization: Bearer {env_secret}")
    context.add_line("stderr", json.dumps({"token": values["json"]}))
    context.add_line("stderr", f"Authorization: Basic {values['basic']}")
    context.add_line("stderr", f'"Authorization": "Basic {values["basic"]}"')
    context.add_line(
        "stderr",
        'Authorization: Digest username="user", realm="private", '
        f'response="{values["digest"]}"',
    )

    path = _failure.emit_failure_report(context, "runtime", returncode=2)

    assert path is not None
    assert path.parent == log_dir
    payload = path.read_text()
    terminal = capsys.readouterr().err
    for secret in (env_secret, *values.values()):
        assert secret not in payload
        assert secret not in terminal
    assert "<redacted>" in payload
    assert f"Config: {tmp_path / 'service.yaml'}" in terminal
    report = json.loads(payload)
    assert report["evidence"]["truncated"] is True
    assert report["evidence"]["bytes_kept"] <= _failure._MAX_EVIDENCE_BYTES
    assert len(report["evidence"]["tail"]) <= _failure._MAX_EVIDENCE_LINES


def test_noncredential_environment_values_are_not_redacted(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("LIVEKIT_API_KEY_ENV", "LIVEKIT_API_KEY")
    monkeypatch.setenv("SOME_CREDENTIALS_DIR", "/cache/public-credentials")
    context = _failure.FailureContext("worker", ["worker"], tmp_path, None)
    ordinary = (
        "TOKENIZERS_PARALLELISM=false "
        "LIVEKIT_API_KEY_ENV=LIVEKIT_API_KEY "
        "SOME_CREDENTIALS_DIR=/cache/public-credentials max_tokens=32"
    )
    context.add_line("stdout", ordinary)

    evidence = context.evidence()

    assert evidence["tail"] == [{"stream": "stdout", "text": ordinary}]


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("--password 'review-test phrase'", "review-test phrase"),
        ('--api-key "review double value"', "review double value"),
    ],
)
def test_quoted_cli_credentials_are_redacted_without_env_values(text, secret):
    redacted = _failure._redact(text, ())

    assert secret not in redacted
    assert redacted.endswith("<redacted>")


def test_overlong_authorization_line_is_omitted_before_label_can_be_lost(
    monkeypatch, tmp_path, capsys
):
    secret = "q" * 5000
    log_dir = _set_log_dir(monkeypatch, tmp_path / "logs")
    context = _failure.FailureContext("worker", ["worker"], tmp_path, None)
    context.add_line("stderr", f"Authorization: Bearer {secret}")

    path = _failure.emit_failure_report(context, "runtime", returncode=2)

    assert path is not None
    payload = path.read_text()
    terminal = capsys.readouterr().err
    assert secret not in payload
    assert secret not in terminal
    report = json.loads(payload)
    assert report["evidence"]["tail"] == [
        {
            "stream": "stderr",
            "text": f"[line omitted: exceeded {_failure._MAX_LINE_BYTES} bytes]",
        }
    ]
    assert report["evidence"]["truncated"] is True
    assert path.parent == log_dir


def test_evidence_line_count_cap_is_independent_of_byte_cap(tmp_path):
    context = _failure.FailureContext("worker", ["worker"], tmp_path, None)
    for number in range(_failure._MAX_EVIDENCE_LINES + 10):
        context.add_line("stdout", str(number))

    evidence = context.evidence()

    assert len(evidence["tail"]) == _failure._MAX_EVIDENCE_LINES
    assert evidence["tail"][0]["text"] == "10"
    assert evidence["truncated"] is True
    assert evidence["bytes_kept"] < _failure._MAX_EVIDENCE_BYTES


def test_post_redaction_expansion_remains_bounded(tmp_path):
    context = _failure.FailureContext(
        "worker",
        ["worker"],
        tmp_path,
        None,
        _secrets=("aaaa", "<red", "reda", "edac", "dact", "acte"),
    )
    context.add_line("stderr", "aaaa" * 1024)

    evidence = context.evidence()

    assert evidence["bytes_kept"] <= _failure._MAX_EVIDENCE_BYTES
    assert evidence["tail"] == [
        {
            "stream": "stderr",
            "text": (
                f"[line omitted: exceeded {_failure._MAX_EVIDENCE_BYTES} "
                "bytes after redaction]"
            ),
        }
    ]
    assert evidence["truncated"] is True


def test_report_write_uses_unique_name_around_preexisting_fixed_symlink(
    monkeypatch, tmp_path
):
    log_dir = _set_log_dir(monkeypatch, tmp_path / "logs")
    log_dir.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.write_text("unchanged", encoding="utf-8")
    (log_dir / "failure-report.json").symlink_to(victim)
    context = _failure.FailureContext("worker", ["worker"], tmp_path, None)

    first = _failure.emit_failure_report(context, "runtime", returncode=2)
    second = _failure.emit_failure_report(context, "runtime", returncode=2)

    assert first is not None and second is not None
    assert first != second
    assert first.parent == second.parent == log_dir
    assert first.stat().st_mode & 0o777 == 0o600
    assert second.stat().st_mode & 0o777 == 0o600
    assert victim.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.integration
def test_real_preferred_write_failure_uses_single_file_fallback(
    monkeypatch, tmp_path, capsys
):
    bad_root = tmp_path / "not-a-directory"
    bad_root.write_text("file", encoding="utf-8")
    configured_dir = _set_log_dir(monkeypatch, bad_root)
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    monkeypatch.setattr(_failure, "_gettempdir", lambda: str(fallback_dir))
    _write_module(tmp_path, "write_fallback", "raise SystemExit(6)\n")

    with pytest.raises(SystemExit) as excinfo:
        _run_python_module(monkeypatch, tmp_path, "write_fallback")

    assert excinfo.value.code == 6
    reports = sorted(fallback_dir.glob("xr-ai-failure-report-*.json"))
    assert len(reports) == 1
    assert reports[0].stat().st_mode & 0o777 == 0o600
    report = json.loads(reports[0].read_text())
    assert report["outcome"]["exit_code"] == 6
    assert report["log_dir"] == str(configured_dir)
    terminal = capsys.readouterr().err
    assert f"Failure report: {reports[0]}" in terminal
    assert "Suggested log inspection:" not in terminal


@pytest.mark.integration
def test_real_total_report_write_failure_does_not_replace_process_exit(
    monkeypatch, tmp_path, capsys
):
    bad_root = tmp_path / "not-a-directory"
    bad_root.write_text("file", encoding="utf-8")
    _set_log_dir(monkeypatch, bad_root)
    bad_fallback = tmp_path / "not-a-fallback-directory"
    bad_fallback.write_text("file", encoding="utf-8")
    monkeypatch.setattr(_failure, "_gettempdir", lambda: str(bad_fallback))
    _write_module(tmp_path, "write_failure", "raise SystemExit(6)\n")

    with pytest.raises(SystemExit) as excinfo:
        _run_python_module(monkeypatch, tmp_path, "write_failure")

    assert excinfo.value.code == 6
    terminal = capsys.readouterr().err
    assert "Failure report could not be written:" in terminal
    assert "Suggested log inspection:" not in terminal
    assert "captured output above" not in terminal


def test_spawn_error_is_reported_without_replacing_original(monkeypatch, tmp_path):
    log_dir = _set_log_dir(monkeypatch, tmp_path / "logs")
    original = OSError("executable format error")
    monkeypatch.setattr(_stack.shutil, "which", lambda _command: None)
    monkeypatch.setattr(_stack.subprocess, "Popen", Mock(side_effect=original))

    with pytest.raises(OSError) as excinfo:
        _stack._spawn(
            _stack.Process("worker", tmp_path, "worker", config="service.yaml"),
            tmp_path,
            tmp_path / "ready",
        )

    assert excinfo.value is original
    report = json.loads(_only_report(log_dir).read_text())
    assert report["phase"] == "spawn"
    assert report["config_path"] == str(tmp_path / "service.yaml")
    assert report["outcome"]["error"] == "OSError: executable format error"


def test_startup_cancellation_has_no_failure_artifact(monkeypatch, tmp_path):
    log_dir = _set_log_dir(monkeypatch, tmp_path / "logs")
    monkeypatch.setattr(_stack, "load_credentials", lambda: None)
    fake = Mock()
    fake.poll.return_value = None
    monkeypatch.setattr(_stack, "_spawn", Mock(return_value=fake))
    monkeypatch.setattr(_stack, "_wait_ready", Mock(side_effect=KeyboardInterrupt))
    monkeypatch.setattr(_stack, "_shutdown", Mock())

    with pytest.raises(SystemExit) as excinfo:
        _stack.run_stack([_stack.Process("worker", tmp_path, "worker")], tmp_path)

    assert excinfo.value.code == 130
    assert _reports(log_dir) == []


@pytest.mark.integration
@pytest.mark.parametrize("fails", [False, True])
def test_reporting_adds_no_subprocess_or_network_probe(monkeypatch, tmp_path, fails):
    log_dir = _set_log_dir(monkeypatch, tmp_path / "logs")
    if fails:
        source = "raise SystemExit(3)\n"
    else:
        source = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "Path(sys.argv[sys.argv.index('--ready-file') + 1]).touch()\n"
            "time.sleep(30)\n"
        )
    _write_module(tmp_path, "probe_test_service", source)
    monkeypatch.setattr(_stack, "load_credentials", lambda: None)
    monkeypatch.setattr(_stack.shutil, "which", lambda _command: None)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    original_popen = _stack.subprocess.Popen
    popen = Mock(side_effect=original_popen)
    network = Mock(side_effect=AssertionError("unexpected network probe"))
    monkeypatch.setattr(_stack.subprocess, "Popen", popen)
    monkeypatch.setattr(socket, "create_connection", network)
    monkeypatch.setattr(urllib.request, "urlopen", network)

    if fails:
        with pytest.raises(SystemExit) as excinfo:
            _stack.run_stack(
                [_stack.Process("test-service", tmp_path, "probe_test_service")],
                tmp_path,
            )
        assert excinfo.value.code == 3
        assert len(_reports(log_dir)) == 1
    else:
        _stack.run_stack(
            [_stack.Process("test-service", tmp_path, "probe_test_service")],
            tmp_path,
            exit_after_ready=True,
        )
        assert _reports(log_dir) == []

    assert popen.call_count == 1
    network.assert_not_called()


def test_parallel_readiness_failure_releases_other_waiters(tmp_path):
    dead = Mock()
    dead.poll.return_value = 5
    dead.returncode = 5
    alive = Mock()
    alive.poll.return_value = None
    alive_ready = tmp_path / "alive.ready"
    failures: list[BaseException] = []

    def wait() -> None:
        try:
            _stack._wait_ready_parallel(
                [
                    ("alive", alive_ready, alive),
                    ("dead", tmp_path / "dead.ready", dead),
                ]
            )
        except BaseException as exc:
            failures.append(exc)

    started = time.monotonic()
    waiter = threading.Thread(target=wait, daemon=True)
    waiter.start()
    waiter.join(timeout=2.0)
    released = not waiter.is_alive()
    if not released:
        alive_ready.touch()
        waiter.join(timeout=2.0)

    assert not waiter.is_alive(), "readiness waiter did not stop during test cleanup"
    assert released, "parallel readiness did not release the live sibling"
    assert len(failures) == 1
    assert isinstance(failures[0], SystemExit)
    assert failures[0].code == 5
    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize("returncode,status", [(0, 1), (-15, 143), (-9, 137)])
def test_failure_exit_status_is_nonzero_and_maps_signals(returncode, status):
    assert _failure.failure_exit_status(returncode) == status
