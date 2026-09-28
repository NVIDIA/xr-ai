# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import signal

import pytest
from xr_ai_launcher import _artifacts


def test_sigterm_during_handler_restoration_exits_cleanly_and_restores(monkeypatch):
    original = signal.SIG_DFL
    current = original
    interrupted = False

    def install(signum, handler):
        nonlocal current, interrupted
        if handler is original and not interrupted:
            interrupted = True
            current(signum, None)
        previous = current
        current = handler
        return previous

    monkeypatch.setattr(_artifacts.signal, "getsignal", lambda _signal: current)
    monkeypatch.setattr(_artifacts.signal, "signal", install)

    with pytest.raises(SystemExit) as error:
        _artifacts.prepare_or_exit("example", lambda: None)

    assert error.value.code == 143
    assert current is original


def test_repeated_sigterm_does_not_interrupt_action_cleanup(monkeypatch):
    current = signal.SIG_DFL
    cleaned = False

    def install(_signal, handler):
        nonlocal current
        previous = current
        current = handler
        return previous

    def action():
        nonlocal cleaned
        try:
            current(signal.SIGTERM, None)
        finally:
            current(signal.SIGTERM, None)
            cleaned = True

    monkeypatch.setattr(_artifacts.signal, "getsignal", lambda _signal: current)
    monkeypatch.setattr(_artifacts.signal, "signal", install)

    with pytest.raises(SystemExit) as error:
        _artifacts.prepare_or_exit("example", action)

    assert error.value.code == 143
    assert cleaned
    assert current is signal.SIG_DFL


def test_unknown_prior_handler_restores_default(monkeypatch):
    current = None

    def install(_signal, handler):
        nonlocal current
        previous = current
        current = handler
        return previous

    monkeypatch.setattr(_artifacts.signal, "getsignal", lambda _signal: current)
    monkeypatch.setattr(_artifacts.signal, "signal", install)

    _artifacts.prepare_or_exit("example", lambda: None)

    assert current is signal.SIG_DFL
