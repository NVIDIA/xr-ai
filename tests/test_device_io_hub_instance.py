# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A second DeviceIOHub must refuse to start instead of taking over the first."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import device_io_hub.__main__ as hub_main
import pytest
from device_io_hub import _instance
from device_io_hub._errors import StartupError
from device_io_hub._instance import hold_instance_lock


def test_second_holder_is_refused_and_told_who_runs(tmp_path: Path) -> None:
    lock = tmp_path / "hub.lock"

    with hold_instance_lock(lock):
        with pytest.raises(StartupError, match="already running") as refused:
            with hold_instance_lock(lock):
                pytest.fail("a second hub acquired the lock")

    assert f"pid {os.getpid()}" in str(refused.value)


def test_lock_is_free_again_after_the_holder_exits(tmp_path: Path) -> None:
    lock = tmp_path / "hub.lock"

    with hold_instance_lock(lock):
        pass

    with hold_instance_lock(lock):
        pass


def test_lock_from_a_killed_hub_does_not_block_the_next_one(tmp_path: Path) -> None:
    lock = tmp_path / "hub.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import sys, time
                from pathlib import Path
                from device_io_hub._instance import hold_instance_lock
                with hold_instance_lock(Path({str(lock)!r})):
                    print("locked", flush=True)
                    time.sleep(60)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        # pytest's pythonpath setting does not reach child processes.
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(StartupError, match=f"pid {holder.pid}"):
            with hold_instance_lock(lock):
                pass
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=10)

    # SIGKILL skips every cleanup handler; the kernel still releases the lock.
    with hold_instance_lock(lock):
        pass


def test_entry_point_refuses_before_starting_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    lock = tmp_path / "hub.lock"
    monkeypatch.setattr(_instance, "LOCK_PATH", lock)
    monkeypatch.setattr(sys, "argv", ["device_io_hub"])

    async def must_not_run(**_kwargs: object) -> None:
        pytest.fail("the second hub started")

    monkeypatch.setattr(hub_main, "main", must_not_run)

    with hold_instance_lock(lock):
        with pytest.raises(SystemExit) as exited:
            hub_main.run()

    assert exited.value.code == 1
    assert "Another DeviceIOHub is already running" in capsys.readouterr().err
