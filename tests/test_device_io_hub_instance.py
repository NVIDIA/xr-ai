# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A second DeviceIOHub must refuse to start instead of taking over the first."""
from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import device_io_hub.__main__ as hub_main
import pytest
from device_io_hub import _instance
from device_io_hub._errors import StartupError
from device_io_hub._instance import acquire_instance_lock


def test_second_holder_is_refused_and_told_how_to_find_the_first(tmp_path: Path) -> None:
    lock = tmp_path / "hub.lock"

    fd = acquire_instance_lock(lock)
    try:
        with pytest.raises(StartupError, match="already running") as refused:
            acquire_instance_lock(lock)
    finally:
        os.close(fd)

    assert f"fuser -v {lock}" in str(refused.value)


def test_lock_is_free_again_after_the_holder_closes_it(tmp_path: Path) -> None:
    lock = tmp_path / "hub.lock"

    os.close(acquire_instance_lock(lock))
    os.close(acquire_instance_lock(lock))


@pytest.mark.parametrize("mask", [0o077, 0o777])
def test_restrictive_umask_still_creates_a_lock_every_user_can_read(
    tmp_path: Path, mask: int,
) -> None:
    lock = tmp_path / "hub.lock"
    previous_mask = os.umask(mask)
    try:
        os.close(acquire_instance_lock(lock))
    finally:
        os.umask(previous_mask)

    assert stat.S_IMODE(lock.stat().st_mode) == 0o644


@pytest.mark.skipif(os.geteuid() == 0, reason="root opens any file")
def test_unreadable_lock_file_is_refused_with_a_banner(tmp_path: Path) -> None:
    lock = tmp_path / "hub.lock"
    lock.touch(mode=0o000)

    with pytest.raises(StartupError, match="Cannot open the DeviceIOHub lock file") as refused:
        acquire_instance_lock(lock)

    assert f"sudo rm {lock}" in str(refused.value)


@pytest.mark.integration
def test_lock_from_a_killed_hub_does_not_block_the_next_one(tmp_path: Path) -> None:
    lock = tmp_path / "hub.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import time
                from pathlib import Path
                from device_io_hub._instance import acquire_instance_lock
                acquire_instance_lock(Path({str(lock)!r}))
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
        with pytest.raises(StartupError, match="already running"):
            acquire_instance_lock(lock)
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=10)

    # SIGKILL skips every cleanup handler; the kernel still releases the lock.
    os.close(acquire_instance_lock(lock))


def test_entry_point_refuses_before_starting_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    lock = tmp_path / "hub.lock"
    monkeypatch.setattr(_instance, "LOCK_PATH", lock)
    monkeypatch.setattr(sys, "argv", ["device_io_hub"])

    async def must_not_run(**_kwargs: object) -> None:
        pytest.fail("the second hub started")

    monkeypatch.setattr(hub_main, "main", must_not_run)

    fd = acquire_instance_lock(lock)
    try:
        with pytest.raises(SystemExit) as exited:
            hub_main.run()
    finally:
        os.close(fd)

    assert exited.value.code == 1
    assert "Another DeviceIOHub is already running" in capsys.readouterr().err


def test_entry_point_holds_the_lock_while_running_and_after_main_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = tmp_path / "hub.lock"
    monkeypatch.setattr(_instance, "LOCK_PATH", lock)
    monkeypatch.setattr(sys, "argv", ["device_io_hub"])
    held: list[int] = []

    def acquire_and_remember() -> int:
        held.append(acquire_instance_lock())
        return held[-1]

    async def lock_is_held(**_kwargs: object) -> None:
        with pytest.raises(StartupError, match="already running"):
            acquire_instance_lock(lock)

    monkeypatch.setattr(hub_main, "acquire_instance_lock", acquire_and_remember)
    monkeypatch.setattr(hub_main, "main", lock_is_held)

    try:
        hub_main.run()
        # Still held for atexit handlers, until the process exits.
        with pytest.raises(StartupError, match="already running"):
            acquire_instance_lock(lock)
    finally:
        for fd in held:
            os.close(fd)
