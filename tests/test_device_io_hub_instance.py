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
from device_io_hub._instance import hold_instance_lock


def test_second_holder_is_refused_with_an_unverified_pid_record(tmp_path: Path) -> None:
    lock = tmp_path / "hub.lock"

    with hold_instance_lock(lock):
        with pytest.raises(StartupError, match="already running") as refused:
            with hold_instance_lock(lock):
                pytest.fail("a second hub acquired the lock")

    assert f"Last recorded PID (unverified): {os.getpid()}" in str(refused.value)


def test_read_only_holder_does_not_attribute_a_stale_pid_to_the_running_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = tmp_path / "hub.lock"
    recorded_pid = os.getppid()
    lock.write_text(f"{recorded_pid}\n", encoding="utf-8")

    def must_not_look_up_process(_path: Path) -> bytes:
        pytest.fail("an unverified PID must not be attributed to a process")

    monkeypatch.setattr(Path, "read_bytes", must_not_look_up_process)
    with monkeypatch.context() as holder_patch:
        holder_patch.setattr(_instance, "_open", lambda path: os.open(path, os.O_RDONLY))
        with hold_instance_lock(lock):
            holder_patch.undo()
            with pytest.raises(StartupError, match="already running") as refused:
                with hold_instance_lock(lock):
                    pytest.fail("a second hub acquired the lock")

    assert f"Last recorded PID (unverified): {recorded_pid}" in str(refused.value)
    assert "Running hub" not in str(refused.value)
    assert lock.read_text(encoding="utf-8") == f"{recorded_pid}\n"


def test_lock_is_free_again_after_the_holder_exits(tmp_path: Path) -> None:
    lock = tmp_path / "hub.lock"

    with hold_instance_lock(lock):
        pass

    with hold_instance_lock(lock):
        pass


@pytest.mark.parametrize("mask", [0o077, 0o777])
def test_restrictive_umask_keeps_lock_readable_for_the_next_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mask: int,
) -> None:
    lock = tmp_path / "hub.lock"
    previous_mask = os.umask(mask)
    try:
        with hold_instance_lock(lock):
            inode = lock.stat().st_ino
            assert stat.S_IMODE(lock.stat().st_mode) == 0o644

            real_open = os.open

            def open_as_another_user(path: Path, flags: int, mode: int = 0o777) -> int:
                # The kernel denies write access to another user's 0644 file.
                # Keep real descriptors and flock behavior on the same inode.
                if path == lock and flags == os.O_RDWR:
                    raise PermissionError("another user's lock")
                return real_open(path, flags, mode)

            monkeypatch.setattr(os, "open", open_as_another_user)
            with pytest.raises(StartupError, match="already running"):
                with hold_instance_lock(lock):
                    pytest.fail("a read-only contender acquired the held lock")
    finally:
        os.umask(previous_mask)

    # A read-only descriptor can acquire the released lock without replacing
    # the file, even though it cannot update the diagnostic owner record.
    with hold_instance_lock(lock):
        assert lock.stat().st_ino == inode
        assert stat.S_IMODE(lock.stat().st_mode) == 0o644


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
        with pytest.raises(StartupError, match=rf"Last recorded PID \(unverified\): {holder.pid}"):
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
