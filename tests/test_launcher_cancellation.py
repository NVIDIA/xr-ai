# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise launcher cancellation with real independently grouped CPU workers."""

import os
import signal
import subprocess
import sys
import time

import pytest
import xr_ai_launcher._stack as stack


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups and signals")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("phase", ["sequential", "parallel", "spawn", "spawn-parallel"])
@pytest.mark.parametrize("launch_mode", ["own", "persist"])
def test_cancel_startup_reaps_children(tmp_path, signum, phase, launch_mode):
    worker = """
import os
import signal
import time
from pathlib import Path
Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))
# Repeated cancellation while the launcher waits for cleanup must be harmless.
def stop(*_args):
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(0.1)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
while True:
    time.sleep(1)
"""
    for name in ("first", "second"):
        (tmp_path / f"{name}.py").write_text(worker)
    program = """
import os
import signal
import sys
from pathlib import Path
import xr_ai_launcher._stack as stack
stack.load_credentials = lambda: None
stack.shutil.which = lambda _: None
phase, mode, signum = sys.argv[1:]
original = stack._spawn
def spawn(*args, **kwargs):
    child = original(*args, **kwargs)
    Path(args[0].name + '.spawned').write_text(str(child.pid))
    if phase.startswith('spawn'):
        os.kill(os.getpid(), int(signum))
    return child
stack._spawn = spawn
items = [stack.Process('first', '.', 'first', launch_mode=mode)]
if 'parallel' in phase:
    items = [stack.Parallel(items + [stack.Process('second', '.', 'second')])]
items.insert(0, stack.Process('external', '.', 'unused', launch_mode='reuse'))
stack.run_stack(items, Path.cwd(), exit_after_ready=True)
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])}
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) as external:
        parent = subprocess.Popen(
            [sys.executable, "-c", program, phase, launch_mode, str(signum)],
            cwd=tmp_path, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            expected = 2 if phase == "parallel" else 1
            if not phase.startswith("spawn"):
                deadline = time.monotonic() + 10
                while len(list(tmp_path.glob("*.pid"))) < expected:
                    assert parent.poll() is None
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
                parent.send_signal(signum)
            _, stderr = parent.communicate(timeout=10)
            assert parent.returncode == 128 + signum, stderr.decode()
            assert external.poll() is None
            children = [int(path.read_text()) for path in tmp_path.glob("*.spawned")]
            assert len(children) == expected
            for pid in children:
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
        finally:
            # Even a regressed launcher must not leak the test's disposable workers.
            if parent.poll() is None:
                parent.kill()
            parent.communicate(timeout=5)
            for path in tmp_path.glob("*.spawned"):
                try:
                    os.killpg(int(path.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass  # ignored: successful launcher cleanup already reaped this worker.
            external.terminate()


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups and signals")
@pytest.mark.parametrize("phase", ["sequential", "parallel"])
@pytest.mark.parametrize("trigger", ["failure", "SIGINT", "SIGTERM"])
def test_later_startup_abort_preserves_ready_persist(tmp_path, phase, trigger):
    worker = """
import os
import signal
import sys
import time
from pathlib import Path
name = Path(__file__).stem
Path(name + '.pid').write_text(str(os.getpid()))
if name == 'last':
    deadline = time.monotonic() + 10
    while not all(Path(path).exists() for path in (
        'llm.initialized', 'vlm.initialized', 'owned.initialized', 'pending.pid',
    )):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    trigger = os.environ['TEST_STARTUP_TRIGGER']
    if trigger == 'failure':
        raise SystemExit(7)
    os.kill(os.getppid(), getattr(signal, trigger))
elif name != 'pending':
    Path(sys.argv[sys.argv.index('--ready-file') + 1]).touch()
    Path(name + '.initialized').touch()
while True:
    time.sleep(1)
"""
    for name in ("llm", "vlm", "owned", "pending", "last"):
        (tmp_path / f"{name}.py").write_text(worker)
    program = """
import sys
from pathlib import Path
import xr_ai_launcher._stack as stack
stack.load_credentials = lambda: None
stack.shutil.which = lambda _: None
ready = [stack.Process(name, '.', name, launch_mode='persist') for name in ('llm', 'vlm')]
ready.append(stack.Process('owned', '.', 'owned'))
pending = stack.Process('pending', '.', 'pending', launch_mode='persist')
last = stack.Process('last', '.', 'last')
if sys.argv[1] == 'parallel':
    # Failure comes first so cleanup cannot rely on the waiter's iteration order.
    items = [stack.Parallel([last, pending, *ready])]
else:
    items = [*ready, stack.Parallel([last, pending])]
stack.run_stack(items, Path.cwd(), exit_after_ready=True)
"""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path]),
        "TEST_STARTUP_TRIGGER": trigger,
    }
    parent = subprocess.Popen(
        [sys.executable, "-c", program, phase],
        cwd=tmp_path, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, stderr = parent.communicate(timeout=15)
        expected_status = 7 if trigger == "failure" else 128 + getattr(signal, trigger)
        assert parent.returncode == expected_status, stderr.decode()
        for name in ("llm", "vlm"):
            os.kill(int((tmp_path / f"{name}.pid").read_text()), 0)
        for name in ("owned", "pending", "last"):
            with pytest.raises(ProcessLookupError):
                os.kill(int((tmp_path / f"{name}.pid").read_text()), 0)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.communicate(timeout=5)
        # Explicitly stop the retained servers and any workers leaked by a regression.
        for path in tmp_path.glob("*.pid"):
            try:
                os.killpg(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass  # ignored: the launcher already reaped this worker.


def test_cancellation_restores_handlers_after_cleanup(monkeypatch, tmp_path):
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    child = object()
    monkeypatch.setattr(stack, "load_credentials", lambda: None)

    def spawn(*_args, **_kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        return child

    def shutdown(processes, no_kill):
        assert processes == {"worker": child}
        assert no_kill == set()
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(stack, "_spawn", spawn)
    monkeypatch.setattr(stack, "_shutdown", shutdown)
    with pytest.raises(SystemExit, match="143"):
        stack.run_stack([stack.Process("worker", ".", "worker", launch_mode="persist")], tmp_path)
    assert {sig: signal.getsignal(sig) for sig in original} == original


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("operation", ["set", "clear"])
def test_cancellation_does_not_reenter_readiness_lock(tmp_path, operation):
    program = """
import os
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
import xr_ai_launcher._stack as stack
stack.load_credentials = lambda: None
original = stack.stack_health
@contextmanager
def stack_health(*args):
    with original(*args) as initialized:
        method = getattr(initialized, sys.argv[1])
        def interrupt():
            # Deliver repeated signals while an Event method owns its lock.
            with initialized._cond:
                os.kill(os.getpid(), signal.SIGTERM)
                os.kill(os.getpid(), signal.SIGINT)
            method()
        setattr(initialized, sys.argv[1], interrupt)
        yield initialized
stack.stack_health = stack_health
stack.run_stack([], Path.cwd(), exit_after_ready=True)
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run(
        [sys.executable, "-c", program, operation],
        cwd=tmp_path, env=env, capture_output=True, timeout=5,
    )
    assert result.returncode == (143 if operation == "set" else 0), result.stderr.decode()
