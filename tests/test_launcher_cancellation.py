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
