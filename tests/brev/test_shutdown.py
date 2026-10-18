# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "deployments/brev/launch.sh"


@unittest.skipUnless(sys.platform == "linux" and shutil.which("flock"), "requires Linux")
class ShutdownTests(unittest.TestCase):
    def test_stale_sample_pid_does_not_signal_unrelated_process(self):
        with tempfile.TemporaryDirectory() as directory, subprocess.Popen(
            ["sleep", "30"], start_new_session=True,
        ) as unrelated:
            state = Path(directory)
            (state / "sample.pid").write_text(str(unrelated.pid))
            (state / "sample.current").write_text("old-sample\n")
            (state / "models.active").write_text("recorded-stack\nmodels\n")
            try:
                result = subprocess.run(
                    ["bash", "-c", '''set -- help; source "$TEST_LAUNCH" >/dev/null
uv() { echo unexpected-model-stop; }
stop_locked
'''], env=os.environ | {"XR_AI_STATE": directory, "TEST_LAUNCH": str(LAUNCH)},
                    capture_output=True, text=True, timeout=5,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("does not match this launcher and state", result.stderr)
                self.assertIsNone(unrelated.poll())
                self.assertNotIn("unexpected-model-stop", result.stdout)
                self.assertEqual((state / "sample.pid").read_text(), str(unrelated.pid))
                self.assertTrue((state / "models.active").exists())
            finally:
                if unrelated.poll() is None:
                    unrelated.terminate()
                unrelated.wait(timeout=5)

    def test_stop_during_real_stack_readiness(self):
        runtime = Path(os.environ.get("XR_AI_TEST_ROOT", ROOT)).resolve()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / "bin").mkdir()
            (state / "models").mkdir()
            (state / "models/pyproject.toml").write_text('[project.scripts]\nmodels = "unused:main"\n')
            (state / "worker.py").write_text(
                "import os, time\nfrom pathlib import Path\n"
                "Path(os.environ['XR_AI_STATE'], 'worker.pid').write_text(str(os.getpid()))\n"
                "while True: time.sleep(1)\n"
            )
            (state / "stack.py").write_text(
                "from pathlib import Path\nfrom xr_ai_launcher import Process, run_stack\n"
                "run_stack([Process(name='worker', project='.', command='worker')], Path.cwd())\n"
            )
            # Replace only uv's environment resolution, not run_stack or its child handling.
            uv = state / "bin/uv"
            uv.write_text(
                f"#!{sys.executable}\nimport os, sys\n"
                "args = sys.argv[1:]\n"
                "if 'models' in args: sys.exit(0)\n"
                "script = 'worker.py' if 'worker' in args else 'stack.py'\n"
                "os.execv(sys.executable, [sys.executable, os.path.join(os.environ['XR_AI_STATE'], script)])\n"
            )
            uv.chmod(0o700)
            runner = state / "runner"
            runner.write_text(
                '#!/bin/bash\nset -- help; source "$TEST_LAUNCH" >/dev/null\n'
                'export PATH="$XR_AI_STATE/bin:$PATH"\n'
                'model_project() { echo "$XR_AI_STATE/models"; }\n'
                'run_sample sample sample\n'
            )
            runner.chmod(0o700)
            script = '''set -- help; source "$TEST_LAUNCH" >/dev/null
export PATH="$XR_AI_STATE/bin:$PATH"
SELF="$XR_AI_STATE/runner"
catalog() { echo probe >> "$STATE_DIR/probes"; printf 'sample\\tsample\\tsample\\n'; }
refresh_public_ip() { PUBLIC_IP=127.0.0.1; }
configure_runtime() { :; }
start_sample sample
for ((i=0; i<100; i++)); do [[ -s "$STATE_DIR/worker.pid" ]] && break; sleep .1; done
[[ -s "$STATE_DIR/worker.pid" ]]
kill -0 "$(<"$STATE_DIR/worker.pid")"
stop_locked
[[ ! -e "$PID_FILE" ]]
! kill -0 "$(<"$STATE_DIR/worker.pid")" 2>/dev/null
'''
            try:
                result = subprocess.run(
                    ["bash", "-c", script], capture_output=True, text=True, timeout=45,
                    env=os.environ | {
                        "HOME": directory, "XR_AI_STATE": directory, "TEST_LAUNCH": str(LAUNCH),
                        "XR_AI_DEPLOYMENT_PYTHON": sys.executable,
                        "PYTHONPATH": str(runtime / "utils/xr-ai-launcher"),
                    },
                )
                log = (state / "sample.log").read_text()
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr + log)
                self.assertIn("Aborting", log)
                self.assertNotIn("All processes ready", log)
                self.assertEqual((state / "probes").read_text(), "probe\n")
            finally:
                for name in ("worker.pid", "sample.pid"):
                    path = state / name
                    if path.exists():
                        try:
                            os.killpg(int(path.read_text()), signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_incomplete_shutdown_keeps_state_and_blocks_model_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["bash", "-c", '''set -- help; source "$TEST_LAUNCH" >/dev/null
prepare_state
echo 12345 > "$PID_FILE"
echo sample > "$CURRENT_FILE"
pid_alive() { return 0; }
sample_owned() { return 0; }
kill() { return 0; }
sleep() { :; }
uv() { echo unexpected-model-stop; }
stop_locked
'''], text=True, capture_output=True,
                env=os.environ | {"XR_AI_STATE": directory, "TEST_LAUNCH": str(LAUNCH)},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to switch", result.stderr)
            self.assertNotIn("unexpected-model-stop", result.stdout)
            self.assertTrue((Path(directory) / "sample.pid").exists())
