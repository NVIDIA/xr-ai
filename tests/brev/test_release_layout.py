# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "deployments" / "brev"


@unittest.skipUnless(os.environ.get("XR_AI_TEST_ROOT"), "release checkout supplied by compatibility CI")
class ReleaseLayoutTests(unittest.TestCase):
    def test_real_release_catalog_and_model_commands(self):
        repo = Path(os.environ["XR_AI_TEST_ROOT"]).resolve()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / "public-ip").write_text("203.0.113.1\n")
            binary = state / "bin"
            binary.mkdir()
            smi = binary / "nvidia-smi"
            smi.write_text("""#!/bin/bash
case "$*" in
  *query-compute-apps*) exit 0 ;;
  *pci.bus_id*) echo '0, GPU-test, 0000:01:00.0, RTX PRO 6000 Blackwell, 12.0, 98304, 97000, 1304' ;;
  *encoder.stats*) echo '0, 0' ;;
  *) echo '0, RTX PRO 6000 Blackwell, 12.0, 98304, 580.1' ;;
esac
""")
            smi.chmod(0o755)
            # Model commands are mocked; catalog only needs these build tools on PATH.
            for name in ("node", "npm"):
                tool = binary / name
                tool.write_text("#!/bin/sh\nexit 0\n")
                tool.chmod(0o755)
            fixture = state / "probe.json"
            fixture.write_text(
                json.dumps(
                    {
                        "driver_version": "580.1",
                        "docker_version": "27",
                        "docker_runtime": True,
                        "codec_libraries": True,
                        "gpus": [
                            {"index": 0, "name": "Blackwell", "compute": 12.0, "memory_mib": 98304, "nvenc": True}
                        ],
                    }
                )
            )
            env = os.environ | {
                "XR_AI_ROOT": str(repo),
                "XR_AI_STATE": directory,
                "XR_AI_PROBE_FIXTURE": str(fixture),
                "PATH": str(binary) + ":" + os.environ["PATH"],
                "HF_TOKEN": "hf_test",
                "XR_AI_DEPLOYMENT_PYTHON": sys.executable,
                "XR_AI_WEB_PORT": "",
            }
            result = subprocess.run(
                ["bash", str(ROOT / "launch.sh"), "catalog"], env=env, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            catalog = json.loads(result.stdout)
            self.assertGreaterEqual(len(catalog["demos"]), 3)
            expected_samples = {}
            for path in (repo / "agent-samples").glob("*/pyproject.toml"):
                project = tomllib.loads(path.read_text())["project"]
                if path.parent.name != "model-servers" and any(
                    dependency.startswith("xr-ai-launcher") for dependency in project.get("dependencies", ())
                ):
                    expected_samples[path.parent.relative_to(repo).as_posix()] = next(iter(project["scripts"]))
            self.assertEqual({demo["project"]: demo["command"] for demo in catalog["demos"]}, expected_samples)
            for demo in catalog["demos"]:
                with self.subTest(sample=demo["id"]):
                    (state / "calls").write_text("")
                    self.assertEqual(demo["web_port"], 8080)
                    self.assertEqual(demo["blockers"], [] if os.uname().sysname == "Linux" else ["Linux is required"])
                    resolved = subprocess.run(
                        ["bash", str(ROOT / "launch.sh"), "catalog", "--resolve", demo["id"]],
                        env=env, text=True, capture_output=True,
                    )
                    self.assertEqual(resolved.returncode, 0 if demo["runnable"] else 1, resolved.stderr)
                    if demo["runnable"]:
                        self.assertEqual(resolved.stdout.rstrip("\n").split("\t"),
                                         [demo["id"], demo["project"], demo["command"], demo["recording_flag"]])
                    script = """set -- help
source ./launch.sh >/dev/null
uv() { echo "$*" >> "$STATE_DIR/calls"; }
configure_runtime "$PROJECT" 203.0.113.1
(run_sample "$PROJECT" "$COMMAND")
pid_alive() { return 1; }
stop_locked
if [[ -n "$RECORDING_FLAG" ]]; then
  (run_sample "$PROJECT" "$COMMAND" "$RECORDING_FLAG")
  stop_locked
fi
"""
                    run = subprocess.run(
                        ["bash", "-c", script], cwd=ROOT,
                        env=env | {"PROJECT": demo["project"], "COMMAND": demo["command"],
                                   "RECORDING_FLAG": demo["recording_flag"]},
                        text=True, capture_output=True,
                    )
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertIn("XR_AI_SAMPLE_START", run.stdout)
                    self.assertFalse((state / "models.active").exists())
                    models = next(path for path in (
                        repo / demo["project"] / "model-servers",
                        repo / "model-server-samples/model-servers", repo / "agent-samples/model-servers",
                    ) if (path / "pyproject.toml").is_file())
                    model_metadata = tomllib.loads((models / "pyproject.toml").read_text())["project"]
                    model_command = next(iter(model_metadata["scripts"]))
                    model_call = f"run --quiet --python 3.12 --project {models} {model_command}"
                    expected_calls = [
                        model_call,
                        f"run --quiet --python 3.12 --project {repo / demo['project']} {demo['command']}",
                        model_call + " --stop",
                    ]
                    if demo["recording_flag"]:
                        expected_calls += [model_call, expected_calls[1] + " " + demo["recording_flag"],
                                           model_call + " --stop"]
                    self.assertEqual((state / "calls").read_text().splitlines(), expected_calls)
            for sample, expected in (("auto", 0 if catalog["recommended"] else 1), ("unknown", 2)):
                resolved = subprocess.run(
                    ["bash", str(ROOT / "launch.sh"), "catalog", "--resolve", sample],
                    env=env, text=True, capture_output=True,
                )
                self.assertEqual(resolved.returncode, expected, resolved.stderr)
                if expected == 0:
                    self.assertEqual(resolved.stdout.split("\t")[0], catalog["recommended"])
