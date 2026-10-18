# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "deployments" / "brev"


class LauncherRegressionTests(unittest.TestCase):
    def test_log_error_color_uses_severity_not_substrings(self):
        page = (ROOT / "ui.html").read_text()
        pattern = re.search(r"/(\(\^\|\\s\).*?)/i\.test", page).group(1)
        error = re.compile(pattern, re.IGNORECASE)
        self.assertIsNone(error.search("DEBUG smart_open {'errors': None}"))
        for line in ("ERROR service failed", "RuntimeError: failed", "Traceback"):
            self.assertIsNotNone(error.search(line))

    def test_runtime_configuration_uses_xr_ai_tls_for_both_hub_paths(self):
        for filename in ("device_io_hub.yaml", "xr_media_hub.yaml"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "xr-ai"
                yaml = root / "agent-samples/simple-vlm-example/yaml"
                yaml.mkdir(parents=True)
                (root / "agent-samples/simple-vlm-example/pyproject.toml").write_text("[project]\nname='simple'\n")
                hub_path = yaml / filename
                sans = (
                    "# web_server_extra_sans:\n#   - 198.51.100.9\n"
                    if filename == "device_io_hub.yaml"
                    else "web_server_extra_sans:\n  - 198.51.100.9\n"
                )
                hub_path.write_text(
                    """# lk_use_external_ip: true
web_server_port: 8080
cert_file: "/old/tls.crt"
key_file: "/old/tls.key"
"""
                    + sans
                )
                worker = yaml / "simple_vlm_example_worker.yaml"
                worker.write_text("voice_gate_yaml: voice_gate.yaml\n")
                state = Path(directory) / "state"
                state.mkdir()
                (state / "runtime-root").write_text(str(root) + "\n")
                command = ("set -- help; source launch.sh >/dev/null; "
                           "configure_runtime agent-samples/simple-vlm-example 203.0.113.7")
                env = os.environ | {"XR_AI_STATE": str(state), "XR_AI_DEPLOYMENT_PYTHON": sys.executable}
                env["XR_AI_LIVEKIT_WS_PORT"] = "9880"
                result = subprocess.run(["bash", "-c", command], cwd=ROOT, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(worker.read_text(), "voice_gate_yaml: voice_gate.yaml\n")
                hub = hub_path.read_text()
                self.assertIn("lk_use_external_ip: false", hub)
                self.assertIn("lk_skip_external_ip_validation: false", hub)
                self.assertIn('cert_file: ""', hub)
                self.assertIn('key_file: ""', hub)
                self.assertIn("web_server_extra_sans:\n  - 203.0.113.7", hub)
                self.assertIn('lk_port_ws: 9880', hub)
                self.assertIn('lk_internal_url: "ws://127.0.0.1:9880"', hub)
                self.assertNotIn("/old/tls", hub)
                self.assertNotIn("198.51.100.9", hub)
                for host, external in (("192.168.1.50", "false"), ("8.8.8.8", "true")):
                    result = subprocess.run(
                        ["bash", "-c", command.replace("203.0.113.7", host)],
                        cwd=ROOT, env=env, text=True, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    hub = hub_path.read_text()
                    self.assertIn(f"lk_use_external_ip: {external}", hub)
                    self.assertIn(f"lk_skip_external_ip_validation: {external}", hub)
                    self.assertIn(f"web_server_extra_sans:\n  - {host}", hub)
                # Current runtimes receive overrides without changing the YAML.
                hook = root / "services/device-io-hub/device_io_hub/deployment.py"
                hook.parent.mkdir(parents=True)
                hook.touch()
                original = hub_path.read_text()
                result = subprocess.run(
                    ["bash", "-c", command + '; printf "%s" "$DEVICE_IO_HUB_NETWORK"'],
                    cwd=ROOT, env=env, text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(hub_path.read_text(), original)
                overrides = json.loads(result.stdout)
                self.assertEqual(overrides["web_server_extra_sans"], ["203.0.113.7"])
                self.assertEqual(overrides["cert_file"], "")
                self.assertEqual(overrides["key_file"], "")
                self.assertEqual(overrides["lk_port_ws"], 9880)
                self.assertNotIn("lk_internal_url", overrides)
