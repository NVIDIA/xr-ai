# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only inventory tests: no Docker daemon, GPU, or model downloads."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("deployment_catalog", ROOT / "deployment/brev/catalog.py")
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.memory = "98304"
        self.encoder = "0"
        self.driver = "580.1"
        self.docker = "27.0.1"
        self.runtimes = '{"nvidia":{}}'
        self.tools = True
        self.codec = True
        self.commands = []
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(catalog.subprocess, "check_output", side_effect=self.output))
        self.stack.enter_context(patch.object(catalog.ctypes.util, "find_library", side_effect=lambda _: self.codec))
        self.stack.enter_context(patch.object(catalog.shutil, "which", side_effect=lambda _: self.tools))
        self.stack.enter_context(patch.object(catalog.sys, "platform", "linux"))

    def output(self, argv, **_kwargs):
        self.commands.append(argv)
        query = " ".join(argv)
        if "query-compute-apps" in query:
            return ""
        if "pci.bus_id" in query:
            return f"0, GPU-test, 0000:01:00.0, NVIDIA GB10, 12.1, {self.memory}, {self.memory}, {self.memory}"
        if "encoder.stats" in query:
            return f"0, {self.encoder}"
        if "query-gpu" in query:
            return f"0, NVIDIA GB10, 12.1, {self.memory}, {self.driver}"
        if argv[:2] == ["docker", "version"]:
            return self.docker
        if argv[:2] == ["docker", "info"]:
            return self.runtimes
        return "commit-for-test"

    def test_current_samples_are_discovered_without_a_hardcoded_list(self):
        inventory = catalog.catalog(ROOT)
        expected = {path.parent.relative_to(ROOT).as_posix()
                    for path in (ROOT / "agent-samples").glob("*/model-servers/pyproject.toml")}
        self.assertEqual({str(Path(item["project"]) / "model-servers") for item in inventory["demos"]}, expected)
        self.assertEqual(inventory["recommended"], "simple-vlm")
        self.assertTrue(all(item["runnable"] for item in inventory["demos"]))
        self.assertTrue(all(not item["blockers"] for item in inventory["demos"]))
        self.assertTrue(all(argv[0] in {"nvidia-smi", "docker", "git"} for argv in self.commands))

    def test_spark_preserves_gpu_with_unavailable_memory(self):
        for value in ("N/A", "[N/A]", "Not Supported", "[Not Supported]"):
            with self.subTest(memory=value):
                self.memory = value
                inventory = catalog.catalog(ROOT)
                self.assertEqual(inventory["driver_version"], "580.1")
                self.assertEqual(inventory["gpus"], [{
                    "index": 0, "name": "NVIDIA GB10", "compute": 12.1,
                    "memory_mib": None, "nvenc": True,
                }])
                self.assertTrue(all(item["runnable"] for item in inventory["demos"]))

    def test_zero_encoder_sessions_means_nvenc_supported_not_one_encoder(self):
        inventory = catalog.catalog(ROOT)
        self.assertIs(inventory["gpus"][0]["nvenc"], True)
        self.encoder = "[Not Supported]"
        inventory = catalog.catalog(ROOT)
        self.assertIs(inventory["gpus"][0]["nvenc"], False)
        self.assertFalse(any(item["runnable"] for item in inventory["demos"]))

    def test_host_blockers_are_visible(self):
        self.driver, self.docker, self.runtimes, self.codec = "565.1", "23", "{}", False
        inventory = catalog.catalog(ROOT)
        self.assertEqual(inventory["recommended"], "")
        for item in inventory["demos"]:
            self.assertFalse(item["runnable"])
            self.assertEqual(len(item["blockers"]), 4)

    def test_render_requires_build_tools_without_blocking_other_samples(self):
        self.tools = False
        inventory = catalog.catalog(ROOT)
        render = next(item for item in inventory["demos"] if item["id"] == "xr-render-demo")
        self.assertIn("XR Render web client needs Node.js and npm", render["blockers"])
        self.assertEqual(inventory["recommended"], "simple-vlm")

    def test_missing_tools_report_a_blocked_inventory(self):
        with patch.object(catalog.subprocess, "check_output", side_effect=FileNotFoundError):
            inventory = catalog.catalog(ROOT)
        self.assertEqual(inventory["gpus"], [])
        self.assertFalse(any(item["runnable"] for item in inventory["demos"]))

    def test_new_sample_and_missing_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "agent-samples/new-sample"
            sample.mkdir(parents=True)
            metadata = ('[project]\nname="new-sample"\ndependencies=["xr-ai-launcher"]\n'
                        '[project.scripts]\nnew_sample="main:run"\n')
            (sample / "pyproject.toml").write_text(metadata)
            inventory = catalog.catalog(root)
            self.assertEqual(inventory["demos"][0]["command"], "new_sample")
            self.assertEqual(inventory["demos"][0]["blockers"], ["No matching GPU profile"])
            profiles = sample / "model-servers/yaml"
            profiles.mkdir(parents=True)
            (profiles / "spark.yaml").touch()
            self.assertEqual(catalog.catalog(root)["recommended"], "new-sample")
            (sample / "pyproject.toml").write_text(metadata + 'another="main:run"\n')
            with self.assertRaisesRegex(ValueError, "exactly one sample console command"):
                catalog.catalog(root)

    def test_resolve_auto_unknown_and_blocked(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            catalog.main(ROOT, ["--resolve", "auto"])
        self.assertEqual(output.getvalue(), "simple-vlm\tagent-samples/simple-vlm-example\tsimple_vlm_example\n")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            catalog.main(ROOT, ["--resolve", "missing"])
        self.assertEqual(error.exception.code, 2)
        self.encoder = "N/A"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            catalog.main(ROOT, ["--resolve", "auto"])
        self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
