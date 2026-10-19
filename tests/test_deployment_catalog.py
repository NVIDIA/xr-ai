# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only inventory tests: no Docker daemon, GPU, or model downloads."""

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("deployment_catalog", ROOT / "deployment/brev/catalog.py")
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


@pytest.fixture
def host(monkeypatch):
    nvml = SimpleNamespace(
        nvmlInit_v2=Mock(return_value=0), nvmlShutdown=Mock(return_value=0),
        nvmlDeviceGetHandleByUUID=Mock(return_value=0), nvmlDeviceGetEncoderCapacity=Mock(return_value=0),
    )
    state = SimpleNamespace(memory="[N/A]", name="NVIDIA GB10", compute="12.1", count=1,
                            driver="580.1", docker="27.0.1", runtimes='{"nvidia":{}}',
                            codec=True, gpu_error=None, calls=[], nvml=nvml)

    def output(argv, **_kwargs):
        state.calls.append(argv)
        query = " ".join(argv)
        if "pci.bus_id" in query:
            if state.gpu_error:
                raise state.gpu_error
            return "\n".join(f"{i}, GPU-{i}, 0000:0{i}:00.0, {state.name}, {state.compute}, "
                             f"{state.memory}, {state.memory}, 0" for i in range(state.count))
        if "query-compute-apps" in query:
            return ""
        if "driver_version" in query:
            return state.driver
        if argv[:2] == ["docker", "version"]:
            return state.docker
        if argv[:2] == ["docker", "info"]:
            return state.runtimes
        assert argv[0] == "git", argv
        return "commit-for-test"

    monkeypatch.setattr(catalog.subprocess, "check_output", output)
    monkeypatch.setattr(catalog.ctypes.util, "find_library", lambda _: "cached.so" if state.codec else None)

    def load(name):
        if name == "libnvidia-ml.so.1":
            return nvml
        if not state.codec:
            raise OSError("codec library unavailable")
        return object()

    monkeypatch.setattr(catalog.ctypes, "CDLL", load)
    monkeypatch.setattr(catalog.sys, "platform", "linux")
    return state


@pytest.mark.parametrize("name,compute,memory,count,profile", [
    ("NVIDIA GB10", "12.1", "[N/A]", 1, "spark"),
    ("RTX PRO 6000 Blackwell", "12.0", "98304", 1, "96G_blackwell"),
    ("NVIDIA L40S", "8.9", "46068", 2, "dual_48G_ada"),
])
def test_metadata_and_profiles(host, name, compute, memory, count, profile):
    host.name, host.compute, host.memory, host.count = name, compute, memory, count
    inventory = catalog.catalog(ROOT)
    assert len(inventory["gpus"]) == count
    expected = {path.parent.parent.relative_to(ROOT).as_posix()
                for path in (ROOT / "agent-samples").glob("*/model-servers/pyproject.toml")}
    assert {item["project"] for item in inventory["demos"]} == expected
    for item in inventory["demos"]:
        project = tomllib.loads((ROOT / item["project"] / "pyproject.toml").read_text())["project"]
        assert item["id"] == project["name"]
        assert item["description"] == project["description"]
        assert item["runnable"]
        assert item["blockers"] == []
        assert (ROOT / item["project"] / "model-servers/yaml" / f"{profile}.yaml").is_file()
    assert sum("pci.bus_id" in " ".join(argv) for argv in host.calls) == 1


@pytest.mark.parametrize("memory", ["N/A", "[N/A]", "Not Supported", "[Not Supported]"])
def test_spark_unavailable_memory(host, memory):
    host.memory = memory
    inventory = catalog.catalog(ROOT)
    assert inventory["gpus"][0]["memory_mib"] is None
    assert all(item["runnable"] for item in inventory["demos"])


@pytest.mark.parametrize("result,expected,message", [
    (0, True, None), (3, False, "at least one NVENC-capable GPU is required"),
    (999, None, "NVENC support could not be verified"),
])
def test_encoder_capability_not_sessions(host, result, expected, message):
    host.nvml.nvmlDeviceGetEncoderCapacity.return_value = result
    inventory = catalog.catalog(ROOT)
    assert inventory["gpus"][0]["nvenc"] is expected
    assert all(item["blockers"] == ([] if message is None else [message]) for item in inventory["demos"])
    assert host.nvml.nvmlDeviceGetEncoderCapacity.call_args.args[1] == 0
    host.nvml.nvmlShutdown.assert_called_once()


def test_nvml_failure_is_unknown_not_unsupported(host):
    host.nvml.nvmlInit_v2.return_value = 999
    assert catalog.catalog(ROOT)["gpus"][0]["nvenc"] is None
    host.nvml.nvmlShutdown.assert_not_called()


@pytest.mark.parametrize("failure", [OSError("NVML unavailable"), AttributeError("entry point unavailable")])
def test_nvml_unavailable_is_reported(host, monkeypatch, failure):
    monkeypatch.setattr(catalog.ctypes, "CDLL", Mock(side_effect=failure))
    assert catalog.encoder_support(catalog.query_gpu_inventory()) == {0: None}


def test_nvml_handle_failure_cleans_up(host):
    host.nvml.nvmlDeviceGetHandleByUUID.return_value = 999
    assert catalog.encoder_support(catalog.query_gpu_inventory()) == {0: None}
    host.nvml.nvmlDeviceGetEncoderCapacity.assert_not_called()
    host.nvml.nvmlShutdown.assert_called_once()


def test_codec_loader_fallback(host, monkeypatch):
    monkeypatch.setattr(catalog.ctypes.util, "find_library", lambda _: None)
    assert all(item["runnable"] for item in catalog.catalog(ROOT)["demos"])
    host.codec = False
    assert all(item["blockers"] == ["NVENC/NVDEC driver libraries are unavailable"]
               for item in catalog.catalog(ROOT)["demos"])


def test_exact_host_blockers(host):
    host.driver, host.docker, host.runtimes, host.codec = "565.1", "23", "{}", False
    for item in catalog.catalog(ROOT)["demos"]:
        assert not item["runnable"]
        assert item["blockers"] == ["NVIDIA driver 580+ is required", "Docker 24+ is required",
                                    "Docker NVIDIA runtime is unavailable",
                                    "NVENC/NVDEC driver libraries are unavailable"]


@pytest.mark.parametrize("error,message", [
    (FileNotFoundError(), "nvidia-smi is unavailable; XR-AI cannot inspect GPU capacity"),
    (subprocess.TimeoutExpired("nvidia-smi", 8), "nvidia-smi GPU inventory timed out"),
    (subprocess.CalledProcessError(1, "nvidia-smi", stderr="driver failed"),
     "nvidia-smi GPU inventory failed: driver failed"),
])
def test_failed_inventory_is_one_cause(host, error, message):
    host.gpu_error = error
    inventory = catalog.catalog(ROOT)
    assert inventory["gpus"] == []
    assert all(item["blockers"] == [message] for item in inventory["demos"])
    assert not any("driver_version" in " ".join(argv) for argv in host.calls)
    host.nvml.nvmlInit_v2.assert_not_called()


def test_non_linux(host, monkeypatch):
    monkeypatch.setattr(catalog.sys, "platform", "darwin")
    inventory = catalog.catalog(ROOT)
    assert inventory["gpus"] == []
    assert all(item["blockers"] == ["Linux is required"] for item in inventory["demos"])
    assert all(argv[0] == "git" for argv in host.calls)


@pytest.mark.parametrize("invalid,message", [
    ('[project]\nname = [', ""),
    ('[project]\ndependencies=["xr-ai-launcher"]\ndescription="Broken"', "project.name"),
    ('[project]\ndependencies=["xr-ai-launcher"]\nname="broken"', "project.description"),
    ('[project]\ndependencies=["xr-ai-launcher"]\nname="broken"\ndescription="Broken"\n'
     '[project.scripts]\none="main:run"\ntwo="main:run"', "exactly one"),
    ('[project]\ndependencies=["xr-ai-launcher"]\nname="broken"\ndescription="Broken"\n'
     '[project.scripts]\n"bad command"="main:run"', "invalid sample console command"),
])
def test_bad_sample_does_not_block_other_resolution(host, tmp_path, capsys, invalid, message):
    sample = tmp_path / "agent-samples/good"
    sample.mkdir(parents=True)
    (sample / "pyproject.toml").write_text('[project]\nname="good"\ndescription="Good sample"\n'
                                         'dependencies=["xr-ai-launcher"]\n[project.scripts]\ngood="main:run"\n')
    profiles = sample / "model-servers/yaml"
    profiles.mkdir(parents=True)
    (profiles / "spark.yaml").touch()
    broken = tmp_path / "agent-samples/broken/pyproject.toml"
    broken.parent.mkdir()
    broken.write_text(invalid)
    before = list(sys.path)
    inventory = catalog.catalog(tmp_path)
    assert sys.path == before
    bad, good = inventory["demos"]
    assert not bad["runnable"] and good["runnable"]
    assert message in bad["blockers"][0]
    catalog.main([str(tmp_path), "--resolve", "good"])
    assert capsys.readouterr().out == "good\tagent-samples/good\tgood\n"
    with pytest.raises(SystemExit) as error:
        catalog.main([str(tmp_path), "--resolve", "broken"])
    assert error.value.code == 1


def test_json_output_and_missing_profile(host, tmp_path, capsys):
    catalog.main([str(ROOT)])
    assert json.loads(capsys.readouterr().out) == catalog.catalog(ROOT)
    sample = tmp_path / "agent-samples/new/pyproject.toml"
    sample.parent.mkdir(parents=True)
    sample.write_text('[project]\nname="new"\ndescription="New sample"\ndependencies=["xr-ai-launcher"]\n'
                      '[project.scripts]\nnew="main:run"\n')
    assert catalog.catalog(tmp_path)["demos"][0]["blockers"] == ["No matching GPU profile: spark"]


@pytest.mark.parametrize("args", [[], [str(ROOT), "--typo"], [str(ROOT), "--resolve"],
                                   [str(ROOT), "--resolve", "auto"]])
def test_cli_rejects_invalid_args(host, args, capsys):
    with pytest.raises(SystemExit) as error:
        catalog.main(args)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
