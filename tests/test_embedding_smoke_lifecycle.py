# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for the embedding smoke test's launch and cleanup."""

import asyncio
from pathlib import Path

import pytest
import test_gpu_embedding_server as smoke
import yaml


async def test_embedding_smoke_skips_unsupported_inventory(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: "/usr/bin/docker")

    def inventory():
        raise smoke.GPUInventoryError("unsupported test topology")

    monkeypatch.setattr(smoke, "detect_gpu_config", inventory)
    monkeypatch.setattr(smoke.asyncio, "create_subprocess_exec", lambda *_args: pytest.fail("launched"))
    with pytest.raises(pytest.skip.Exception, match="unsupported test topology"):
        await smoke.test_tea_embedding_startup_and_retrieval(tmp_path)


@pytest.mark.parametrize("failure", ["exit", "timeout"])
async def test_embedding_smoke_cleans_up_docker_before_readiness(monkeypatch, tmp_path, failure):
    events = []
    launched = {}

    class Process:
        returncode = 1 if failure == "exit" else None

        def terminate(self):
            events.append("terminate")
            self.returncode = -15

        async def wait(self):
            return self.returncode

    async def launch(*args):
        config_path = Path(args[args.index("--config") + 1])
        launched.update(yaml.safe_load(config_path.read_text()))
        return Process()

    async def timeout(_seconds):
        raise TimeoutError("startup timed out")

    def stop(services):
        events.append(services)
        return True

    monkeypatch.setattr(smoke.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(smoke, "detect_gpu_config", lambda: "96G_blackwell")
    monkeypatch.setattr(smoke.asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(smoke.asyncio, "sleep", timeout)
    monkeypatch.setattr(smoke, "stop_persistent_servers", stop)
    with pytest.raises(AssertionError if failure == "exit" else asyncio.TimeoutError):
        await smoke.test_tea_embedding_startup_and_retrieval(tmp_path)
    assert launched["vllm_backend"] == "docker"
    assert launched["gpu_memory_utilization"] == 0.08
    assert events[-1] == [("embedding", launched["port"])]
    assert ("terminate" in events) is (failure == "timeout")
