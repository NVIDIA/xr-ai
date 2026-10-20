# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gpu-test container fixture removes only containers its session created.

Uses throwaway ``busybox sleep`` containers with unique names and a unique
session id, so it can never match a live model server. Needs a docker daemon
and a local ``busybox`` image; skipped otherwise.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import _helpers_docker_cleanup as helpers
import conftest
import pytest
from _helpers_docker_cleanup import foreign_model_servers, purge_session_containers
from xr_ai_vllm._docker import TEST_SESSION_ENV, TEST_SESSION_LABEL


def _docker_ok(*args: str) -> bool:
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, timeout=15,
        ).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def needs_busybox():
    if not _docker_ok("image", "inspect", "busybox"):
        pytest.skip("docker daemon with a local busybox image required")


def _run_fake(name: str, label: str | None = None) -> None:
    argv = ["docker", "run", "-d", "--name", name]
    if label:
        argv += ["--label", f"{TEST_SESSION_LABEL}={label}"]
    subprocess.run([*argv, "busybox", "sleep", "300"], check=True, capture_output=True)


def _exists(name: str) -> bool:
    out = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name=^{name}$"],
        capture_output=True, text=True, check=True,
    ).stdout
    return bool(out.split())


@pytest.fixture
def fakes(needs_busybox):
    """Unique session id + container names; everything created is removed."""
    tag = uuid.uuid4().hex[:8]
    ns = SimpleNamespace()
    ns.session = f"test-{tag}"
    ns.preexisting = f"xr-ai-vllm-fake-preexisting-{tag}"
    ns.owned = f"xr-ai-vllm-fake-owned-{tag}"
    ns.owned_nim = f"xr-ai-nim-fake-owned-{tag}"
    ns.other_session = f"xr-ai-vllm-fake-othersession-{tag}"
    yield ns
    names = [ns.preexisting, ns.owned, ns.owned_nim, ns.other_session]
    subprocess.run(["docker", "rm", "-f", *names], capture_output=True)


@pytest.mark.gpu
def test_preexisting_container_survives_and_session_container_is_removed(fakes):
    _run_fake(fakes.preexisting)                       # persist-mode server stand-in
    _run_fake(fakes.other_session, "someone-elses")    # another session's container
    _run_fake(fakes.owned, fakes.session)
    _run_fake(fakes.owned_nim, fakes.session)

    purge_session_containers(fakes.session)

    assert not _exists(fakes.owned)
    assert not _exists(fakes.owned_nim)
    assert _exists(fakes.preexisting)
    assert _exists(fakes.other_session)


@pytest.mark.gpu
def test_foreign_servers_are_reported_but_not_own_containers(fakes):
    _run_fake(fakes.preexisting)
    _run_fake(fakes.owned, fakes.session)

    foreign = foreign_model_servers(fakes.session)

    assert fakes.preexisting in foreign
    assert fakes.owned not in foreign


def test_gpu_test_is_skipped_naming_preexisting_container(monkeypatch):
    """The scope skips (not kills) when a foreign server is running."""
    purged: list[str] = []
    monkeypatch.setattr(helpers, "foreign_model_servers",
                        lambda sid: ["xr-ai-vllm-vlm-server"])
    monkeypatch.setattr(helpers, "purge_session_containers", purged.append)

    with pytest.raises(pytest.skip.Exception, match="xr-ai-vllm-vlm-server"):
        with helpers.scoped_gpu_test_containers("s1"):
            pytest.fail("body must not run")
    assert purged == ["s1"]  # only the pre-test own-session purge ran


def test_scope_purges_session_after_body(monkeypatch):
    purged: list[str] = []
    monkeypatch.setattr(helpers, "foreign_model_servers", lambda sid: [])
    monkeypatch.setattr(helpers, "purge_session_containers", purged.append)
    with pytest.raises(RuntimeError):
        with helpers.scoped_gpu_test_containers("s2"):
            raise RuntimeError("test blew up")
    assert purged == ["s2", "s2"]


@pytest.mark.parametrize("failure", [
    subprocess.CalledProcessError(1, ["docker", "ps"]),
    subprocess.TimeoutExpired(["docker", "ps"], 10),
    FileNotFoundError("docker"),
])
def test_failed_inventory_never_enters_gpu_test(monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise failure

    monkeypatch.setattr(helpers.subprocess, "run", unavailable)
    with pytest.raises(pytest.skip.Exception, match="ownership.*unavailable"):
        with helpers.scoped_gpu_test_containers("this-session"):
            pytest.fail("unknown inventory must not admit GPU work")


def test_inventory_includes_stopped_foreign_and_excludes_own(monkeypatch):
    calls = []

    def inventory(*args, timeout):
        calls.append(args)
        return "xr-ai-vllm-stopped\tother\nxr-ai-vllm-own\tmine\n"

    monkeypatch.setattr(helpers, "_docker", inventory)
    assert foreign_model_servers("mine") == ["xr-ai-vllm-stopped"]
    assert len(calls) == 2
    assert all(call[:2] == ("ps", "-a") for call in calls)


def test_purge_targets_only_label_filtered_immutable_ids(monkeypatch):
    calls = []

    def docker(*args, timeout):
        calls.append(args)
        return "a" * 64 + "\n" + "b" * 64 if args[0] == "ps" else ""

    monkeypatch.setattr(helpers, "_docker", docker)
    purge_session_containers("mine")
    assert calls == [
        ("ps", "-aq", "--no-trunc", "--filter", f"label={TEST_SESSION_LABEL}=mine"),
        ("rm", "-f", "a" * 64, "b" * 64),
    ]


def test_purge_rejects_empty_session_before_docker(monkeypatch):
    monkeypatch.setattr(helpers, "_docker", lambda *a, **kw: pytest.fail("unsafe inventory"))
    with pytest.raises(ValueError, match="nonempty"):
        purge_session_containers("")


def test_cli_boundary_keeps_foreign_containers_out_of_removal(monkeypatch, tmp_path):
    """Exercise actual subprocess argv without contacting a Docker daemon."""
    calls = tmp_path / "commands.jsonl"
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"with open({str(calls)!r}, 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'ps':\n"
        "    if '--format' in sys.argv:\n"
        "        print('xr-ai-vllm-foreign\\tother')\n"
        "    else:\n"
        "        print('a' * 64)\n",
        encoding="utf-8",
    )
    docker.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    with pytest.raises(pytest.skip.Exception, match="foreign"):
        with helpers.scoped_gpu_test_containers("mine"):
            pytest.fail("foreign inventory must not admit GPU work")
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert commands[-2:] == [
        ["ps", "-aq", "--no-trunc", "--filter", f"label={TEST_SESSION_LABEL}=mine"],
        ["rm", "-f", "a" * 64],
    ]
    assert all("xr-ai-vllm-foreign" not in command for command in commands)


def test_fixture_exports_and_passes_same_fresh_id_then_restores_environment(monkeypatch):
    monkeypatch.setenv(TEST_SESSION_ENV, "inherited-other-session")
    observed = []

    @contextmanager
    def scope(session):
        observed.append((session, os.environ[TEST_SESSION_ENV]))
        yield

    monkeypatch.setattr(conftest, "scoped_gpu_test_containers", scope)
    request = SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda name: True))
    with monkeypatch.context() as fixture_patch:
        fixture = conftest._scoped_container_cleanup_for_gpu_tests.__wrapped__(request, fixture_patch)
        next(fixture)
        assert observed == [(conftest._TEST_SESSION_ID, conftest._TEST_SESSION_ID)]
        assert conftest._TEST_SESSION_ID != "inherited-other-session"
        with pytest.raises(StopIteration):
            next(fixture)
    assert os.environ[TEST_SESSION_ENV] == "inherited-other-session"


@pytest.mark.parametrize("workers", [2, "auto"])
def test_selected_gpu_rejects_xdist_before_fixture_cleanup(workers):
    config = SimpleNamespace(getoption=lambda *a, **kw: workers)
    gpu = SimpleNamespace(get_closest_marker=lambda name: True)
    with pytest.raises(pytest.UsageError, match="serial execution"):
        conftest.pytest_collection_modifyitems(config, [gpu])


def test_cpu_only_xdist_is_unchanged():
    config = SimpleNamespace(getoption=lambda *a, **kw: 2)
    cpu = SimpleNamespace(get_closest_marker=lambda name: None)
    conftest.pytest_collection_modifyitems(config, [cpu])
