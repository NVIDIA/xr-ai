# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gpu-test container fixture removes only containers its session created.

Uses throwaway ``busybox sleep`` containers with unique names and a unique
session id, so it can never match a live model server. Needs a docker daemon
and a local ``busybox`` image; skipped otherwise.
"""
from __future__ import annotations

import subprocess
import uuid

import pytest
from _helpers_docker_cleanup import purge_session_containers, running_foreign_model_servers
from xr_ai_vllm import _docker
from xr_ai_vllm._docker import TEST_SESSION_ENV, TEST_SESSION_LABEL, build_run_argv


def _docker_ok(*args: str) -> bool:
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, timeout=15,
        ).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


needs_busybox = pytest.mark.skipif(
    not _docker_ok("image", "inspect", "busybox"),
    reason="docker daemon with a local busybox image required",
)


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
def fakes():
    """Unique session id + container names; everything created is removed."""
    tag = uuid.uuid4().hex[:8]
    ns = type("Fakes", (), {})()
    ns.session = f"test-{tag}"
    ns.preexisting = f"xr-ai-vllm-fake-preexisting-{tag}"
    ns.owned = f"xr-ai-vllm-fake-owned-{tag}"
    ns.owned_nim = f"xr-ai-nim-fake-owned-{tag}"
    ns.other_session = f"xr-ai-vllm-fake-othersession-{tag}"
    yield ns
    names = [ns.preexisting, ns.owned, ns.owned_nim, ns.other_session]
    subprocess.run(["docker", "rm", "-f", *names], capture_output=True)


@needs_busybox
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


@needs_busybox
def test_foreign_servers_are_reported_but_not_own_containers(fakes):
    _run_fake(fakes.preexisting)
    _run_fake(fakes.owned, fakes.session)

    foreign = running_foreign_model_servers(fakes.session)

    assert fakes.preexisting in foreign
    assert fakes.owned not in foreign


def test_launcher_labels_container_only_under_a_test_session(monkeypatch, tmp_path):
    kwargs = dict(
        image="img", container_name="xr-ai-vllm-x", port=8100,
        model_cache=tmp_path, hf_token=None, cuda_visible_devices=None,
        extra_env=None, extra_pip=None, vllm_argv=["vllm", "serve", "m"],
    )

    def labels(argv):
        return [argv[i + 1] for i, v in enumerate(argv) if v == "--label"]

    monkeypatch.delenv(TEST_SESSION_ENV, raising=False)
    plain = build_run_argv(**kwargs)
    assert not any(label.startswith(TEST_SESSION_LABEL) for label in labels(plain))

    monkeypatch.setenv(TEST_SESSION_ENV, "abc")
    tagged = build_run_argv(**kwargs)
    assert f"{TEST_SESSION_LABEL}=abc" in labels(tagged)
    # Ownership marker must not change the launch fingerprint (no spurious replace).
    fp = lambda argv: next(  # noqa: E731
        label for label in labels(argv) if label.startswith(f"{_docker._CONFIG_LABEL}=")
    )
    assert fp(plain) == fp(tagged)


def test_gpu_test_is_skipped_naming_preexisting_container(monkeypatch):
    """The scope skips (not kills) when a foreign server is running."""
    import _helpers_docker_cleanup as helpers

    purged: list[str] = []
    monkeypatch.setattr(helpers, "running_foreign_model_servers",
                        lambda sid: ["xr-ai-vllm-vlm-server"])
    monkeypatch.setattr(helpers, "purge_session_containers", purged.append)

    with pytest.raises(pytest.skip.Exception, match="xr-ai-vllm-vlm-server"):
        with helpers.scoped_gpu_test_containers("s1"):
            pytest.fail("body must not run")
    assert purged == ["s1"]  # only the pre-test own-session purge ran


def test_scope_purges_session_after_body(monkeypatch):
    import _helpers_docker_cleanup as helpers

    purged: list[str] = []
    monkeypatch.setattr(helpers, "running_foreign_model_servers", lambda sid: [])
    monkeypatch.setattr(helpers, "purge_session_containers", purged.append)
    with pytest.raises(RuntimeError):
        with helpers.scoped_gpu_test_containers("s2"):
            raise RuntimeError("test blew up")
    assert purged == ["s2", "s2"]
