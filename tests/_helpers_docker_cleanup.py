# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session-scoped container cleanup for ``gpu`` tests.

Docker container names are global to the machine, so a name-prefix purge also
kills model servers that a sample launcher started in persist mode. Ownership
is therefore decided by a label, not by name: the test session exports
``XR_AI_TEST_SESSION`` and ``xr_ai_vllm`` stamps ``xr-ai-test-session=<id>`` on
every container it creates under that env. Only containers carrying this
session's id are removed; anything else is left strictly alone.
"""
from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from xr_ai_vllm._docker import TEST_SESSION_LABEL

_MODEL_SERVER_NAME_FILTERS = ("name=^xr-ai-vllm-", "name=^xr-ai-nim-")


def _docker(*args: str, timeout: float) -> str | None:
    """Run a docker command; ``None`` when docker is missing, hung or failing."""
    try:
        proc = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def purge_session_containers(session_id: str) -> None:
    """Force-remove containers labelled with *session_id* (best-effort).

    Silently no-ops without a usable docker so non-GPU boxes don't break.
    """
    out = _docker(
        "ps", "-aq", "--filter", f"label={TEST_SESSION_LABEL}={session_id}",
        timeout=10,
    )
    ids = (out or "").split()
    if not ids:
        return
    try:
        subprocess.run(
            ["docker", "rm", "-f", *ids],
            timeout=20, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def running_foreign_model_servers(session_id: str) -> list[str]:
    """Names of running ``xr-ai-vllm-*`` / ``xr-ai-nim-*`` containers not
    created by this session (e.g. persist-mode servers in live use)."""
    fmt = '{{.Names}}\t{{.Label "' + TEST_SESSION_LABEL + '"}}'
    names: list[str] = []
    for name_filter in _MODEL_SERVER_NAME_FILTERS:
        out = _docker("ps", "--filter", name_filter, "--format", fmt, timeout=10)
        for line in (out or "").splitlines():
            name, _, label = line.partition("\t")
            if name and label.strip() != session_id and name not in names:
                names.append(name)
    return names


@contextmanager
def scoped_gpu_test_containers(session_id: str) -> Iterator[None]:
    """Around one ``gpu`` test: purge this session's containers before and
    after; skip (never kill) if a foreign model server is running."""
    purge_session_containers(session_id)
    foreign = running_foreign_model_servers(session_id)
    if foreign:
        pytest.skip(
            "pre-existing model-server container(s) running, not removing: "
            + ", ".join(foreign)
        )
    try:
        yield
    finally:
        purge_session_containers(session_id)
