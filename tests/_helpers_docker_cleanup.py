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


class _DockerInventoryError(RuntimeError):
    """Docker could not establish the ownership needed for safe testing."""


def _docker(*args: str, timeout: float) -> str:
    try:
        proc = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout,
            check=True,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        raise _DockerInventoryError(
            "Docker ownership inventory/cleanup unavailable; not running GPU tests"
        ) from exc
    return proc.stdout


def purge_session_containers(session_id: str) -> None:
    """Force-remove only immutable IDs labelled with this nonempty session."""
    if not session_id:
        raise ValueError("container cleanup requires a nonempty session ID")
    out = _docker(
        "ps", "-aq", "--no-trunc", "--filter", f"label={TEST_SESSION_LABEL}={session_id}",
        timeout=10,
    )
    ids = out.split()
    if not ids:
        return
    _docker("rm", "-f", *ids, timeout=20)


def foreign_model_servers(session_id: str) -> list[str]:
    """Foreign model containers, including stopped ones the launcher could reuse."""
    fmt = '{{.Names}}\t{{.Label "' + TEST_SESSION_LABEL + '"}}'
    names: list[str] = []
    for name_filter in _MODEL_SERVER_NAME_FILTERS:
        out = _docker("ps", "-a", "--filter", name_filter, "--format", fmt, timeout=10)
        for line in out.splitlines():
            name, _, label = line.partition("\t")
            if name and label.strip() != session_id and name not in names:
                names.append(name)
    return names


@contextmanager
def scoped_gpu_test_containers(session_id: str) -> Iterator[None]:
    """Around one ``gpu`` test: purge this session's containers before and
    after; skip (never kill) if ownership is unknown or a foreign server exists."""
    try:
        foreign = foreign_model_servers(session_id)
        purge_session_containers(session_id)
    except _DockerInventoryError as exc:
        pytest.skip(str(exc))
    if foreign:
        pytest.skip(
            "foreign model-server container(s), not removing: "
            + ", ".join(foreign)
            + ". Inspect `docker ps -a --filter label=xr-ai-test-session` and "
            "confirm a labelled container belongs to a dead test run before "
            "manually removing its exact ID with `docker rm -f <container-id>`."
        )
    try:
        yield
    finally:
        purge_session_containers(session_id)
