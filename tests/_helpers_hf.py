# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face cache fixtures for local-first model resolution tests."""
from __future__ import annotations

import socket
from pathlib import Path

import pytest

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def use_hub_cache(
    monkeypatch: pytest.MonkeyPatch, cache: Path, *, offline: bool = False
) -> Path:
    """Point ``huggingface_hub`` at *cache* and set its offline mode."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    return cache


def cache_file(
    cache: Path,
    repo_id: str,
    filename: str,
    content: str = "",
    *,
    revision: str | None = "main",
    commit: str = COMMIT,
) -> Path:
    """Write one file into the Hub cache layout and return its snapshot path.

    *revision* ``None`` records no ref, as for a file pinned by commit hash.
    """
    repo = cache / f"models--{repo_id.replace('/', '--')}"
    if revision is not None:
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / revision).write_text(commit)
    path = repo / "snapshots" / commit / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def block_network(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Fail every socket connection and return the attempted ones."""
    attempts: list[tuple] = []

    def refuse(*args, **_kwargs):
        attempts.append(args)
        raise OSError("network access is disabled in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    return attempts
