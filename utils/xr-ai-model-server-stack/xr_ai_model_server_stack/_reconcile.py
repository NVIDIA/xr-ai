# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private to xr_ai_model_server_stack: serialized, opt-in model reconciliation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import subprocess
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from xr_ai_vllm import persistent_server_identity, stop_persistent_servers


def _open_private(path: Path):
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise RuntimeError(f"cannot open private model state {path}") from exc
    stream = os.fdopen(fd, "r+")
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        stream.close()
        raise RuntimeError(f"{path} must be an owned private regular file")
    return stream


@contextmanager
def model_state(path: Path):
    path = path.absolute()
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = path.parent.lstat()
    except OSError as exc:
        raise RuntimeError("cannot prepare private model state directory") from exc
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("model state directory must be owned and private")
    with _open_private(path.with_name(path.name + ".lock")) as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another model lifecycle action is in progress") from exc
        with _open_private(path) as stream:
            text = stream.read(65537)
        try:
            records = json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid model state; inspect it before retrying") from exc
        if (
            len(text) > 65536
            or not isinstance(records, dict)
            or any(
                not isinstance(name, str)
                or not isinstance(record, dict)
                or set(record) != {"port", "digest", "identity"}
                or type(record["port"]) is not int
                or not 1 <= record["port"] <= 65535
                or not isinstance(record["digest"], str)
                or record["identity"] is not None
                and not isinstance(record["identity"], str)
                for name, record in records.items()
            )
        ):
            raise RuntimeError("invalid model state; inspect it before retrying")
        yield _State(path, records)


class _State:
    def __init__(self, path: Path, records: dict):
        self.path, self.records = path, records

    def _save(self, records: dict) -> None:
        temporary = self.path.with_name(self.path.name + ".pending")
        try:
            with _open_private(temporary) as stream:
                stream.seek(0)
                json.dump(records, stream)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise RuntimeError("cannot persist model state; stop recorded models before retrying") from exc
        self.records = records

    def _identities(self) -> dict[str, str | None]:
        identities = {}
        for name, record in self.records.items():
            current = persistent_server_identity(name, record["port"])
            if current and record["identity"] and current != record["identity"]:
                raise RuntimeError(f"{name} instance changed; refusing to stop or reuse it")
            identities[name] = current
        return identities

    def stop(self) -> None:
        self._identities()
        if not stop_persistent_servers([(name, record["port"]) for name, record in self.records.items()]):
            raise RuntimeError("one or more recorded model servers could not be stopped")
        self._save({})

    def prepare(self, configs: dict, root: Path) -> None:
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.PIPE,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("cannot identify the model checkout revision") from exc
        desired = {}
        for name, config in configs.items():
            payload = [str(root), revision, config, os.environ.get("HF_TOKEN"), os.environ.get("CUDA_VISIBLE_DEVICES")]
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            desired[name] = dict(port=config["port"], digest=digest, identity=None)
        current = self._identities()
        old_ports = {record["port"] for record in self.records.values()}
        for name, record in desired.items():
            if record["port"] not in old_ports and persistent_server_identity(name, record["port"]):
                raise RuntimeError(f"untracked model on port {record['port']}; stop it before enabling reconciliation")
        stop = []
        for name, record in self.records.items():
            target = desired.get(name)
            if current[name] and record["identity"] is None:
                raise RuntimeError(f"{name} has an unfinished launch; use --stop with this state file first")
            reuse = bool(current[name] and target and target["digest"] == record["digest"])
            if reuse:
                try:
                    with build_opener(ProxyHandler({})).open(
                        f"http://127.0.0.1:{record['port']}/health",
                        timeout=2,
                    ) as response:
                        reuse = response.status == 200
                except (OSError, URLError):
                    reuse = False
            if reuse:
                target["identity"] = current[name]
                print(f"[{name}] keeping compatible model server", flush=True)
            else:
                stop.append((name, record["port"]))
        if not stop_persistent_servers(stop):
            raise RuntimeError("model replacement failed; refusing to launch the next stack")
        # Publish before launch: cancellation or failure still leaves every selected port
        # available to explicit stop, but unverified partial starts are never adopted.
        self._save(desired)

    def ready(self) -> None:
        records = {}
        for name, record in self.records.items():
            identity = persistent_server_identity(name, record["port"])
            if identity is None or record["identity"] and identity != record["identity"]:
                raise RuntimeError(f"{name} instance disappeared or changed during startup")
            records[name] = {**record, "identity": identity}
        self._save(records)
