# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a launcher-owned NVIDIA Magpie TTS NIM container."""

from __future__ import annotations

import argparse
import os
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from loguru import logger
from xr_ai_logging import setup_logging

_OWNERSHIP_LABEL = "com.nvidia.xr-ai.service=magpie-tts-nim"


def _resolve_cache(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _health_ready(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/v1/health/ready", timeout=2
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _docker(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=check,
    )


def _container_is_owned(name: str) -> bool:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            '{{index .Config.Labels "com.nvidia.xr-ai.service"}}',
            name,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "magpie-tts-nim"


def _stop_owned_container(name: str) -> None:
    if not _container_is_owned(name):
        return
    _docker("stop", "--time", "10", name)
    _docker("rm", "-f", name)


def _login_ngc(api_key: str) -> None:
    result = subprocess.run(
        ["docker", "login", "nvcr.io", "-u", "$oauthtoken", "--password-stdin"],
        input=api_key.encode(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"docker login nvcr.io failed: {detail}")


def _export_is_ready(export: Path) -> bool:
    for archive in export.glob("*.tar.gz"):
        try:
            with tarfile.open(archive, "r:*") as model_store:
                names = model_store.getnames()
        except (OSError, tarfile.TarError):
            continue
        if (
            any(name.endswith(".plan") for name in names)
            and any(name.endswith("config.pbtxt") for name in names)
        ):
            return True
    return False


def _prepare_mount(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    # Speech NIM runs as uid 1000, which commonly differs from the host uid.
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077 != 0o077:
        path.chmod(mode | 0o077)


def _build_docker_argv(
    config: dict,
    cache: Path,
    export: Path,
    *,
    export_only: bool = False,
) -> list[str]:
    port = int(config.get("port", 9000))
    host = str(config.get("host", "127.0.0.1"))
    argv = [
        "docker",
        "run",
        "--rm",
        "--name",
        str(config.get("container_name", "xr-ai-magpie-tts-nim")),
        "--label",
        _OWNERSHIP_LABEL,
        "--runtime",
        "nvidia",
        "--gpus",
        f"device={config.get('gpu_device', '0')}",
        "--shm-size",
        str(config.get("shm_size", "8g")),
        "-e",
        "NGC_API_KEY",
        "-e",
        f"NIM_HTTP_API_PORT={port}",
        "-e",
        "NIM_TAGS_SELECTOR="
        + str(
            config.get(
                "profile",
                "name=magpie-tts-multilingual,batch_size=8",
            )
        ),
        "-p",
        f"{host}:{port}:{port}",
        "-v",
        f"{cache}:/opt/nim/.cache",
        "-v",
        f"{export}:/opt/nim/export",
        "-e",
        "NIM_EXPORT_PATH=/opt/nim/export",
    ]
    if not export_only:
        argv.extend(["-e", "NIM_DISABLE_MODEL_DOWNLOAD=true"])
    argv.append(
        str(
            config.get(
                "image",
                "nvcr.io/nim/nvidia/magpie-tts-multilingual:1.9.0",
            )
        )
    )
    return argv


def _serve(config: dict, config_path: Path, ready_file: Path | None) -> None:
    api_key = os.environ.get("NGC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("NGC_API_KEY is required to launch Magpie TTS NIM")
    try:
        _docker("version", check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Docker Engine and NVIDIA Container Toolkit are required for Magpie TTS NIM"
        ) from exc

    name = str(config.get("container_name", "xr-ai-magpie-tts-nim"))
    host = str(config.get("host", "127.0.0.1"))
    port = int(config.get("port", 9000))
    cache = _resolve_cache(
        config_path,
        str(config.get("model_cache", "../../models/nim-magpie-tts")),
    )
    export = _resolve_cache(
        config_path,
        str(config.get("model_export", cache / "export")),
    )
    _prepare_mount(cache)
    _prepare_mount(export)

    stopping = False

    def stop(_sig=None, _frame=None) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        logger.info("Stopping Magpie TTS NIM container {}", name)
        _stop_owned_container(name)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    if _container_is_owned(name):
        if not _health_ready(host, port):
            _stop_owned_container(name)
        else:
            logger.info("Reusing launcher-owned Magpie TTS NIM container {}", name)
            if ready_file:
                ready_file.touch()
            while not stopping and _health_ready(host, port):
                time.sleep(2)
            return

    _login_ngc(api_key)
    if not _export_is_ready(export):
        argv = _build_docker_argv(config, cache, export, export_only=True)
        logger.info("Building and exporting Magpie TTS NIM model store to {}", export)
        process = subprocess.Popen(argv)
        while not stopping:
            return_code = process.poll()
            if return_code is not None:
                break
            time.sleep(2)
        if stopping:
            return
        if return_code != 0 or not _export_is_ready(export):
            raise RuntimeError(f"Magpie TTS NIM export failed (code {return_code})")
        logger.info("Magpie TTS NIM model export ready")

    argv = _build_docker_argv(config, cache, export)
    logger.info("Launching Magpie TTS NIM {} on http://{}:{}", argv[-1], host, port)
    process = subprocess.Popen(argv)
    try:
        while not stopping and not _health_ready(host, port):
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"Magpie TTS NIM exited before ready (code {return_code})"
                )
            time.sleep(2)
        if stopping:
            return
        if ready_file:
            ready_file.touch()
        logger.info("Magpie TTS NIM ready")
        return_code = process.wait()
        if return_code and not stopping:
            raise RuntimeError(f"Magpie TTS NIM exited (code {return_code})")
    finally:
        if process.poll() is None:
            stop()


def run() -> None:
    parser = argparse.ArgumentParser(description="Run NVIDIA Magpie TTS NIM")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, default=None)
    args = parser.parse_args()

    setup_logging("tts-magpie-nim")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Magpie TTS NIM config must be a mapping")
    try:
        _serve(config, args.config.resolve(), args.ready_file)
    except Exception:
        logger.exception("Magpie TTS NIM failed")
        sys.exit(1)


if __name__ == "__main__":
    run()
