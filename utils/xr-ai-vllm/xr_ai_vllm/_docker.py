# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NGC docker container backend for vLLM.

Runs `docker run nvcr.io/nvidia/vllm:<tag> vllm serve …` with the same flags
the pip backend would pass to its locally-installed vLLM. Persistent mode
runs detached + named so the container survives wrapper restarts; cleanup is
via :func:`stop_container` (called from `xr_ai_vllm.stop_persistent_servers`).

NGC auth: if the image is from `nvcr.io/` and `NGC_API_KEY` is in the
environment, this module runs `docker login nvcr.io` once per process so the
pull can proceed. Existing `~/.docker/config.json` entries take priority and
are not overwritten.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import _lifecycle

log = logging.getLogger(__name__)

_DOCKER_CONFIG = Path.home() / ".docker" / "config.json"
_LOGIN_DONE: set[str] = set()


# ── docker run argv builder ──────────────────────────────────────────────────


def build_run_argv(
    *,
    image: str,
    container_name: str,
    detached: bool,
    model_cache: Path,
    hf_token: str | None,
    cuda_visible_devices: str | None,
    extra_env: dict[str, str] | None,
    vllm_argv: list[str],
) -> list[str]:
    """Build the `docker run …` argv that hosts vllm.

    `--network host` and `--ipc host` mirror the LiveKit docker precedent
    (`server-runtime/xr_media_hub/transport/livekit/_docker.py`) — both are
    required for vLLM tensor-parallel workers and for binding the serving
    port without docker-proxy.

    The model cache is bind-mounted to the same path inside and outside so
    `HF_HOME` does not need translation between the wrapper YAML and the
    container.
    """
    argv: list[str] = ["docker", "run", "--rm"]
    if detached:
        argv.append("-d")
    argv += ["--name", container_name]
    argv += ["--network", "host"]
    # vLLM workers communicate via /dev/shm; the default 64 MiB tmpfs is too
    # small for the KV cache shards.  --ipc host gives them the host's larger
    # shared memory namespace.
    argv += ["--ipc", "host"]

    if cuda_visible_devices:
        argv += ["--gpus", f"device={cuda_visible_devices}"]
    else:
        argv += ["--gpus", "all"]

    env_vars: dict[str, str] = {
        "HF_HOME": str(model_cache),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
    if extra_env:
        env_vars.update(extra_env)
    for key, val in env_vars.items():
        argv += ["-e", f"{key}={val}"]

    argv += ["-v", f"{model_cache}:{model_cache}"]

    argv.append(image)
    argv += vllm_argv
    return argv


# ── docker container helpers ─────────────────────────────────────────────────


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def container_exists(name: str) -> bool:
    """True if a container named *name* is currently listed by docker (any state)."""
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-aq", "-f", f"name=^{re.escape(name)}$"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return bool(out)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def container_running(name: str) -> bool:
    """True if container *name* is in the running state."""
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-q", "-f", f"name=^{re.escape(name)}$"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return bool(out)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def stop_container(name: str, timeout_s: int = 20) -> bool:
    """Stop container *name* if it exists; return True if a container was stopped."""
    if not container_exists(name):
        return False
    try:
        subprocess.run(
            ["docker", "stop", "-t", str(timeout_s), name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as exc:
        log.warning(
            "docker stop %s failed (rc=%d): %s — escalating to docker kill",
            name,
            exc.returncode,
            (exc.stderr or b"").decode(errors="replace").strip(),
        )
        try:
            subprocess.run(
                ["docker", "kill", name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
    except FileNotFoundError:
        return False


# ── NGC auth ────────────────────────────────────────────────────────────────


def _registry_for(image: str) -> str | None:
    """Return the registry host for *image* if it is fully qualified, else None."""
    head = image.split("/", 1)[0]
    return head if "." in head or ":" in head else None


def _already_logged_in(registry: str) -> bool:
    """Best-effort: True if ~/.docker/config.json already has credentials for *registry*."""
    try:
        data = json.loads(_DOCKER_CONFIG.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return registry in data.get("auths", {})


def _maybe_ngc_login(image: str) -> None:
    """Run `docker login nvcr.io` if the image needs NGC auth and a key is available.

    Skips silently if (a) image is not from nvcr.io, (b) NGC_API_KEY is not set,
    or (c) docker is already authenticated to that registry.
    """
    registry = _registry_for(image)
    if registry != "nvcr.io":
        return
    if registry in _LOGIN_DONE or _already_logged_in(registry):
        _LOGIN_DONE.add(registry)
        return
    token = os.environ.get("NGC_API_KEY", "").strip()
    if not token:
        return
    try:
        result = subprocess.run(
            ["docker", "login", registry, "-u", "$oauthtoken", "--password-stdin"],
            input=token.encode(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return
    if result.returncode == 0:
        _LOGIN_DONE.add(registry)
        log.debug("docker login %s succeeded via NGC_API_KEY", registry)
    else:
        log.warning(
            "docker login %s failed: %s — pull may fail",
            registry,
            (result.stderr or b"").decode(errors="replace").strip(),
        )


# ── log forwarding ──────────────────────────────────────────────────────────


def _container_log_path(container_name: str) -> Path:
    """Sibling log file inside the per-run xr-ai-logging directory.

    Reads ``XR_AI_LOG_NAMESPACE`` / ``XR_AI_LOG_TIMESTAMP`` / ``XR_AI_LOG_ROOT``
    stamped by ``setup_logging`` so the container log lands next to the
    wrapper's own log. Falls back to ``XR_AI_LOG_ROOT`` (or ``/tmp``) when
    the env vars are absent (e.g. running this module outside a stack).
    """
    ns    = os.environ.get("XR_AI_LOG_NAMESPACE")
    stamp = os.environ.get("XR_AI_LOG_TIMESTAMP")
    root  = Path(os.environ.get("XR_AI_LOG_ROOT", "/tmp"))
    if ns and stamp:
        log_dir = root / f"log_{ns}_{stamp}"
    else:
        log_dir = root
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{container_name}.log"


def _start_log_streamer(container_name: str) -> tuple[subprocess.Popen | None, Path | None]:
    """Stream container stdout/stderr to a sibling file (not to the terminal).

    `docker run -d` does not pipe container output back to the parent and
    `--rm` deletes the container on exit, so without this streamer a
    startup failure leaves no trace. The streamer writes directly to a
    file fd so the launcher's stdout forwarder (and the wrapper's loguru
    sinks) stay quiet — the user reads the container log on demand via
    ``tail -f``. ``docker logs -f`` replays from container start, so a
    fast crash is still captured.
    """
    log_path = _container_log_path(container_name)
    try:
        out_fd = open(log_path, "ab", buffering=0)
    except OSError as exc:
        log.warning("vllm_docker: could not open %s for streaming: %s", log_path, exc)
        return None, None
    try:
        proc = subprocess.Popen(
            # -t prefixes each line with the daemon-side RFC3339 timestamp so
            # the file is searchable without going through loguru.
            ["docker", "logs", "-f", "-t", container_name],
            stdout=out_fd, stderr=out_fd,
        )
    except FileNotFoundError:
        out_fd.close()
        return None, None
    out_fd.close()  # the child holds its own dup'd fd
    log.info("container logs → %s", log_path)
    return proc, log_path


def _stop_log_streamer(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def _append_post_mortem(container_name: str, log_path: Path | None, n: int = 200) -> None:
    """Append `docker logs --tail` to the container log file as a fallback.

    Covers the small race where the streamer attaches just after the
    container starts producing output, or where the container exited
    before the streamer's process opened its connection to the daemon.
    Best-effort — silent on any I/O error.
    """
    target = log_path or _container_log_path(container_name)
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(n), container_name],
            capture_output=True, check=False,
        )
    except FileNotFoundError:
        return
    blob = (result.stdout or b"") + (result.stderr or b"")
    if not blob.strip():
        return
    try:
        with open(target, "ab") as f:
            f.write(f"\n---- post-mortem `docker logs --tail={n}` ----\n".encode())
            f.write(blob if blob.endswith(b"\n") else blob + b"\n")
            f.write(b"---- end post-mortem ----\n")
    except OSError:
        return


# ── run flow ────────────────────────────────────────────────────────────────


def run(
    *,
    persistent: bool,
    image: str,
    container_name: str,
    log_prefix: str,
    vllm_argv: list[str],
    host: str,
    port: int,
    model_cache: Path,
    hf_token: str | None,
    cuda_visible_devices: str | None,
    extra_env: dict[str, str] | None,
    ready_file: Path | None,
) -> None:
    if not _docker_available():
        log.error(
            "vllm_backend: docker requires docker on PATH and a running daemon "
            "(`docker version` failed). Install Docker Engine and the NVIDIA "
            "Container Toolkit, then retry."
        )
        sys.exit(2)

    health_url = _lifecycle.health_url(host, port)

    if persistent and _lifecycle.health_ok(health_url):
        print(
            f"[{log_prefix}] vLLM container already healthy on port {port} — reusing",
            flush=True,
        )
        if ready_file:
            ready_file.touch()
        _lifecycle.idle_until_stopped(health_url, log_prefix)
        return

    if container_exists(container_name) and not container_running(container_name):
        # Stale --rm container that exited mid-startup; clean up so docker run
        # below does not collide on the name.
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if not container_running(container_name):
        _maybe_ngc_login(image)
        argv = build_run_argv(
            image=image,
            container_name=container_name,
            detached=persistent,
            model_cache=model_cache,
            hf_token=hf_token,
            cuda_visible_devices=cuda_visible_devices,
            extra_env=extra_env,
            vllm_argv=vllm_argv,
        )
        print(
            f"[{log_prefix}] Launching vLLM (docker)  image={image}  "
            f"container={container_name}  http://{host}:{port}/v1",
            flush=True,
        )
        if persistent:
            # `docker run -d` returns immediately after printing the container id.
            result = subprocess.run(argv, capture_output=True, text=True)
            if result.returncode != 0:
                log.error(
                    "docker run failed (rc=%d): %s",
                    result.returncode,
                    (result.stderr or "").strip(),
                )
                sys.exit(result.returncode)
            proc: subprocess.Popen | None = None
        else:
            proc = subprocess.Popen(argv)
    else:
        proc = None

    streamer_proc, log_path = (
        _start_log_streamer(container_name) if persistent else (None, None)
    )
    try:
        _lifecycle.wait_until_healthy(
            health_url,
            is_alive=lambda: (proc is None or proc.poll() is None)
            and container_running(container_name),
        )
    except SystemExit:
        # Give the streamer a beat to flush, then append a tail explicitly in
        # case it attached late or the container is already gone.
        time.sleep(0.5)
        _append_post_mortem(container_name, log_path)
        _stop_log_streamer(streamer_proc)
        if log_path is not None:
            log.error("vLLM container failed — see %s", log_path)
        raise

    log.info("Ready  →  http://localhost:%d/v1  (docker: %s)", port, container_name)
    if ready_file:
        ready_file.touch()

    try:
        if persistent:
            # Persistence parity with pip mode: wrapper exits cleanly without
            # touching the container; cleanup happens via stop_persistent_servers.
            _lifecycle.idle_until_stopped(health_url, log_prefix)
        else:
            assert proc is not None
            rc = proc.wait()
            # Foreground container is auto-removed by --rm; nothing else to clean up.
            if rc != 0:
                sys.exit(rc)
    finally:
        _stop_log_streamer(streamer_proc)


# ── port → pid (used by the docker-aware stop helper for the pip fallback) ──


def pid_on_port(port: int) -> int | None:
    """Return the pid listening on *port* (any v4/v6 socket), or None.

    Tries `ss` first (always present on modern Linux), falls back to `lsof`.
    Used by the stop helper to send SIGTERM to a pip-mode vLLM after `docker
    stop` reports no matching container.
    """
    try:
        out = subprocess.check_output(
            ["ss", "-tlnpH", f"sport = :{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        m = re.search(r"pid=(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return int(out.splitlines()[0])
    except Exception:
        pass
    return None
