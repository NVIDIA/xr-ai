# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NGC NeMo container backend for the in-process NeMo servers (stt, magpie TTS).

Runs the FastAPI server *inside* an NGC NeMo image so torch + nemo + cuDNN come
from the container, escaping host cuDNN/LD_LIBRARY_PATH mismatches that abort
the in-venv path at torch import. The image has torch+nemo+cuDNN but NOT our
server package, so the run command:

  * bind-mounts the repo read-only and points PYTHONPATH at the mounted server
    package + xr-ai-logging,
  * pip-installs only the LIGHT deps the server needs that the NeMo image lacks
    (fastapi, uvicorn, hf_transfer, + per-server extras) — never the server's
    own pyproject, which would drag nemo_toolkit/torch and conflict,
  * runs `python -m <module> --_serve --config <mounted>`.

The container runs foreground with --network host so /health is reachable on
the host; the wrapper touches the ready-file once /health returns 200.

NGC auth: if the image is from nvcr.io and NGC_API_KEY is set, this runs
`docker login nvcr.io` once per process. Existing ~/.docker/config.json entries
take priority and are not overwritten.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import _lifecycle

log = logging.getLogger(__name__)

_DOCKER_CONFIG = Path.home() / ".docker" / "config.json"
_LOGIN_DONE: set[str] = set()

# Where xr-ai-logging lives in the repo, used to build the in-container
# PYTHONPATH. Relative to the repo root that gets bind-mounted.
_XR_AI_LOGGING_REL = "utils/xr-ai-logging"


# ── docker run argv builder ──────────────────────────────────────────────────


def build_run_argv(
    *,
    image: str,
    container_name: str,
    port: int,
    repo_root: Path,
    server_pkg_dir: Path,
    server_module: str,
    config_path: Path,
    model_cache: Path,
    nemo_cache_dir: Path | None,
    hf_token: str | None,
    cuda_visible_devices: str | None,
    extra_pip: list[str] | None,
    extra_env: dict[str, str] | None,
) -> list[str]:
    """Build the `docker run …` argv that hosts a NeMo FastAPI server.

    Always foreground (no -d). The caller spawns this with
    start_new_session=True so the container escapes the launcher's process
    group but remains stoppable by name. With --network host the server's
    /health is reachable on the host.

    *repo_root* is bind-mounted read-only; *server_pkg_dir* (the directory
    that CONTAINS the server package) and *_XR_AI_LOGGING_REL* go on
    PYTHONPATH so the server imports from the mount without a pip install of
    its own pyproject.
    """
    argv: list[str] = ["docker", "run"]
    argv += ["--name", container_name]
    # Label lets stop_on_signal / external tooling find this container by port
    # without knowing the name — implementation detail stays in this module.
    argv += ["--label", f"xr-ai-nemo.port={port}"]
    argv += ["--network", "host"]
    argv += ["--ipc", "host"]

    if cuda_visible_devices:
        argv += ["--gpus", f"device={cuda_visible_devices}"]
    else:
        argv += ["--gpus", "all"]

    env_vars: dict[str, str] = {
        "HF_HOME": str(model_cache / "huggingface"),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if nemo_cache_dir is not None:
        env_vars["NEMO_CACHE_DIR"] = str(nemo_cache_dir)
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
    if extra_env:
        env_vars.update(extra_env)
    for key, val in env_vars.items():
        argv += ["-e", f"{key}={val}"]

    # Repo read-only (source is enough); model cache read-write (weight
    # downloads land here). Both mounted same-path so the server's
    # YAML-relative model_cache resolution lands inside the volume.
    argv += ["-v", f"{repo_root}:{repo_root}:ro"]
    argv += ["-v", f"{model_cache}:{model_cache}"]

    argv.append(image)

    # PYTHONPATH points at the mounted server package dir + xr-ai-logging so
    # imports resolve from the mount — no `pip install` of our pyproject.
    pythonpath = ":".join([str(server_pkg_dir), str(repo_root / _XR_AI_LOGGING_REL)])

    # Light deps the NeMo image lacks. loguru is required by xr-ai-logging,
    # which we put on PYTHONPATH (not pip-installed), so it must be here.
    pip_pkgs = ["fastapi", "uvicorn[standard]", "hf_transfer", "loguru", "pyyaml"]
    if extra_pip:
        pip_pkgs += extra_pip

    serve_argv = [
        "python", "-m", server_module,
        # --_serve short-circuits the server's own backend dispatch so the
        # in-container process never re-reads `backend: docker` and spawns
        # another container.
        "--_serve",
        "--config", str(config_path),
    ]

    install_cmds = [
        f"pip install -q {shlex.join(pip_pkgs)}",
        f"PYTHONPATH={shlex.quote(pythonpath)} {shlex.join(serve_argv)}",
    ]
    argv += ["bash", "-c", " && ".join(install_cmds)]
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


def remove_container(name: str) -> bool:
    """``docker rm`` *name* if it exists; return True if it was removed."""
    if not container_exists(name):
        return False
    try:
        subprocess.run(
            ["docker", "rm", name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def stop_container(name: str, timeout_s: int = 20) -> bool:
    """Stop container *name* if it exists; return True if one was stopped."""
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
    """Return the registry host for *image* if fully qualified, else None.

    A registry is only present when the reference contains a `/` AND the first
    segment looks like a host (a `.` for a hostname or `:` for a port).
    Without the `/` check a bare tagged image like ``"myimage:latest"`` would
    be misread as a registry because of the tag's colon.
    """
    if "/" not in image:
        return None
    head = image.split("/", 1)[0]
    return head if "." in head or ":" in head else None


def _already_logged_in(registry: str) -> bool:
    """Best-effort: True if ~/.docker/config.json already has creds for *registry*."""
    try:
        data = json.loads(_DOCKER_CONFIG.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return registry in data.get("auths", {})


def _maybe_ngc_login(image: str) -> None:
    """Run `docker login nvcr.io` if the image needs NGC auth and a key exists.

    Skips silently if (a) image is not from nvcr.io, (b) NGC_API_KEY is unset,
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
    wrapper's own log. Falls back to ``XR_AI_LOG_ROOT`` (or ``/tmp``) when the
    env vars are absent (e.g. running this module outside a stack).
    """
    ns = os.environ.get("XR_AI_LOG_NAMESPACE")
    stamp = os.environ.get("XR_AI_LOG_TIMESTAMP")
    root = Path(os.environ.get("XR_AI_LOG_ROOT", "/tmp"))
    log_dir = root / f"log_{ns}_{stamp}" if ns and stamp else root
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{container_name}.log"


def _start_log_streamer(
    container_name: str,
) -> tuple[subprocess.Popen | None, Path | None]:
    """Stream container stdout/stderr to a sibling file (not the terminal).

    Without this a startup failure inside the container leaves no trace. The
    streamer writes to a file fd so the launcher's stdout forwarder stays
    quiet; the user reads the container log on demand via ``tail -f``.
    """
    log_path = _container_log_path(container_name)
    try:
        out_fd = open(log_path, "ab", buffering=0)
    except OSError as exc:
        log.warning("nemo_docker: could not open %s for streaming: %s", log_path, exc)
        return None, None
    try:
        proc = subprocess.Popen(
            ["docker", "logs", "-f", "-t", container_name],
            stdout=out_fd,
            stderr=out_fd,
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
    """Append `docker logs --tail` to the container log as a fallback."""
    target = log_path or _container_log_path(container_name)
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(n), container_name],
            capture_output=True,
            check=False,
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
    image: str,
    container_name: str,
    log_prefix: str,
    server_module: str,
    repo_root: Path,
    server_pkg_dir: Path,
    config_path: Path,
    model_cache: Path,
    nemo_cache_dir: Path | None,
    host: str,
    port: int,
    hf_token: str | None,
    cuda_visible_devices: str | None,
    extra_pip: list[str] | None,
    extra_env: dict[str, str] | None,
    ready_file: Path | None,
) -> None:
    if not _docker_available():
        log.error(
            "nemo_backend: docker requires docker on PATH and a running daemon "
            "(`docker version` failed). Install Docker Engine and the NVIDIA "
            "Container Toolkit, then retry — or set `backend: pip` in the YAML."
        )
        sys.exit(2)

    health_url = _lifecycle.health_url(host, port)

    # On abort (Ctrl-C during model-servers startup) the launcher SIGTERMs every
    # wrapper. Without a handler that kills *this* wrapper but leaves the
    # dockerd-managed container running (still pulling the image / downloading
    # weights). So the wrapper stops its own container by name on a signal.
    # On a clean run no signal arrives and these handlers stay dormant.
    _state: dict[str, object] = {"proc": None, "streamer": None, "handling": False}
    orig_int = signal.getsignal(signal.SIGINT)
    orig_term = signal.getsignal(signal.SIGTERM)

    def _on_signal(_sig, _frame):
        # Guard FIRST: Python does not block re-entry during the handler's own
        # docker stop, and the user will mash Ctrl-C. A second signal no-ops.
        if _state["handling"]:
            return
        _state["handling"] = True
        print(
            f"[{log_prefix}] signal received — stopping container {container_name}…",
            flush=True,
        )
        cp = _state["proc"]
        if isinstance(cp, subprocess.Popen) and cp.poll() is None:
            cp.terminate()
        # Modest timeout so this completes inside the launcher's stop window
        # before it escalates to SIGKILL. Both helpers work by name.
        stop_container(container_name, timeout_s=10)
        remove_container(container_name)
        sp = _state["streamer"]
        _stop_log_streamer(sp if isinstance(sp, subprocess.Popen) else None)
        sys.exit(130)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Something is already serving this port (a manually-started container, or
    # another live wrapper). We don't own it, so don't tear it down on signal —
    # restore the original handlers and just idle alongside it.
    if _lifecycle.health_ok(health_url):
        print(f"[{log_prefix}] NeMo server already on :{port} — reusing", flush=True)
        if ready_file:
            ready_file.touch()
        signal.signal(signal.SIGINT, orig_int)
        signal.signal(signal.SIGTERM, orig_term)
        _lifecycle.idle_until_stopped(health_url, log_prefix)
        return

    # A stale container would make `docker run --name` fail with a name
    # conflict; clear it so this is a fresh run. stop first so a still-running
    # but unhealthy leftover (mid-load, or wedged from a prior crash) is also
    # removable — `docker rm` alone fails on a running container.
    if container_exists(container_name):
        stop_container(container_name, timeout_s=10)
        remove_container(container_name)

    _maybe_ngc_login(image)
    argv = build_run_argv(
        image=image,
        container_name=container_name,
        port=port,
        repo_root=repo_root,
        server_pkg_dir=server_pkg_dir,
        server_module=server_module,
        config_path=config_path,
        model_cache=model_cache,
        nemo_cache_dir=nemo_cache_dir,
        hf_token=hf_token,
        cuda_visible_devices=cuda_visible_devices,
        extra_pip=extra_pip,
        extra_env=extra_env,
    )
    print(
        f"[{log_prefix}] Launching NeMo server (docker)  image={image}  "
        f"container={container_name}  http://{host}:{port}/v1",
        flush=True,
    )
    proc = subprocess.Popen(argv, start_new_session=True)
    _state["proc"] = proc

    streamer_proc, log_path = _start_log_streamer(container_name)
    _state["streamer"] = streamer_proc
    try:
        _lifecycle.wait_until_healthy(
            health_url,
            is_alive=lambda: proc.poll() is None,
        )
    except SystemExit:
        # Two ways here: (a) the container died on its own — the post-mortem is
        # valuable; (b) our signal handler called sys.exit(130) on abort — it
        # already stopped+removed the container, so a post-mortem on a removed
        # container is misleading. Skip it only in the handler case.
        if _state["handling"]:
            raise
        time.sleep(0.5)
        _append_post_mortem(container_name, log_path)
        _stop_log_streamer(streamer_proc)
        if log_path is not None:
            log.error("NeMo container failed — see %s", log_path)
        raise

    log.info("Ready  →  http://localhost:%d/v1  (docker: %s)", port, container_name)
    if ready_file:
        ready_file.touch()

    # Keep the stop+remove signal handler installed through the idle loop: the
    # NeMo container is meant to DIE WITH THE STACK (unlike vLLM, which persists
    # to dodge multi-minute reloads — NeMo reloads in ~20s from the cached
    # weights on the mounted volume). The launcher's `--stop` path can't reap
    # this container by label (that path lives in xr_ai_vllm and filters on
    # xr-ai-vllm.port=…), so the wrapper must clean up its own container on the
    # shutdown SIGTERM rather than orphan it. _on_signal stops+removes and
    # sys.exit(130)s; this loop just waits for the container to go away.
    try:
        while _lifecycle.health_ok(health_url, timeout=2.0):
            time.sleep(5.0)
    finally:
        # Reached only if the container vanished on its own (the SIGTERM/SIGINT
        # path exits via the handler). Clean up the now-dead container so a
        # later run goes through a fresh `docker run`.
        signal.signal(signal.SIGINT, orig_int)
        signal.signal(signal.SIGTERM, orig_term)
        stop_container(container_name, timeout_s=10)
        remove_container(container_name)
        _stop_log_streamer(streamer_proc)
