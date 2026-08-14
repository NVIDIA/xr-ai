# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NGC docker container backend for vLLM.

Runs `docker run nvcr.io/nvidia/vllm:<tag> vllm serve …` always in the
foreground, with start_new_session=True so the container escapes the
launcher's process group and survives stack restarts.  The vLLM process
is visible to ss(8) on the host via --network host, so cleanup uses the
same pid_on_port → SIGTERM path as pip mode.

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
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
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
    port: int,
    model_cache: Path,
    hf_token: str | None,
    cuda_visible_devices: str | None,
    extra_env: dict[str, str] | None,
    extra_pip: list[str] | None,
    vllm_argv: list[str],
) -> list[str]:
    """Build the `docker run …` argv that hosts vllm.

    Always foreground (no -d).  The caller spawns this with
    start_new_session=True so the container escapes the launcher's process
    group but remains stoppable via pid_on_port + SIGTERM — the same path
    as pip-mode vLLM.  With --network host the vLLM process is visible to
    ss(8) on the host, so no docker-specific stop logic is needed.
    """
    argv: list[str] = ["docker", "run"]
    argv += ["--name", container_name]
    # Label lets container_on_port find this container by port without the
    # caller needing to know the container name — implementation detail stays
    # inside this module.
    argv += ["--label", f"xr-ai-vllm.port={port}"]
    argv += ["--network", "host"]
    # vLLM workers communicate via /dev/shm; the default 64 MiB tmpfs is too
    # small for the KV cache shards.  --ipc host gives them the host's larger
    # shared memory namespace.
    argv += ["--ipc", "host"]

    # Request GPUs via the nvidia runtime, not the legacy `--gpus` hook.
    # Under nvidia-container-toolkit `mode = "auto"`, `--gpus` is rejected
    # whenever the toolkit auto-detects CDI mode (the prestart hook refuses
    # with "use --runtime=nvidia instead"), so it fails non-deterministically
    # across hosts. The nvidia runtime injects devices itself and works under
    # both legacy and CDI modes — don't depend on a host's mutable toolkit
    # mode. NVIDIA_VISIBLE_DEVICES takes the same comma list as
    # CUDA_VISIBLE_DEVICES, or "all".
    argv += ["--runtime", "nvidia"]
    argv += ["-e", f"NVIDIA_VISIBLE_DEVICES={cuda_visible_devices or 'all'}"]
    # The NGC image sets compute,utility,video, but set it explicitly so a
    # non-NGC image (or one with a narrower default) still gets CUDA compute.
    argv += ["-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]

    env_vars: dict[str, str] = {
        "HF_HOME": str(model_cache),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if hf_token:
        # Name-only passthrough keeps the token off the ps-visible argv;
        # docker reads the value from this process's environment (run()
        # exports it before spawning).
        argv += ["-e", "HF_TOKEN"]
    if extra_env:
        env_vars.update(extra_env)
    for key, val in env_vars.items():
        argv += ["-e", f"{key}={val}"]

    argv += ["-v", f"{model_cache}:{model_cache}"]

    # Some vLLM images default to `vllm serve`; override the entrypoint so
    # setup installs run in a shell before the server starts.
    argv += ["--entrypoint", "/bin/bash", image]
    # Install hf_transfer before starting vLLM — the NGC image doesn't ship it
    # but HF_HUB_ENABLE_HF_TRANSFER=1 will error if it's missing.
    install_cmds = ["pip install -q hf_transfer"]
    if extra_pip:
        # extra_pip is the seam for models whose architecture needs a wheel
        # the NGC image doesn't bundle (e.g. mamba-ssm for Nemotron-Omni's
        # hybrid backbone). Use --no-build-isolation so the source build
        # can see the container's pre-installed torch — mamba-ssm and its
        # causal_conv1d peer both `import torch` from setup.py at config
        # time, and pip's default isolated build env doesn't have it.
        install_cmds.append(
            f"pip install -q --no-build-isolation {shlex.join(extra_pip)}"
        )
    install_cmds.append(shlex.join(vllm_argv))
    argv += ["-c", " && ".join(install_cmds)]
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


def remove_container(name: str) -> bool:
    """``docker rm`` *name* if it exists; return True if the container was removed.

    Called by ``stop_persistent_servers`` after ``stop_container`` so the next
    launch goes through a full ``docker run`` and picks up any YAML config
    changes (``--limit-mm-per-prompt``, ``extra_pip``, etc.). A missing
    docker binary or a stale-but-already-gone container is non-fatal.
    """
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


def _requested_env(argv: list[str]) -> dict[str, str]:
    """KEY=VALUE pairs requested via ``-e`` (name-only forwards excluded)."""
    env: dict[str, str] = {}
    for i, arg in enumerate(argv[:-1]):
        if arg == "-e" and "=" in argv[i + 1]:
            key, _, value = argv[i + 1].partition("=")
            env[key] = value
    return env


def _container_config_matches(name: str, argv: list[str], image: str) -> bool:
    """True iff *name* was created from *image* with every requested KEY=VALUE.

    Fail-open: if the container cannot be inspected, reuse proceeds and the
    health gate decides; the warning makes the unverified reuse visible.
    """
    requested = _requested_env(argv)
    try:
        out = subprocess.run(
            ["docker", "inspect", "--format",
             "{{.Config.Image}}\n{{range .Config.Env}}{{println .}}{{end}}",
             name],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired,
            subprocess.CalledProcessError):
        log.warning(
            "could not inspect container %s; reusing it WITHOUT verifying "
            "its image and configuration match the current YAML", name,
        )
        return True
    lines = out.splitlines()
    if not lines or lines[0] != image:
        return False
    actual = dict(
        line.partition("=")[::2] for line in lines[1:] if "=" in line
    )
    return all(actual.get(key) == value for key, value in requested.items())


def evict_local_listener(port: int, log_prefix: str) -> None:
    """SIGTERM an xr-ai pip-mode server holding *port* (profile switch).

    A non-xr-ai listener is left alone with a hint; the subsequent launch
    fails to bind rather than this helper killing an unrelated process.
    """
    pid, checked, listening = pid_on_port_checked(port)
    if not checked or not listening or pid is None:
        return
    if not (
        is_xr_ai_server_process(pid, "vllm", port)
        or is_xr_ai_server_process(pid, "stt", port)
    ):
        print(
            f"[{log_prefix}] port {port} is held by pid {pid}, which is not "
            f"an xr-ai server; the launch will fail to bind unless it is "
            f"stopped",
            flush=True,
        )
        return
    print(
        f"[{log_prefix}] port {port} is held by xr-ai server pid {pid}; "
        f"stopping it to make way",
        flush=True,
    )
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(40):
            time.sleep(0.5)
            os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


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
    """Return the registry host for *image* if it is fully qualified, else None.

    A registry is only present when the reference contains a `/` AND the first
    segment looks like a host (contains `.` for a hostname or `:` for a port).
    Without the `/` check a bare tagged image like ``"myimage:latest"`` would
    be misread as a registry because of the tag's colon.
    """
    if "/" not in image:
        return None
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


class _LogStreamer:
    """Stream container stdout/stderr to a sibling file (not to the terminal).

    `docker run -d` does not pipe container output back to the parent and
    `--rm` deletes the container on exit, so without this streamer a
    startup failure leaves no trace. The streamer writes directly to a
    file fd so the launcher's stdout forwarder (and the wrapper's loguru
    sinks) stay quiet — the user reads the container log on demand via
    ``tail -f``. ``docker logs -f`` replays from container start, so a
    fast crash is still captured.

    ``docker run`` returns before dockerd registers the container, and a
    ``docker logs -f`` attached too early exits with "No such container"
    and never recovers. A supervisor thread therefore waits for the container
    to exist before attaching (image pulls can hold this off for minutes) and
    re-attaches if the streamer exits while the container is still expected,
    i.e. until :meth:`stop`.
    """

    def __init__(self, container_name: str) -> None:
        self._name    = container_name
        self.log_path = _container_log_path(container_name)
        self._stop_evt  = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._announced = False
        # RFC3339 start point for re-attaches; a plain `docker logs -f`
        # replays from container start, duplicating the file's contents.
        self._since: str | None = None
        self._thread = threading.Thread(
            target=self._supervise, name=f"docker-logs-{container_name}", daemon=True,
        )
        self._thread.start()

    def _attach(self) -> subprocess.Popen | None:
        try:
            out_fd = open(self.log_path, "ab", buffering=0)
        except OSError as exc:
            log.warning("vllm_docker: could not open %s for streaming: %s",
                        self.log_path, exc)
            return None
        argv = ["docker", "logs", "-f", "-t"]
        if self._since:
            argv += ["--since", self._since]
        argv.append(self._name)
        try:
            proc = subprocess.Popen(
                # -t prefixes each line with the daemon-side RFC3339 timestamp
                # so the file is searchable without going through loguru.
                argv,
                stdout=out_fd, stderr=out_fd,
            )
        except OSError as exc:
            log.warning("vllm_docker: could not attach docker logs: %s", exc)
            return None
        finally:
            out_fd.close()  # on success the child holds its own dup'd fd
        return proc

    def _supervise(self) -> None:
        while not self._stop_evt.is_set():
            delay = 0.2
            while not container_exists(self._name):
                if self._stop_evt.wait(delay):
                    return
                delay = min(delay * 2, 5.0)
            proc = self._attach()
            if proc is None:
                return
            self._proc = proc
            if self._stop_evt.is_set():
                self._terminate(proc)
                return
            if not self._announced:
                self._announced = True
                log.info("container logs → %s", self.log_path)
            proc.wait()
            self._proc = None
            # Back-date the cursor: lines the dead follower never delivered
            # would land before "now" and be skipped forever. A re-attach may
            # duplicate up to 30 s of output; losing diagnostics is worse.
            self._since = (
                datetime.now(timezone.utc) - timedelta(seconds=30)
            ).isoformat()
            # docker logs -f also exits when the container stops; pause so a
            # stopped-but-expected container doesn't spin the attach loop.
            self._stop_evt.wait(1.0)

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
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

    def stop(self) -> None:
        self._stop_evt.set()
        proc = self._proc
        if proc is not None:
            self._terminate(proc)
        self._thread.join(timeout=5)


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
    extra_pip: list[str] | None,
    ready_file: Path | None,
) -> None:
    if hf_token:
        # Exported for build_run_argv's name-only passthrough.
        os.environ["HF_TOKEN"] = hf_token
    argv = build_run_argv(
        image=image,
        container_name=container_name,
        port=port,
        model_cache=model_cache,
        hf_token=hf_token,
        cuda_visible_devices=cuda_visible_devices,
        extra_env=extra_env,
        extra_pip=extra_pip,
        vllm_argv=vllm_argv,
    )
    run_container(
        argv=argv,
        image=image,
        container_name=container_name,
        log_prefix=log_prefix,
        port=port,
        health_url=_lifecycle.health_url(host, port),
        launch_banner=(
            f"Launching vLLM (docker)  image={image}  "
            f"container={container_name}  http://{host}:{port}/v1"
        ),
        reuse_banner=f"vLLM already running on port {port} — reusing",
        ready_banner=f"Ready  →  http://localhost:{port}/v1  (docker: {container_name})",
        ready_file=ready_file,
    )


def run_container(
    *,
    argv: list[str],
    image: str,
    container_name: str,
    log_prefix: str,
    port: int,
    health_url: str,
    launch_banner: str,
    reuse_banner: str,
    ready_banner: str,
    ready_file: Path | None,
) -> None:
    """Shared container lifecycle for the vLLM docker backend and NIMs.

    Reuse-if-healthy, foreign-holder eviction, running-container adoption,
    stopped-container recreation, signal cleanup, log streaming, and
    post-mortem capture behave identically for both backends.
    """
    if not _docker_available():
        log.error(
            "docker backend requires docker on PATH and a running daemon "
            "(`docker version` failed). Install Docker Engine and the NVIDIA "
            "Container Toolkit, then retry."
        )
        sys.exit(2)

    # On abort (Ctrl-C during model-servers startup) the launcher passes
    # no_kill=set() and SIGTERMs every wrapper's process group. Without a
    # handler, SIGTERM kills *this* wrapper but leaves the dockerd-managed
    # container running (still pulling the image / downloading weights). The
    # --stop path can't clean that up either: it gates on /health 200 and a
    # mid-download container is never healthy. So the wrapper stops its own
    # container by name (works regardless of health) when it receives a signal.
    # On a clean run no signal arrives and these handlers stay dormant.
    _state: dict[str, object] = {"proc": None, "streamer": None, "handling": False}
    orig_int  = signal.getsignal(signal.SIGINT)
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
        # Modest timeout so this completes inside the launcher's _STOP_TIMEOUT
        # (20s) window before it escalates to SIGKILL. Both helpers work by
        # name and are idempotent regardless of container health.
        stop_container(container_name, timeout_s=10)
        remove_container(container_name)
        sp = _state["streamer"]
        if isinstance(sp, _LogStreamer):
            sp.stop()
        sys.exit(130)

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # A profile switch can leave a different persistent xr-ai server holding
    # this port (e.g. a NIM container where the local vLLM belongs, or vice
    # versa). Evict it by label before any reuse probe: a foreign server can
    # answer the health probe and be silently mistaken for ours.
    holder, checked = container_on_port_checked(port)
    if not checked:
        print(
            f"[{log_prefix}] could not inspect port {port} ownership; if the "
            f"launch fails to bind, stop whatever holds the port",
            flush=True,
        )
    if checked and holder and holder != container_name:
        print(
            f"[{log_prefix}] port {port} is held by container {holder}; "
            f"stopping it to make way for {container_name}",
            flush=True,
        )
        stop_container(holder)
        if not remove_container(holder) and container_running(holder):
            # Falling through would reuse the foreign server under our
            # identity via the health probe.
            log.error("could not evict container %s from port %d", holder, port)
            sys.exit(1)
    if checked and not holder:
        # The same profile switch can leave a pip-mode server on this port;
        # it too can answer the health probe and be mistaken for ours.
        evict_local_listener(port, log_prefix)

    # A running same-name container may have been created under a different
    # configuration (a profile switch that moves GPUs, an edited YAML, or a
    # bumped image pin); its creation-time config is immutable, so recreate
    # on mismatch.
    if container_running(container_name) and not _container_config_matches(
        container_name, argv, image,
    ):
        print(
            f"[{log_prefix}] container {container_name} is running with a "
            f"different configuration; recreating",
            flush=True,
        )
        stop_container(container_name)
        if not remove_container(container_name):
            log.error("could not remove outdated container %s", container_name)
            sys.exit(1)

    # Reuse a container that survived a wrapper restart (weight persistence).
    if _lifecycle.health_ok(health_url):
        print(f"[{log_prefix}] {reuse_banner}", flush=True)
        if ready_file:
            ready_file.touch()
        signal.signal(signal.SIGINT,  orig_int)
        signal.signal(signal.SIGTERM, orig_term)
        _lifecycle.idle_until_stopped(health_url, log_prefix)
        return

    if container_running(container_name):
        # Running but not yet healthy — e.g. started by a wrapper that died,
        # or a NIM mid engine-download. Adopt it instead of a doomed
        # `docker run` (the name is taken).
        print(
            f"[{log_prefix}] container {container_name} already running — "
            f"waiting for readiness",
            flush=True,
        )
        proc = None
    else:
        if container_exists(container_name):
            # A container's command and entrypoint are immutable. Recreate
            # failed containers so launcher fixes and changed service
            # arguments take effect.
            print(
                f"[{log_prefix}] Recreating stopped container {container_name}",
                flush=True,
            )
            if not remove_container(container_name):
                log.error("Could not remove stopped container %s", container_name)
                sys.exit(1)

        _maybe_ngc_login(image)
        print(f"[{log_prefix}] {launch_banner}", flush=True)
        proc = subprocess.Popen(argv, start_new_session=True)
    _state["proc"] = proc

    streamer = _LogStreamer(container_name)
    _state["streamer"] = streamer
    try:
        _lifecycle.wait_until_healthy(
            health_url,
            is_alive=lambda: (
                proc.poll() is None if proc is not None
                else container_running(container_name)
            ),
        )
    except SystemExit:
        # Two ways to land here: (a) wait_until_healthy raised SystemExit(1)
        # because the container died on its own — the post-mortem log is
        # valuable; (b) our signal handler called sys.exit(130) on abort — it
        # already stopped+removed the container, so a "container failed"
        # post-mortem on a now-removed container is misleading and wasteful.
        # Skip the post-mortem only in the handler case.
        if _state["handling"]:
            raise
        time.sleep(0.5)
        streamer.stop()
        _append_post_mortem(container_name, streamer.log_path)
        log.error("container %s failed — see %s", container_name, streamer.log_path)
        raise

    print(f"[{log_prefix}] {ready_banner}", flush=True)
    if ready_file:
        ready_file.touch()

    # Past readiness the abort-cleanup handler is no longer needed: a persist
    # container that reached ready is meant to outlive the launcher, and the
    # --stop path (health-gated) can now reach it. Restore the original
    # handlers so steady-state signal behavior is unchanged.
    signal.signal(signal.SIGINT,  orig_int)
    signal.signal(signal.SIGTERM, orig_term)

    try:
        _lifecycle.idle_until_stopped(health_url, log_prefix)
    finally:
        streamer.stop()


# ── port → container / pid (used by the stop helper) ────────────────────────

_CONTAINER_PREFIX = "xr-ai-vllm-"


def container_on_port_checked(port: int) -> tuple[str | None, bool]:
    """Return a labelled container and whether Docker discovery succeeded."""
    try:
        out = subprocess.check_output(
            ["docker", "ps",
             f"--filter=label=xr-ai-vllm.port={port}",
             "--format", "{{.Names}}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        names = out.splitlines()
        return (names[0] if names else None), True
    except FileNotFoundError:
        # Without the Docker CLI, a local Docker container cannot be managed;
        # pip-mode ownership is still established from the listener process.
        return None, True
    except subprocess.CalledProcessError:
        return None, False


def container_on_port(port: int) -> str | None:
    """Return the name of a running xr-ai-vllm container serving *port*, or None.

    ``docker ps --filter publish=<port>`` silently misses ``--network host``
    containers.  We label each container with ``xr-ai-vllm.port=<port>`` at
    run time and filter by that label here instead.
    """
    container, _ = container_on_port_checked(port)
    return container


def pid_on_port_checked(port: int) -> tuple[int | None, bool, bool]:
    """Return the listening PID, inspection status, and listener presence.

    Tries `ss` first (always present on modern Linux), falls back to `lsof`.
    A listener without a visible PID is still reported so callers fail closed
    instead of mistaking an uninspectable listener for an unused port.
    """
    try:
        out = subprocess.check_output(
            ["ss", "-tlnpH", f"sport = :{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        m = re.search(r"pid=(\d+)", out)
        if m:
            return int(m.group(1)), True, True
        return None, True, bool(out.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return int(out.splitlines()[0]), True, True
        return None, True, False
    except subprocess.CalledProcessError:
        return None, True, False
    except FileNotFoundError:
        return None, False, False


def pid_on_port(port: int) -> int | None:
    """Return the pid listening on *port* (any v4/v6 socket), or None."""
    pid, _, _ = pid_on_port_checked(port)
    return pid


def is_xr_ai_server_process(pid: int, label: str, port: int) -> bool:
    """Return whether *pid* has the expected xr-ai server command line."""
    try:
        command = Path(f"/proc/{pid}/cmdline").read_text(errors="replace")
    except OSError:
        return False
    if label == "stt":
        return "stt_server" in command
    try:
        environment = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return False
    return (
        b"XR_AI_VLLM_MANAGED=1\0" in environment
        and f"XR_AI_VLLM_PORT={port}\0".encode() in environment
    )
