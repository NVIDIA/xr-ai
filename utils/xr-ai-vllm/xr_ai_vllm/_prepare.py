# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact preparation for vLLM and NIM service wrappers."""
from __future__ import annotations

import hashlib
import os
import re
import shlex
import signal
import subprocess
from pathlib import Path

from xr_ai_launcher import (
    ArtifactManifest,
    read_artifact_manifest,
    repair_hf_snapshot,
    report_prepare_status,
    write_artifact_manifest,
)
from xr_ai_launcher import (
    docker_image_size as _image_size,
)

from . import _docker

_HF_DOWNLOAD_CODE = """
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo_id = sys.argv[1]
force_download = sys.argv[2] == "1"
requested_revision = "main"
resolved_revision = None
expected = {}
if force_download:
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    info = api.repo_info(repo_id=repo_id, revision=requested_revision)
    resolved_revision = info.sha
    if not resolved_revision:
        raise RuntimeError(f"Hugging Face returned no revision for {repo_id}")
    entries = api.list_repo_tree(
        repo_id=repo_id,
        revision=resolved_revision,
        recursive=True,
    )
    for entry in entries:
        path = getattr(entry, "path", None)
        size = getattr(entry, "size", None)
        if path is not None and size is not None:
            expected[str(path)] = int(size)
    if not expected:
        raise RuntimeError(
            f"Hugging Face returned no files for {repo_id}@{resolved_revision}"
        )

snapshot = Path(snapshot_download(
    repo_id=repo_id,
    revision=requested_revision,
    force_download=force_download,
))
if force_download and snapshot.name != resolved_revision:
    raise RuntimeError(
        f"Hugging Face repair returned {snapshot.name} instead of "
        f"{resolved_revision} for {repo_id}"
    )
for filename, size in expected.items():
    parts = filename.split("/")
    unsafe = any(part in {"", ".", ".."} for part in parts)
    if not filename or filename.startswith("/") or unsafe:
        raise RuntimeError(
            f"Hugging Face returned an unsafe path for {repo_id}: {filename!r}"
        )
    artifact = snapshot.joinpath(*parts)
    if not artifact.is_file() or artifact.stat().st_size != size:
        raise RuntimeError(
            f"Hugging Face repair did not restore {repo_id}/{filename}"
        )
if force_download:
    ref = snapshot.parent.parent / "refs" / requested_revision
    try:
        cached_revision = ref.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(
            f"Hugging Face repair did not publish "
            f"{repo_id}@{requested_revision}"
        ) from exc
    if cached_revision != resolved_revision:
        raise RuntimeError(
            f"Hugging Face repair left {repo_id}@{requested_revision} at "
            f"{cached_revision!r}, expected {resolved_revision}"
        )
print(snapshot, flush=True)
os.sync()
""".strip()


def _hf_hub_cache(model_cache: Path) -> Path:
    hf_home = os.environ.get("HF_HOME", str(model_cache))
    value = os.environ.get(
        "HF_HUB_CACHE",
        os.environ.get("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub")),
    )
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _snapshot_marker(
    model_cache: Path,
    model: str,
    *,
    backend: str,
    hub_cache: Path,
) -> Path:
    identity = f"vllm\0{backend}\0{hub_cache}\0{model}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return model_cache / ".xr-ai-prepare" / f"hf-vllm-{digest}"


def _write_snapshot_marker(marker: Path, snapshot: Path) -> None:
    write_artifact_manifest(
        marker,
        snapshot,
        (child for child in snapshot.rglob("*") if child.is_file()),
    )


def _snapshot_parent(hub_cache: Path, model: str) -> Path:
    return hub_cache / ("models--" + model.replace("/", "--")) / "snapshots"


def _read_snapshot_marker(
    marker: Path,
    *,
    hub_cache: Path,
    model: str,
) -> ArtifactManifest | None:
    manifest = read_artifact_manifest(marker)
    if manifest is None:
        return None
    if manifest.root.resolve().parent != _snapshot_parent(hub_cache, model).resolve():
        return None
    return manifest


def _snapshot_from_output(stdout: str, *, hub_cache: Path, model: str) -> Path:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            f"Hugging Face model {model} downloaded without reporting its snapshot"
        )
    snapshot = Path(lines[-1]).resolve()
    if not snapshot.is_dir() or snapshot.parent != _snapshot_parent(
        hub_cache, model
    ).resolve():
        raise RuntimeError(
            f"Hugging Face model {model} reported an invalid snapshot path: {snapshot}"
        )
    return snapshot


def _image_id(image: str) -> str:
    try:
        return subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return image


def prepare_image(image: str) -> None:
    """Ensure a Docker image is local."""
    size = _image_size(image)
    if size is not None:
        report_prepare_status(f"container image {image}", "cached", size)
        return
    report_prepare_status(f"container image {image}", "downloading", None)
    _docker._maybe_ngc_login(image)
    try:
        subprocess.run(["docker", "pull", image], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "artifact preparation requires docker on PATH and a running daemon"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to pull container image {image}") from exc


def _docker_hf_argv(
    *,
    image: str,
    model: str,
    model_cache: Path,
    hf_token: str | None,
    container_name: str,
    force_download: bool,
) -> list[str]:
    env_vars = _docker._hf_download_env(model_cache)
    env = [item for key, value in env_vars.items() for item in ("-e", f"{key}={value}")]
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        env += ["-e", "HF_TOKEN"]

    command = shlex.join(
        ["python3", "-c", _HF_DOWNLOAD_CODE, model, "1" if force_download else "0"]
    )
    commands = [*_docker._hf_xet_setup_commands(env_vars), command]

    return [
        "docker", "run", "--rm", "--name", container_name, *env,
        "-v", f"{model_cache}:{model_cache}",
        "--entrypoint", "/bin/bash", image,
        "-c", " && ".join(commands),
    ]


def _download_container_name(kind: str, identity: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]", "-", identity).strip("-.")
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:8]
    return f"xr-ai-prepare-{kind}-{normalized[:32]}-{suffix}-{os.getpid()}"


def _run_download_container(
    argv: list[str],
    container_name: str,
    *,
    capture_stdout: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            stdout=subprocess.PIPE if capture_stdout else None,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "artifact preparation requires docker on PATH and a running daemon"
        ) from exc
    finally:
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            {signal.SIGINT, signal.SIGTERM},
        )
        try:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except OSError:
                pass
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _prepare_docker_snapshot(
    *,
    image: str,
    model: str,
    model_cache: Path,
    hf_token: str | None,
) -> None:
    hub_cache = (model_cache / "hub").resolve()
    marker = _snapshot_marker(
        model_cache,
        model,
        backend="docker",
        hub_cache=hub_cache,
    )
    repair = marker.is_file()
    cached = _read_snapshot_marker(marker, hub_cache=hub_cache, model=model)
    if cached is not None:
        report_prepare_status(f"Hugging Face model {model}", "cached", cached.size)
        return
    report_prepare_status(f"Hugging Face model {model}", "downloading", None)
    container_name = _download_container_name("hf", model)
    result = _run_download_container(
        _docker_hf_argv(
            image=image,
            model=model,
            model_cache=model_cache,
            hf_token=hf_token,
            container_name=container_name,
            force_download=repair,
        ),
        container_name,
        capture_stdout=True,
    )
    if result.returncode != 0:
        detail = (
            f"interrupted by signal {-result.returncode}"
            if result.returncode < 0
            else f"exited with status {result.returncode}"
        )
        raise RuntimeError(f"failed to download Hugging Face model {model}: {detail}")
    snapshot = _snapshot_from_output(result.stdout or "", hub_cache=hub_cache, model=model)
    _write_snapshot_marker(marker, snapshot)


def _prepare_pip_snapshot(
    *,
    model: str,
    model_cache: Path,
    hf_token: str | None,
) -> None:
    from huggingface_hub import snapshot_download

    hub_cache = _hf_hub_cache(model_cache)
    marker = _snapshot_marker(
        model_cache,
        model,
        backend="pip",
        hub_cache=hub_cache,
    )
    repair = marker.is_file()
    cached = _read_snapshot_marker(marker, hub_cache=hub_cache, model=model)
    if cached is not None:
        report_prepare_status(f"Hugging Face model {model}", "cached", cached.size)
        return
    report_prepare_status(f"Hugging Face model {model}", "downloading", None)
    if repair:
        snapshot = repair_hf_snapshot(
            model,
            hub_cache,
            token=hf_token,
        )
    else:
        snapshot = Path(
            snapshot_download(
                repo_id=model,
                revision=None,
                token=hf_token,
                cache_dir=hub_cache,
                force_download=False,
            )
        )
    if snapshot.parent.resolve() != _snapshot_parent(hub_cache, model).resolve():
        raise RuntimeError(
            f"Hugging Face model {model} returned an invalid snapshot path: {snapshot}"
        )
    _write_snapshot_marker(marker, snapshot)


def prepare_vllm(
    *,
    backend: str,
    image: str,
    model: str,
    model_cache: Path,
    hf_token: str | None,
) -> None:
    """Prepare the image and model snapshot for one vLLM service."""
    if backend == "docker":
        prepare_image(image)
        _prepare_docker_snapshot(
            image=image,
            model=model,
            model_cache=model_cache,
            hf_token=hf_token,
        )
        return
    if backend == "pip":
        _prepare_pip_snapshot(
            model=model,
            model_cache=model_cache,
            hf_token=hf_token,
        )
        return
    raise ValueError(
        f"unknown vllm_backend: {backend!r} (expected 'pip' or 'docker')"
    )


def _nim_cache_dir(nim_cache: Path, container_name: str) -> Path:
    cache = nim_cache / container_name
    try:
        cache.mkdir(parents=True, exist_ok=True)
        cache.chmod(0o777)
    except PermissionError:
        if not cache.is_dir() or cache.stat().st_mode & 0o777 != 0o777:
            raise RuntimeError(
                f"cannot make NIM cache {cache} world-writable; chmod 777 it "
                "or configure a different nim_cache"
            ) from None
    return cache


def _gpu_identity(cuda_visible_devices: str | None) -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def _write_nim_marker(cache: Path, marker: Path) -> None:
    artifact_root = cache / "ngc"
    artifacts = tuple(
        child
        for child in artifact_root.rglob("*")
        if child.is_file()
        and not {".locks", "refs"}.intersection(child.relative_to(artifact_root).parts)
    )
    if not any(child.stat().st_size > 0 for child in artifacts):
        raise RuntimeError(f"prepared NIM cache {cache} contains no ngc artifacts")
    write_artifact_manifest(
        marker,
        artifact_root,
        artifacts,
    )


def prepare_nim(
    *,
    image: str,
    container_name: str,
    nim_cache: Path,
    cuda_visible_devices: str | None,
    extra_env: dict[str, str] | None,
) -> None:
    """Download the hardware-selected NIM profile without starting its server."""
    ngc_api_key = os.environ.get("NGC_API_KEY", "").strip()
    if not ngc_api_key:
        raise RuntimeError(
            "NIM artifact preparation requires NGC_API_KEY; get one at "
            "https://ngc.nvidia.com/setup/api-key"
        )
    os.environ["NGC_API_KEY"] = ngc_api_key
    prepare_image(image)

    cache = _nim_cache_dir(nim_cache, container_name)
    payload = "\0".join(
        [
            image,
            _image_id(image),
            cuda_visible_devices or "all",
            _gpu_identity(cuda_visible_devices),
        ]
        + [f"{key}={value}" for key, value in sorted((extra_env or {}).items())]
    )
    marker = cache / (
        ".xr-ai-prepare-" + hashlib.sha256(payload.encode()).hexdigest()[:20]
    )
    cached = read_artifact_manifest(marker, expected_root=cache / "ngc")
    if cached is not None and cached.size > 0:
        report_prepare_status(f"NIM model profile {image}", "cached", cached.size)
        return

    report_prepare_status(f"NIM model profile {image}", "downloading", None)
    download_name = _download_container_name("nim", container_name)
    argv = [
        "docker", "run", "--rm", "--name", download_name,
        "--runtime", "nvidia",
        "-e", f"NVIDIA_VISIBLE_DEVICES={cuda_visible_devices or 'all'}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-e", "NGC_API_KEY",
        "-e", "NIM_CACHE_PATH=/opt/nim/.cache",
    ]
    for key, value in (extra_env or {}).items():
        argv += ["-e", f"{key}={value}"]
    argv += [
        "-v", f"{cache}:/opt/nim/.cache",
        "--entrypoint", "/bin/sh", image,
        "-c",
        "if command -v download-to-cache >/dev/null 2>&1; then "
        "exec download-to-cache; "
        "elif command -v nim_download_to_cache >/dev/null 2>&1; then "
        "exec nim_download_to_cache; "
        "else echo 'NIM image has no download-to-cache utility' >&2; exit 127; fi",
    ]
    result = _run_download_container(argv, download_name)
    if result.returncode != 0:
        detail = (
            f"interrupted by signal {-result.returncode}"
            if result.returncode < 0
            else f"exited with status {result.returncode}"
        )
        raise RuntimeError(
            f"NIM image {image} could not prepare its model profile with "
            "download-to-cache; check that the image supports NVIDIA NIM cache "
            f"utilities ({detail})"
        )
    _write_nim_marker(cache, marker)
