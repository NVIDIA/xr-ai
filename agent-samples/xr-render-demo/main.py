# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr-render-demo orchestrator. Runs the process stack for this sample.

Architecture (per AGENTS.md + the Agentic AI for XR design doc):

  Web client ── LiveKit ──► DeviceIOHub ──IPC──► worker (this sample's agent)
  Web client ── WebRTC ──► cloudxr-runtime
                        worker ──native tool──► scene ──► LOVR (OpenXR)

The worker receives voice queries from the hub and routes them through a
supervisor plus five focused subagents (placement, appearance, object,
vision, memory). Each subagent calls sample-local scene tools to read and
mutate the XR scene. The scene process owns LOVR and scene state. CloudXR
runs alongside as its own stream; neither stack passes through the other.

Prerequisites
-------------
All model services must already be running before this demo starts. The sample
never starts or stops them. The shared model stack includes Piper TTS:

    uv run --project agent-samples/model-servers model_servers

How to run (from the repo root or any directory):
    uv run --project agent-samples/xr-render-demo xr_render_demo

On first run the orchestrator auto-downloads LOVR v0.18.0 to deps/lovr/ inside
the repo and, for WebRTC device profiles, builds the web vendor bundle
(requires npm + network). Native CloudXR profiles never load the web page, so
the vendor build is skipped and the hub serves only its signaling endpoints.
Both steps are skipped once their current-version outputs exist.

To use a custom LOVR build instead of the auto-downloaded one:
    export LOVR_BIN=/path/to/your/lovr      # or set lovr_bin: in scene/scene_service.yaml

Then open https://<host>:8080, click "Start Mic", click "Launch XR" (or the
WebXR DevUI on desktop). Speak a scene command; the agent interprets it and
mutates the XR scene (move, recolor, add, remove, etc.).

The CloudXR EULA is accepted via cloudxr_runtime.yaml (see ``accept_eula``).
"""
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from loguru import logger
from xr_ai_launcher import (
    Process,
    is_native_profile,
    read_device_profile,
    run_stack,
)
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_WORKER_CONFIG = "yaml/xr_render_demo_worker.yaml"
_CLOUDXR_CONFIG = "yaml/cloudxr_runtime.yaml"

# Must match _config_loader.NO_WEB_CLIENT_ENV.
_NO_WEB_CLIENT_ENV = "DEVICE_IO_HUB_NO_WEB_CLIENT"


# ── Process stack ─────────────────────────────────────────────────────────────
#
_MODEL_PROCESSES = [
    Process(
        "stt", "../../services/stt-server", "stt_server",
        launch_mode="reuse",
    ),
    Process(
        "omni", "../../services/nemotron-omni-llm",
        "nemotron_omni_llm_server",
        launch_mode="reuse",
    ),
    Process(
        "vlm", "../../services/vlm-server", "vlm_server",
        launch_mode="reuse",
    ),
    Process(
        "tts", "../../services/piper-tts", "piper_tts_server",
        launch_mode="reuse",
    ),
]


def _build_processes() -> list[Process]:
    return [
        *_MODEL_PROCESSES,
        Process("hub",        "../../services/device-io-hub",                "device_io_hub",
                config="yaml/device_io_hub.yaml"),
        Process("cloudxr",    "../../services/cloudxr-runtime",               "cloudxr_runtime",
                config="yaml/cloudxr_runtime.yaml"),
        Process("video-memory", "../../services/video-memory-service", "video_memory_service",
                config="yaml/video_memory_service.yaml"),
        Process("scene",      "scene",                                "xr_render_scene",
                config="scene/scene_service.yaml"),
        Process("openxr-service", "../../services/openxr-service",  "openxr_service",
                config="yaml/openxr_service.yaml",
                quiet_native_output=True),
        Process("worker",     "worker",                              "xr_render_demo_worker",
                config=_WORKER_CONFIG),
    ]


# Match an uncommented `lovr_bin:` line with a non-empty value.
_LOVR_BIN_LINE = re.compile(r"^\s*lovr_bin\s*:\s*\S")

# ── LOVR auto-download ────────────────────────────────────────────────────────

_LOVR_VERSION  = "0.18.0"
_LOVR_CACHE    = (_BASE / "../../deps/lovr").resolve()
_LOVR_BASE_URL = f"https://github.com/bjornbytes/lovr/releases/download/v{_LOVR_VERSION}"

# (sys.platform, platform.machine().lower()) → release asset filename
_LOVR_ASSETS: dict[tuple[str, str], str] = {
    ("linux",  "x86_64"): f"lovr-v{_LOVR_VERSION}-x86_64.AppImage",
}


def _dl_progress(block_num: int, block_size: int, total_size: int) -> None:
    # Carriage-return progress is intentionally still raw print() — loguru
    # records are line-oriented and would emit a fresh line per update,
    # defeating the in-place spinner.  The "downloading…" log line is
    # emitted via logger before urlretrieve begins, providing the file-log
    # context the spinner doesn't.
    if total_size > 0:
        pct = min(100, block_num * block_size * 100 // total_size)
        print(f"\r  [setup]   {pct}%   ", end="", flush=True)
    else:
        mb = block_num * block_size // (1024 * 1024)
        print(f"\r  [setup]   {mb} MB  ", end="", flush=True)


def _ensure_lovr_bin() -> None:
    """Resolve, download if needed, and expose the LOVR binary via $LOVR_BIN.

    Resolution order:
      1. $LOVR_BIN env var (already set by caller or shell)
      2. lovr_bin: in scene/scene_service.yaml (scene reads it directly)
      3. Cached AppImage under deps/lovr/ inside the repo
      4. Auto-download from GitHub releases into the cache, then chmod +x
    """
    if os.environ.get("LOVR_BIN"):
        return

    yaml_path = (_BASE / "scene/scene_service.yaml").resolve()
    if yaml_path.exists():
        for line in yaml_path.read_text().splitlines():
            if _LOVR_BIN_LINE.match(line):
                return

    key = (sys.platform, platform.machine().lower())
    asset = _LOVR_ASSETS.get(key)
    if asset is None:
        sys.exit(
            f"\n  xr-render-demo: LOVR auto-download is not supported on "
            f"{sys.platform}/{platform.machine()}.\n"
            f"\n"
            f"  Download LOVR v{_LOVR_VERSION} manually from:\n"
            f"    https://github.com/bjornbytes/lovr/releases/tag/v{_LOVR_VERSION}\n"
            f"\n"
            f"  Then set one of:\n"
            f"    export LOVR_BIN=/path/to/lovr\n"
            f"    lovr_bin: /path/to/lovr   (in scene/scene_service.yaml)\n"
        )

    cached = _LOVR_CACHE / asset
    if not cached.exists():
        url = f"{_LOVR_BASE_URL}/{asset}"
        logger.info("LOVR v{} not found — downloading from {}", _LOVR_VERSION, url)
        _LOVR_CACHE.mkdir(parents=True, exist_ok=True)
        partial = cached.with_suffix(cached.suffix + ".partial")
        try:
            urllib.request.urlretrieve(url, partial, _dl_progress)
            print()  # end progress line (paired with _dl_progress's \r updates)
            partial.rename(cached)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            sys.exit(f"\n  [setup] LOVR download failed: {exc}\n")
        cached.chmod(cached.stat().st_mode | 0o111)
        logger.info("LOVR saved to {}", cached)
    else:
        logger.info("Using cached LOVR: {}", cached)

    os.environ["LOVR_BIN"] = str(cached)


# ── Web vendor bundle ─────────────────────────────────────────────────────────

def _ensure_web_vendor() -> None:
    """Build the web vendor bundle when outputs are missing or out of date.

    Runs client-samples/web-xr-build/build.sh, which downloads the CloudXR SDK
    from NGC and produces vendor/cloudxr-sdk.esm.mjs and livekit-client.esm.mjs.
    Requires npm on PATH. Skipped when both outputs carry the version selected
    by web-xr-build/.sdk-version.
    """
    vendor_dir   = (_BASE / "../../client-samples/web-xr/vendor").resolve()
    cloudxr_out  = vendor_dir / "cloudxr-sdk.esm.mjs"
    livekit_out  = vendor_dir / "livekit-client.esm.mjs"
    build_sh = (_BASE / "../../client-samples/web-xr-build/build.sh").resolve()
    if not build_sh.exists():
        logger.warning(
            "web vendor bundle missing or stale and {} not found — skipping",
            build_sh,
        )
        return

    sdk_version_path = build_sh.parent / ".sdk-version"
    version_marker = vendor_dir / ".cloudxr-sdk-version"
    try:
        sdk_version = sdk_version_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        sys.exit(f"\n  [setup] failed to read {sdk_version_path}: {exc}\n")
    if not sdk_version:
        sys.exit(f"\n  [setup] {sdk_version_path} is empty.\n")

    try:
        built_version = version_marker.read_text(encoding="utf-8").strip()
    except OSError:
        built_version = ""
    if (
        cloudxr_out.exists()
        and livekit_out.exists()
        and built_version == sdk_version
    ):
        return

    if not shutil.which("npm"):
        sys.exit(
            "\n  xr-render-demo: web vendor bundle missing or stale and npm is "
            "not on PATH.\n"
            "  Install Node.js (https://nodejs.org), then re-run, or build manually:\n"
            f"    cd {build_sh.parent} && ./build.sh\n"
        )

    logger.info(
        "Web vendor bundle missing or stale (have={!r}, want={!r}) — "
        "running build.sh: {}",
        built_version or None,
        sdk_version,
        build_sh,
    )
    result = subprocess.run([str(build_sh)], cwd=str(build_sh.parent))
    if result.returncode != 0:
        sys.exit(
            f"\n  [setup] build.sh failed (exit {result.returncode}).\n"
            f"  Check the output above, then re-run.\n"
        )
    missing = [path.name for path in (cloudxr_out, livekit_out) if not path.exists()]
    if missing:
        sys.exit(
            "\n  [setup] build.sh completed without producing: "
            f"{', '.join(missing)}.\n"
            "  Check the output above, then re-run.\n"
        )
    try:
        built_version = version_marker.read_text(encoding="utf-8").strip()
    except OSError:
        built_version = ""
    if built_version != sdk_version:
        sys.exit(
            "\n  [setup] build.sh completed without recording CloudXR SDK "
            f"version {sdk_version!r} in {version_marker}.\n"
        )
    logger.info("Web vendor bundle ready")


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    setup_logging("orchestrator", namespace="xr-render-demo")
    if is_native_profile(read_device_profile(_BASE / _CLOUDXR_CONFIG)):
        os.environ[_NO_WEB_CLIENT_ENV] = "1"
        logger.info("native device profile: web client page disabled, skipping vendor build")
    else:
        _ensure_web_vendor()
    _ensure_lovr_bin()
    run_stack(_build_processes(), _BASE)


if __name__ == "__main__":
    run()
