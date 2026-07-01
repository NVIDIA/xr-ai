# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr-render-demo orchestrator. Runs the process stack for this sample.

Architecture (per AGENTS.md + the Agentic AI for XR design doc):

  Web client ── LiveKit ──► xr-media-hub ──IPC──► worker (this sample's agent)
  Web client ── WebRTC ──► cloudxr-runtime
                                            worker ──ZMQ──► render-mcp ──► LOVR (OpenXR)

The worker consumes audio from the hub, computes a sphere radius from voice
loudness, and pushes a render command to render-mcp. render-mcp owns the LOVR
child process and forwards render commands to it. CloudXR runs alongside as
its own stream — neither stack passes through the other.

Prerequisites
-------------
The four AI inference servers must be running before this demo starts:

    uv run --project agent-samples/model-servers model_servers

How to run (from the repo root or any directory):
    uv run --project agent-samples/xr-render-demo xr_render_demo

On first run the orchestrator auto-downloads LOVR v0.18.0 to deps/lovr/ inside
the repo and, for WebRTC device profiles, builds the web vendor bundle
(requires npm + network). Native CloudXR profiles never load the web page, so
the vendor build is skipped and the hub serves only its signaling endpoints.
Both steps are skipped once their outputs exist.

To use a custom LOVR build instead of the auto-downloaded one:
    export LOVR_BIN=/path/to/your/lovr      # or set lovr_bin: in render_mcp.yaml

Then open https://<host>:8080, click "Start Mic", click "Launch XR" (or the
WebXR DevUI on desktop). Speak; the sphere tracks your voice in the headset.

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
    ensure_credentials,
    is_native_profile,
    read_device_profile,
    run_stack,
)
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_WORKER_CONFIG = "yaml/xr_render_demo_worker.yaml"
_CLOUDXR_CONFIG = "yaml/cloudxr_runtime.yaml"

# Read the model_backend scalar from the worker YAML without pyyaml — the
# orchestrator is stdlib-only (mirrors the lovr_bin regex read below).
_BACKEND_RE = re.compile(r"^\s*model_backend\s*:\s*[\"']?(\w+)[\"']?", re.MULTILINE)

# Must match _config_loader.NO_WEB_CLIENT_ENV.
_NO_WEB_CLIENT_ENV = "XR_MEDIA_HUB_NO_WEB_CLIENT"


def _model_backend() -> str:
    try:
        m = _BACKEND_RE.search((_BASE / _WORKER_CONFIG).read_text())
    except OSError:
        return "local"
    return m.group(1).lower() if m else "local"


# ── Process stack ─────────────────────────────────────────────────────────────
#
# Model servers are launch_mode="reuse" — they are started and owned by
# model-servers, not this demo.  The entries document the dependency and
# the launcher skips spawning them; start them first with:
#   uv run --project agent-samples/model-servers model_servers
#
# With model_backend: nim (in xr_render_demo_worker.yaml) the worker loads
# models.nim.yaml and vlm-mcp is pointed at vlm_mcp_server.nim.yaml here
# automatically — run LLM/VLM on hosted NIM and just don't start the local
# llm / agent-llm / vlm model-servers. STT/TTS stay local. See
# docs/ai-services.md "Hosting models on NVIDIA NIM".
def _build_processes(backend: str) -> list[Process]:
    # The worker reaches the VLM through vlm-mcp, so vlm-mcp must use the same
    # backend as the worker's models config.
    vlm_mcp_config = (
        "yaml/vlm_mcp_server.nim.yaml" if backend == "nim"
        else "yaml/vlm_mcp_server.yaml"
    )
    return [
        Process("stt",       "../../ai-services/stt-server",         "stt_server",
                launch_mode="reuse"),
        Process("agent-llm", "../../ai-services/llm/nemotron3_nano", "nemotron3_nano_llm_server",
                launch_mode="reuse"),
        Process("vlm",       "../../ai-services/vlm-server",         "vlm_server",
                launch_mode="reuse"),
        Process("llm",       "../../ai-services/llm/llama_nemotron",  "llama_nemotron_llm_server",
                launch_mode="reuse"),
        Process("hub",        "../../server-runtime",                "xr_media_hub",
                config="yaml/xr_media_hub.yaml"),
        Process("cloudxr",    "../../cloudxr-runtime",               "cloudxr_runtime",
                config="yaml/cloudxr_runtime.yaml"),
        Process("tts",        "../../ai-services/tts/piper",         "piper_tts_server",
                config="yaml/piper_tts_server.yaml"),
        Process("vlm-mcp",    "../../agent-mcp-servers/vlm-mcp",     "vlm_mcp_server",
                config=vlm_mcp_config),
        Process("video-mcp",  "../../agent-mcp-servers/video-mcp",   "video_mcp_server",
                config="yaml/video_mcp_server.yaml"),
        Process("render-mcp", "../../agent-mcp-servers/render-mcp",  "render_mcp"),
        Process("oxr-mcp",    "../../agent-mcp-servers/oxr-mcp",     "oxr_mcp_server",
                config="yaml/oxr_mcp_server.yaml",
                quiet_native_output=True),
        Process("vec-mcp",    "../../agent-mcp-servers/vec-mcp",     "vec_mcp_server"),
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
      2. lovr_bin: in render_mcp.yaml (render-mcp reads it directly — we just skip)
      3. Cached AppImage under deps/lovr/ inside the repo
      4. Auto-download from GitHub releases into the cache, then chmod +x
    """
    if os.environ.get("LOVR_BIN"):
        return

    yaml_path = (_BASE / "../../agent-mcp-servers/render-mcp/render_mcp.yaml").resolve()
    if yaml_path.exists():
        for line in yaml_path.read_text().splitlines():
            if _LOVR_BIN_LINE.match(line):
                return  # render-mcp will read lovr_bin directly from its YAML

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
            f"    lovr_bin: /path/to/lovr   (in render_mcp.yaml)\n"
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
    """Build the web vendor bundle (CloudXR + LiveKit ESM) if not already present.

    Runs client-samples/web-xr-build/build.sh, which downloads the CloudXR SDK
    from NGC and produces vendor/cloudxr-sdk.esm.mjs and livekit-client.esm.mjs.
    Requires npm on PATH. Skipped when the output files already exist.
    """
    vendor_dir   = (_BASE / "../../client-samples/web/vendor").resolve()
    cloudxr_out  = vendor_dir / "cloudxr-sdk.esm.mjs"
    if cloudxr_out.exists():
        return

    build_sh = (_BASE / "../../client-samples/web-xr-build/build.sh").resolve()
    if not build_sh.exists():
        logger.warning(
            "web vendor bundle missing and {} not found — skipping", build_sh,
        )
        return

    if not shutil.which("npm"):
        sys.exit(
            "\n  xr-render-demo: web vendor bundle missing and npm is not on PATH.\n"
            "  Install Node.js (https://nodejs.org), then re-run, or build manually:\n"
            f"    cd {build_sh.parent} && ./build.sh\n"
        )

    logger.info("Web vendor bundle not found — running build.sh: {}", build_sh)
    result = subprocess.run([str(build_sh)], cwd=str(build_sh.parent))
    if result.returncode != 0:
        sys.exit(
            f"\n  [setup] build.sh failed (exit {result.returncode}).\n"
            f"  Check the output above, then re-run.\n"
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
    backend = _model_backend()
    if backend == "nim":
        ensure_credentials("NGC_API_KEY")
    run_stack(_build_processes(backend), _BASE)


if __name__ == "__main__":
    run()
