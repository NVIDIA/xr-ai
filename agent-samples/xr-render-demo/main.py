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

How to run (from the repo root or any directory):
    uv run --project agent-samples/xr-render-demo xr_render_demo

On first run the orchestrator auto-downloads LOVR v0.18.0 to deps/lovr/ inside
the repo and builds the web vendor bundle (requires npm + network). Both steps
are skipped once the outputs exist.

To use a custom LOVR build instead of the auto-downloaded one:
    export LOVR_BIN=/path/to/your/lovr      # or set lovr_bin: in render_mcp.yaml

Then open https://<host>:8080, click "Start Mic", click "Launch XR" (or the
WebXR DevUI on desktop). Speak; the sphere tracks your voice in the headset.

The CloudXR EULA is accepted via cloudxr_runtime.yaml (see ``accept_eula``).
"""
import argparse
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from loguru import logger
from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent


def _detect_gpu_config() -> str:
    """Return the GPU config profile by querying nvidia-smi.

    Profiles
    --------
    dual_48G_ada   — 2× ADA 48 GB (default / current dev box)
    spark          — 1× Blackwell GB10 (DGX Spark; ~96 GiB GPU-visible HBM)
    96G_blackwell  — 1× Blackwell ~96 GB

    Falls back to ``dual_48G_ada`` on any detection failure.
    """
    # Query name, compute_cap, and memory.total.
    # Use only csv,noheader — nounits is not supported on all driver versions.
    # Memory values arrive as "47940 MiB" or "Not Supported"; strip units below.
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except Exception as exc:
        logger.warning("nvidia-smi unavailable ({}) — using dual_48G_ada", exc)
        return "dual_48G_ada"

    # Known Spark GPU names (unified memory, no discrete memory.total).
    _SPARK_NAMES = {"gb10", "b10"}

    gpus: list[tuple[str, float, float]] = []  # (name, compute_cap, mem_mib)
    for line in raw:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, cap_str, mem_str = parts[0], parts[1], parts[2]
        try:
            cap = float(cap_str)
        except ValueError:
            continue
        # Memory: "47940 MiB", "N/A", "Not Supported" → extract number or 0
        mem = 0.0
        for tok in mem_str.split():
            try:
                mem = float(tok)
                break
            except ValueError:
                pass
        gpus.append((name.lower(), cap, mem))

    if not gpus:
        logger.warning("GPU detection returned no parseable data — using dual_48G_ada")
        return "dual_48G_ada"

    n_gpus       = len(gpus)
    first_name   = gpus[0][0]
    first_cap    = gpus[0][1]
    is_blackwell = first_cap >= 10.0
    # Spark: name contains a known Spark identifier, OR all memory values are
    # zero (unified memory) and the GPU is Blackwell.
    is_spark     = any(s in first_name for s in _SPARK_NAMES)
    known_mem    = [m for _, _, m in gpus if m > 0]
    total_mem_gb = sum(known_mem) / 1024 if known_mem else 0.0

    if is_blackwell and (is_spark or (not known_mem)):
        cfg = "spark"
    elif is_blackwell and total_mem_gb >= 120:
        cfg = "spark"
    elif is_blackwell:
        cfg = "96G_blackwell"
    elif n_gpus >= 2:
        cfg = "dual_48G_ada"
    else:
        cfg = "dual_48G_ada"

    mem_str = f"{total_mem_gb:.0f} GiB" if known_mem else "unified memory"
    logger.info(
        "GPU config: {}  ({}x {}, {}, SM{:.1f})",
        cfg, n_gpus, gpus[0][0].upper(), mem_str, first_cap,
    )
    return cfg


# agent-llm (Nemotron-30B) loads first so its FlashInfer MoE JIT compilation
# runs with the full GPU free.  The compiled kernels are cached after the
# first run.  On ADA, nemotron3_nano is pinned to cuda:1 so this order has
# no downside there either.
def _build_processes() -> list[Process]:
    """Detect the GPU profile and return the per-profile process list.

    Deferred to call time (rather than module import) so log calls inside
    ``_detect_gpu_config`` happen after ``setup_logging`` has installed
    loguru sinks.
    """
    ai = f"yaml/{_detect_gpu_config()}"
    return [
        Process("hub",        "../../server-runtime",                 "xr_media_hub",
                config="yaml/xr_media_hub.yaml"),
        Process("cloudxr",    "../../cloudxr-runtime",                "cloudxr_runtime",
                config="yaml/cloudxr_runtime.yaml"),
        Process("stt",        "../../ai-services/stt-server",         "stt_server",
                config=f"{ai}/stt_server.yaml"),
        Process("tts",        "../../ai-services/tts/piper",          "piper_tts_server",
                config=f"{ai}/piper_tts_server.yaml"),
        Process("agent-llm",  "../../ai-services/llm/nemotron3_nano", "nemotron3_nano_llm_server",
                config=f"{ai}/nemotron3_nano_llm_server.yaml"),
        Process("vlm",        "../../ai-services/vlm-server",         "vlm_server",
                config=f"{ai}/vlm_server.yaml"),
        Process("llm",        "../../ai-services/llm/llama_nemotron", "llama_nemotron_llm_server",
                config=f"{ai}/llama_nemotron_llm_server.yaml"),
        Process("vlm-mcp",    "../../agent-mcp-servers/vlm-mcp",      "vlm_mcp_server",
                config="yaml/vlm_mcp_server.yaml"),
        Process("video-mcp",  "../../agent-mcp-servers/video-mcp",    "video_mcp_server",
                config="yaml/video_mcp_server.yaml"),
        Process("render-mcp", "../../agent-mcp-servers/render-mcp",   "render_mcp"),
        Process("oxr-mcp",    "../../agent-mcp-servers/oxr-mcp",      "oxr_mcp_server",
                config="yaml/oxr_mcp_server.yaml"),
        Process("worker",     "worker",                               "xr_render_demo_worker",
                config="yaml/xr_render_demo_worker.yaml"),
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


# ── Model cleanup (--stop) ────────────────────────────────────────────────────

# vLLM-backed servers that survive stack shutdown (start_new_session=True).
# Ports match the defaults in each server's YAML; override here if you change them.
_PERSISTENT_SERVERS: list[tuple[str, int]] = [
    ("vlm",       8100),
    ("llm",       8106),
    ("agent-llm", 8107),
]


def _pid_on_port(port: int) -> int | None:
    """Return the PID of the process listening on *port*, or None."""
    # Try ss first (iproute2, always present on modern Linux).
    try:
        out = subprocess.check_output(
            ["ss", "-tlnpH", f"sport = :{port}"],
            text=True, stderr=subprocess.DEVNULL,
        )
        m = re.search(r"pid=(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    # Fallback: lsof.
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{port}"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return int(out.splitlines()[0])
    except Exception:
        pass
    return None


def _stop_models() -> None:
    """Send SIGTERM to any persisted vLLM processes, wait, then SIGKILL if needed."""
    found = False
    for name, port in _PERSISTENT_SERVERS:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as r:
                if r.status != 200:
                    continue
        except Exception:
            continue

        pid = _pid_on_port(port)
        if pid is None:
            print(f"  [{name}] running on :{port} but could not find PID — "
                  f"kill manually", flush=True)
            found = True
            continue

        print(f"  [{name}] stopping (pid={pid}, port={port})…", flush=True)
        found = True
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(40):          # wait up to 20 s
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)      # still alive?
                except ProcessLookupError:
                    print(f"  [{name}] stopped", flush=True)
                    break
            else:
                print(f"  [{name}] force-killing", flush=True)
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            print(f"  [{name}] already gone", flush=True)

    if not found:
        print("  No persistent model servers found running.", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    setup_logging("orchestrator", namespace="xr-render-demo")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--stop", action="store_true",
                   help="Stop any persisted vLLM model servers and exit.")
    ns, _ = p.parse_known_args()

    if ns.stop:
        _stop_models()
        return

    _ensure_web_vendor()
    _ensure_lovr_bin()
    run_stack(_build_processes(), _BASE)


if __name__ == "__main__":
    run()
