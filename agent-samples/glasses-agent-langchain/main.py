# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
glasses-agent-langchain orchestrator — always-on AI assistant for smart glasses.

Process stack:
  hub             → xr_media_hub          (server-runtime)
  stt             → stt_server            (ai-services/stt-server)
  tts             → piper_tts_server      (ai-services/tts/piper)
  vlm             → vlm_server            (ai-services/vlm-server)
  llm             → llama_nemotron_llm_server (ai-services/llm/llama_nemotron)
  vlm-mcp         → vlm_mcp_server        (agent-mcp-servers/vlm-mcp)
  video-mcp       → video_mcp_server      (agent-mcp-servers/video-mcp)
  transcript-mcp  → transcript_mcp_server (agent-mcp-servers/transcript-mcp)
  worker          → glasses_agent_langchain_worker  (this sample's worker)

How to run (from agent-samples/glasses-agent-langchain/):
    uv sync && uv run glasses_agent_langchain

To stop persisted model servers without restarting the full stack:
    uv run glasses_agent_langchain --stop
"""
import argparse
import datetime
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from loguru import logger
from xr_ai_launcher import Process, ensure_credentials, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

PROCESSES = [
    Process("hub",            "../../server-runtime",                   "xr_media_hub",
            config="yaml/xr_media_hub.yaml"),
    Process("stt",            "../../ai-services/stt-server",           "stt_server",
            config="yaml/stt_server.yaml"),
    Process("tts",            "../../ai-services/tts/piper",            "piper_tts_server",
            config="yaml/piper_tts_server.yaml"),
    Process("agent-llm",      "../../ai-services/llm/nemotron3_nano",   "nemotron3_nano_llm_server",
            config="yaml/nemotron3_nano_llm_server.yaml"),
    Process("vlm",            "../../ai-services/vlm-server",           "vlm_server",
            config="yaml/vlm_server.yaml"),
    Process("llm",            "../../ai-services/llm/llama_nemotron",   "llama_nemotron_llm_server",
            config="yaml/llama_nemotron_llm_server.yaml"),
    Process("vlm-mcp",        "../../agent-mcp-servers/vlm-mcp",        "vlm_mcp_server",
            config="yaml/vlm_mcp_server.yaml"),
    Process("video-mcp",      "../../agent-mcp-servers/video-mcp",      "video_mcp_server",
            config="yaml/video_mcp_server.yaml"),
    Process("transcript-mcp", "../../agent-mcp-servers/transcript-mcp", "transcript_mcp_server",
            config="yaml/transcript_mcp_server.yaml"),
    Process("worker",         "worker",                                  "glasses_agent_langchain_worker",
            config="yaml/glasses_agent_langchain_worker.yaml"),
]


# ── Model persistence (--stop) ────────────────────────────────────────────────

# vLLM-backed servers that survive stack shutdown (start_new_session=True in
# the server wrapper). Ports match the defaults in each server's YAML.
_PERSISTENT_SERVERS: list[tuple[str, int]] = [
    ("agent-llm", 8107),
    ("vlm",       8100),
    ("llm",       8106),
    ("stt",       8103),
]


def _pid_on_port(port: int) -> int | None:
    """Return the PID of the process listening on *port*, or None."""
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
            for _ in range(40):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
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


# ── Output path helpers ───────────────────────────────────────────────────────

# Default output dirs when no --run-dir is given (also what --clear wipes).
_DEFAULT_TRANSCRIPTS = Path("/tmp/xr_transcripts")
_DEFAULT_FRAMES      = Path("/tmp/xr_video_queries")
_DEFAULT_TRACE_LOG   = Path("/tmp/glasses-agent-langchain-trace.log")


def _clear_outputs(run_dir: Path | None) -> None:
    """Delete stored transcripts, frames, and trace log."""
    if run_dir:
        targets: list[Path] = [run_dir]
    else:
        targets = [_DEFAULT_TRANSCRIPTS, _DEFAULT_FRAMES, _DEFAULT_TRACE_LOG]
    for p in targets:
        if not p.exists():
            print(f"  (not found) {p}", flush=True)
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"  cleared {p}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--stop",  action="store_true",
                   help="Stop any persisted vLLM model servers and exit.")
    p.add_argument("--clear", action="store_true",
                   help="Delete stored transcripts, frames, and trace log, then exit.")
    p.add_argument("--run-dir", type=Path, default=None, metavar="DIR",
                   help="Directory for transcripts, frames, and trace log. "
                        "Default: auto-generated under /tmp/xr-glasses-langchain/.")
    ns, _ = p.parse_known_args()

    if ns.stop:
        _stop_models()
        return

    if ns.clear:
        _clear_outputs(ns.run_dir)
        return

    # Resolve run dir: use given path or generate a timestamped one.
    run_dir: Path = ns.run_dir or (
        Path("/tmp/xr-glasses-langchain")
        / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XR_RUN_DIR"] = str(run_dir)

    setup_logging("orchestrator", namespace="glasses-agent-langchain")
    logger.info("run dir: {}", run_dir)
    ensure_credentials("HF_TOKEN")
    run_stack(PROCESSES, _BASE)


if __name__ == "__main__":
    run()
