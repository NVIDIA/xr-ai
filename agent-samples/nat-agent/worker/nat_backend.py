# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NAT backend — drives a NAT ``tool_calling_agent`` workflow against the
remote nemotron3_nano LLM and the vlm-mcp / video-mcp MCP servers.

Architecture
------------
    user text  ->  prepend `[Live participant_id: <pid>]` preamble
                 -> tool_calling_agent (NAT + LangChain + LangGraph)
                    | LLM decides: 0 or more tool calls
                    v
              MCP tools registered as mcp_tool_wrapper functions:
                ask_image                 (vlm-mcp)
                get_latest_frame          (video-mcp)
                get_frame_at_time         (video-mcp)
                get_video_stats           (video-mcp)
                query_video               (video-mcp)
                list_live_participants    (video-mcp)
                list_recorded_participants (video-mcp)

Concurrency
-----------
NAT's workflow runs on a dedicated asyncio loop on a worker thread.
``infer()`` is callable from the main thread (or from any thread) and
serialises through ``run_coroutine_threadsafe``. There is no per-turn
shared global — pid travels in the user message preamble.

API version note
----------------
Targets ``nvidia-nat>=1.3``:
- ``WorkflowBuilder.from_config(Config)`` -> async context manager.
- ``builder.build()`` -> coroutine returning a Workflow.
- ``workflow.run(msg).result(to_type=str)`` -> final answer string.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import pathlib
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from config import WorkerConfig, load_nat_workflow_dict

log = logging.getLogger("nat_agent.backend")

# Workflow YAML lives at the sample root, one level up from worker/.
_WORKFLOW_YAML = pathlib.Path(__file__).resolve().parents[1] / "nat_agent_workflow.yaml"


def _ensure_nat_plugins() -> None:
    """Discover NAT plugin types (mcp_tool_wrapper, openai LLM, etc.).

    Idempotent: NAT short-circuits subsequent calls.
    """
    from nat.runtime.loader import PluginTypes, discover_and_register_plugins
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)


class NatBackend:
    """NAT workflow runtime on a dedicated worker-thread asyncio loop."""

    def __init__(self, cfg: WorkerConfig) -> None:
        self._cfg = cfg
        self._from_config_cm: Any = None
        self._workflow: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._ready = threading.Event()
        # Serialise concurrent infer() calls. NAT 1.6's tool_calling_agent
        # builds a single bound LLM (``llm.bind_tools(tools)``) at workflow
        # build time and reuses it across calls; running multiple turns
        # through it concurrently interleaves chat-completion requests and
        # delivers responses to the wrong turn in the queue. Pipecat will
        # happily fire several TranscriptionFrames in a row when the user
        # is actively talking — without this lock, we get exactly that.
        self._infer_lock = threading.Lock()

    # ── worker-thread asyncio loop ────────────────────────────────────────────

    def _start_worker_loop(self) -> None:
        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            self._loop.run_forever()

        self._loop_thread = threading.Thread(
            target=_run, name="nat-backend-loop", daemon=True,
        )
        self._loop_thread.start()
        self._ready.wait()

    def _wait_for_llm_server(
        self, timeout_s: float = 600.0, interval_s: float = 2.0,
    ) -> None:
        """Block until the LLM server's OpenAI ``/v1/models`` responds.

        vLLM cold-start (weight load + reasoning-parser fetch) can take
        a few minutes, hence the long deadline. Without this gate the
        first NAT workflow build fails because LangChain's ChatOpenAI
        probes the model list at construction.
        """
        probe_url = self._cfg.llm_server.rstrip("/") + "/v1/models"
        deadline = time.monotonic() + timeout_s
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                with urllib.request.urlopen(probe_url, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        log.info("LLM server reachable at %s (attempt %d)",
                                 probe_url, attempt)
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            if attempt % 15 == 1:
                log.info("Waiting for LLM server at %s …", probe_url)
            time.sleep(interval_s)
        raise RuntimeError(
            f"LLM server at {probe_url} did not become ready within "
            f"{timeout_s:.0f}s — check the 'llm' process logs."
        )

    async def _ensure_loaded_async(self) -> None:
        if self._workflow is not None:
            return

        _ensure_nat_plugins()

        from nat.builder.workflow_builder import WorkflowBuilder
        from nat.data_models.config import Config

        workflow_dict = load_nat_workflow_dict(self._cfg, _WORKFLOW_YAML)
        config = Config(**workflow_dict)

        cm = WorkflowBuilder.from_config(config)
        try:
            builder = await cm.__aenter__()
            workflow = await builder.build()
        except BaseException:
            with contextlib.suppress(Exception):
                await cm.__aexit__(None, None, None)
            raise

        self._from_config_cm = cm
        self._workflow = workflow
        log.info(
            "NAT tool_calling_agent workflow built  llm_server=%s  "
            "vlm_mcp=%s  video_mcp=%s",
            self._cfg.llm_server, self._cfg.vlm_mcp_url, self._cfg.video_mcp_url,
        )

    def ensure_loaded(self) -> None:
        """Block until the LLM server is reachable and the workflow is built.

        Safe to call multiple times — second call is a no-op once loaded.
        Designed to run from a thread pool executor (e.g. via
        ``loop.run_in_executor``) so it doesn't block the asyncio loop.
        """
        if self._loop is None:
            self._start_worker_loop()
        if self._workflow is not None:
            return
        self._wait_for_llm_server()
        fut = asyncio.run_coroutine_threadsafe(
            self._ensure_loaded_async(), self._loop,
        )
        fut.result()

    # ── inference ─────────────────────────────────────────────────────────────

    async def _invoke_async(self, user: str) -> str:
        """Drive the tool-calling workflow once and return the final text."""
        assert self._workflow is not None
        async with self._workflow.run(user) as runner:
            result = await runner.result(to_type=str)
        return str(result).strip()

    def infer(self, user: str, participant_id: str) -> str:
        """Synchronous one-turn inference.

        Call via ``loop.run_in_executor`` from asyncio code so the main loop
        is not blocked by the round-trip to the LLM + MCP tools.

        The participant_id is prepended to the user message so the LLM has
        it in context and can pass it into video-mcp tool calls. Per-call
        state lives entirely in the augmented user message; the only shared
        state across calls is the workflow itself, which we guard with
        ``self._infer_lock`` (see __init__).
        """
        self.ensure_loaded()
        if participant_id:
            augmented = (
                f"[Live participant_id: {participant_id}]\n"
                f"User: {user}"
            )
        else:
            augmented = user
        # Allow up to several tool round-trips (each capped by NAT's
        # tool_call_timeout) before giving up.
        timeout = (self._cfg.llm_request_timeout_s * 6) + 60.0
        with self._infer_lock:
            fut = asyncio.run_coroutine_threadsafe(
                self._invoke_async(augmented), self._loop,
            )
            return fut.result(timeout=timeout)

    async def close(self) -> None:
        if self._from_config_cm is not None and self._loop is not None:
            fut = asyncio.run_coroutine_threadsafe(
                self._from_config_cm.__aexit__(None, None, None), self._loop,
            )
            try:
                fut.result(timeout=5.0)
            except Exception:
                log.exception("NAT workflow teardown failed")
            self._from_config_cm = None
            self._workflow = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread:
                self._loop_thread.join(timeout=2.0)
            self._loop = None
            self._loop_thread = None
