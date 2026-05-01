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
              MCP tools auto-discovered via mcp_client function groups:
                ask_image                  (vlm-mcp)
                get_frame_from_time        (video-mcp)
                get_video_stats            (video-mcp)
                query_video                (video-mcp)
                list_live_participants     (video-mcp)
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
        self._workflow: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        # _lifecycle task coordination (all accessed only after _init_lock)
        self._workflow_stop_event: asyncio.Event | None = None
        self._workflow_ready = threading.Event()  # set when workflow built or failed
        self._workflow_done = threading.Event()   # set when _lifecycle task exits
        self._workflow_error: BaseException | None = None
        self._init_lock = threading.Lock()
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
            self._loop_ready.set()
            self._loop.run_forever()

        self._loop_thread = threading.Thread(
            target=_run, name="nat-backend-loop", daemon=True,
        )
        self._loop_thread.start()
        self._loop_ready.wait()

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

    # ── workflow lifecycle ────────────────────────────────────────────────────

    async def _lifecycle(self) -> None:
        """Own the WorkflowBuilder context from __aenter__ to __aexit__ in one task.

        anyio cancel scopes require that __aexit__ runs in the same asyncio Task
        as __aenter__.  Submitting __aexit__ as a separate coroutine via
        run_coroutine_threadsafe creates a new Task and breaks that invariant,
        causing "Attempted to exit cancel scope in a different task than it was
        entered in".  By holding the ``async with`` open here and waiting for
        _workflow_stop_event, both enter and exit happen in this single task.
        """
        _ensure_nat_plugins()

        from nat.builder.workflow_builder import WorkflowBuilder
        from nat.data_models.config import Config

        workflow_dict = load_nat_workflow_dict(self._cfg, _WORKFLOW_YAML)
        config = Config(**workflow_dict)

        try:
            async with WorkflowBuilder.from_config(config) as builder:
                self._workflow = await builder.build()
                log.info(
                    "NAT tool_calling_agent workflow built  llm_server=%s  "
                    "vlm_mcp=%s  video_mcp=%s",
                    self._cfg.llm_server, self._cfg.vlm_mcp_url, self._cfg.video_mcp_url,
                )
                self._workflow_ready.set()
                await self._workflow_stop_event.wait()
        except BaseException as exc:
            self._workflow_error = exc
            raise
        finally:
            self._workflow = None
            self._workflow_ready.set()  # unblock ensure_loaded() even on failure
            self._workflow_done.set()

    def ensure_loaded(self) -> None:
        """Block until the LLM server is reachable and the workflow is built.

        Safe to call multiple times — subsequent calls are no-ops once loaded.
        Designed to run from a thread pool executor (e.g. via
        ``loop.run_in_executor``) so it doesn't block the asyncio loop.
        """
        if self._workflow_ready.is_set():
            if self._workflow_error is not None:
                raise RuntimeError(
                    "NAT workflow failed to initialize — check logs"
                ) from self._workflow_error
            return

        with self._init_lock:
            # Double-checked: another thread may have completed init while we waited.
            if self._workflow_ready.is_set():
                if self._workflow_error is not None:
                    raise RuntimeError(
                        "NAT workflow failed to initialize — check logs"
                    ) from self._workflow_error
                return

            if self._loop is None:
                self._start_worker_loop()
            self._wait_for_llm_server()

            async def _start() -> None:
                self._workflow_stop_event = asyncio.Event()
                self._loop.create_task(self._lifecycle(), name="nat-lifecycle")

            asyncio.run_coroutine_threadsafe(_start(), self._loop).result()

        self._workflow_ready.wait()
        if self._workflow_error is not None:
            raise RuntimeError(
                "NAT workflow failed to initialize — check logs"
            ) from self._workflow_error

    # ── inference ─────────────────────────────────────────────────────────────

    async def _invoke_async(self, user: str) -> str:
        """Drive the tool-calling workflow once and return the final text."""
        assert self._workflow is not None
        async with self._workflow.run(user) as runner:
            result = await runner.result(to_type=str)
        return str(result).strip()

    def infer(
        self,
        user:              str,
        participant_id:    str,
        reference_time_us: int = 0,
    ) -> str:
        """Synchronous one-turn inference.

        Call via ``loop.run_in_executor`` from asyncio code so the main loop
        is not blocked by the round-trip to the LLM + MCP tools.

        Per-turn state travels in the augmented user message preamble:

            [Live participant_id: <pid>; user_asked_at_us: <int>]
            User: <user text>

        The LLM forwards both fields verbatim into video-mcp tool calls
        (``participant_id`` and ``reference_time_us`` respectively). Without
        the timestamp anchor, every ``get_frame_from_time`` lookup would be
        offset by the LLM's own thinking latency (5-15 s on eager-mode
        vLLM), which is exactly the kind of "the box is white" hallucination
        we hit in the v1 trace — by the time the tool fires, the live frame
        no longer matches what the user was asking about.

        ``reference_time_us`` is a Unix-microseconds wall clock value. The
        worker captures it at the moment the user finishes speaking (voice
        path) or the data message arrives (data path). ``0`` means "no
        anchor" and falls back to wall clock at tool-fire time.

        The only shared state across calls is the workflow itself, which we
        guard with ``self._infer_lock`` (see ``__init__``).
        """
        self.ensure_loaded()
        if participant_id:
            preamble = (
                f"[Live participant_id: {participant_id}; "
                f"user_asked_at_us: {reference_time_us}]"
            )
            augmented = f"{preamble}\nUser: {user}"
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
        if self._workflow_stop_event is not None and self._loop is not None:
            # Signal _lifecycle to exit the WorkflowBuilder context.  The context
            # exits inside _lifecycle's own task — the same task that entered it.
            self._loop.call_soon_threadsafe(self._workflow_stop_event.set)
            outer_loop = asyncio.get_running_loop()
            await outer_loop.run_in_executor(
                None, lambda: self._workflow_done.wait(timeout=5.0),
            )
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread:
                self._loop_thread.join(timeout=2.0)
            self._loop = None
            self._loop_thread = None
