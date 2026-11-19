# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped test instrumentation; excludes audio/image payloads and HTTP bodies."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class _Recording:
    def __init__(self, directory: Path, *, filename: str = "events.jsonl", hooks: bool = True) -> None:
        self._file = (directory / filename).open("w", buffering=1)
        self._write_lock = threading.Lock()
        self._hooks = hooks
        self.started: float | None = None
        self._patches: list[tuple[object, str, object]] = []
        self._requests: set[str] = set()
        self._error_sink: int | None = None

    def observe_errors(self) -> None:
        from loguru import logger

        if self._error_sink is not None:
            try:
                logger.remove(self._error_sink)
            except ValueError:
                pass  # Native logging setup may already have removed the sink.

        def observed_error(message):
            record = message.record
            exception_type = record["exception"].type if record["exception"] else None
            if record["level"].no < 40 and exception_type is None:
                return
            self.emit(
                "error",
                component="model" if (record["name"] or "").startswith("xr_ai_models") else "runtime",
                logger=record["name"],
                function=record["function"],
                exception_type=exception_type.__name__ if exception_type else None,
            )

        self._error_sink = logger.add(observed_error, level="WARNING")

    def emit(self, event: str, **values) -> None:
        now = time.monotonic()
        with self._write_lock:
            self._file.write(
                json.dumps(
                    {
                        "event": event,
                        "monotonic": now,
                        "elapsed": None if self.started is None else now - self.started,
                        **values,
                    },
                    default=str,
                )
                + "\n"
            )

    def patch(self, owner, name: str, value) -> None:
        self._patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def __enter__(self):
        try:
            return self._install_hooks()
        except BaseException:
            self.__exit__()
            raise

    def _install_hooks(self):
        if not self._hooks:
            return self
        self.observe_errors()
        from xr_ai_tools.tool_calling import ToolLoopError

        error_init = ToolLoopError.__init__

        def observed_tool_error(error, *args, **kwargs):
            error_init(error, *args, **kwargs)
            self.emit(
                "error", component="runtime", logger="xr_ai_tools.tool_calling",
                function="run_tool_loop", exception_type=type(error).__name__,
            )

        self.patch(ToolLoopError, "__init__", observed_tool_error)
        from xr_ai_interaction import UserQuery
        from xr_ai_runtime import AgentRuntime
        from xr_ai_voice._processors.speaker_stt import _SpeakerSttProcessor

        publish = AgentRuntime._publish
        deliver = AgentRuntime._deliver
        process, reset, event = (
            _SpeakerSttProcessor.process_frame,
            _SpeakerSttProcessor._reset,
            _SpeakerSttProcessor._event,
        )
        drop = _SpeakerSttProcessor._drop_backlog

        async def observed_publish(runtime, topic, message, **kwargs):
            if topic.name == "voice.output":
                self.emit("runtime", topic=topic.name, message=message.model_dump(mode="json"), **kwargs)
            return await publish(runtime, topic, message, **kwargs)

        async def observed_deliver(runtime, state, method, topic_name, message, metadata, **kwargs):
            if isinstance(message, UserQuery) and metadata.message_id not in self._requests:
                self._requests.add(metadata.message_id)
                self.emit(
                    "request", request_id=metadata.message_id, topic=topic_name, message=message.model_dump(mode="json")
                )
            return await deliver(runtime, state, method, topic_name, message, metadata, **kwargs)

        async def observed_process(processor, frame, direction):
            result = await process(processor, frame, direction)
            self.emit("stt_queue", seconds=max(processor._queued_bytes.values(), default=0) / 32000)
            return result

        async def observed_reset(processor, pid, **kwargs):
            self.emit("stt_reset", participant_id=pid)
            return await reset(processor, pid, **kwargs)

        async def observed_event(processor, pid, value):
            if value.get("kind") != "partial":
                self.emit("stt_event", participant_id=pid, value=value)
            return await event(processor, pid, value)

        def observed_drop(processor, pid, queue, skipped):
            self.emit("stt_drop", participant_id=pid)
            return drop(processor, pid, queue, skipped)

        for owner, name, value in [
            (AgentRuntime, "_publish", observed_publish),
            (AgentRuntime, "_deliver", observed_deliver),
            (_SpeakerSttProcessor, "process_frame", observed_process),
            (_SpeakerSttProcessor, "_reset", observed_reset),
            (_SpeakerSttProcessor, "_event", observed_event),
            (_SpeakerSttProcessor, "_drop_backlog", observed_drop),
        ]:
            self.patch(owner, name, value)
        return self

    def __exit__(self, *_exc) -> None:
        if self._error_sink is not None:
            from loguru import logger

            try:
                logger.remove(self._error_sink)
            except ValueError:
                pass  # The worker owns its logging setup.
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        with self._write_lock:
            self._file.close()
