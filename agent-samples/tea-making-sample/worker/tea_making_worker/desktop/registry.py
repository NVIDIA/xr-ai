# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select one foreground application before making any model call."""

from __future__ import annotations

import json
from typing import Protocol

from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import LLMRef

from ..agents.factory import build_routed_agent
from ..agents.invoke import invoke_with_tool_retry
from ..agents.prompts import HUMAN
from ..runtime.events import emit
from ..runtime.scope import current_invocation
from ..runtime.state import Session
from .runtime import DesktopRuntime
from .spec import DesktopSpec
from .types import RoutedFunction


class ForegroundApplication(Protocol):
    async def route(self, session: Session, request: str, trace_id: str) -> str: ...


class Desktop:
    def __init__(self, spec: DesktopSpec, runtime: DesktopRuntime) -> None:
        self.spec = spec
        self.runtime = runtime
        self._root: Function | None = None
        self._foreground: dict[str, ForegroundApplication] = {}
        self._functions: tuple[RoutedFunction, ...] = ()

    def register_foreground(self, app_id: str, application: ForegroundApplication) -> None:
        if self.spec.application(app_id).mode != "foreground":
            raise ValueError(f"{app_id} is not a foreground application")
        if app_id in self._foreground:
            raise ValueError(f"foreground application already registered: {app_id}")
        self._foreground[app_id] = application

    async def build(
        self,
        builder: WorkflowBuilder,
        llm_ref: LLMRef,
        functions: tuple[RoutedFunction, ...],
    ) -> None:
        self._functions = functions
        catalog = "; ".join(function.catalog_entry() for function in functions)
        prompt = f"{self.spec.root_prompt}\nRoutes: {catalog}\n{HUMAN}"
        self._root = await build_routed_agent(
            builder,
            name="foreground_root",
            llm_ref=llm_ref,
            prompt=prompt,
            functions=functions,
        )

    async def route(self, session: Session, request: str, trace_id: str) -> str:
        foreground = self.runtime.current(session)
        emit(
            "desktop.route",
            participant_id=session.participant_id,
            trace_id=trace_id,
            foreground=foreground,
            background=sorted(session.desktop.background),
        )
        if foreground != self.runtime.ROOT:
            return await self._foreground[foreground].route(session, request, trace_id)
        if self._root is None:
            raise RuntimeError("desktop has not been built")
        payload = json.dumps({"request": request}, ensure_ascii=False, separators=(",", ":"))
        emit(
            "agent.foreground.request",
            participant_id=session.participant_id,
            step=None,
            trace_id=trace_id,
            foreground=foreground,
            chars=len(payload),
            input=payload,
        )
        call = current_invocation()
        call.route_operation = None

        def retry(feedback: str) -> None:
            emit(
                "agent.foreground.retry",
                participant_id=session.participant_id,
                step=None,
                trace_id=trace_id,
                foreground=foreground,
                reason="invalid_tool_arguments",
                feedback=feedback,
            )

        result = await invoke_with_tool_retry(self._root, payload, retry=retry)
        emit(
            "agent.foreground.response",
            participant_id=session.participant_id,
            step=None,
            trace_id=trace_id,
            foreground=foreground,
            operation=call.route_operation or "answer",
            next_foreground=self.runtime.current(session),
            background=sorted(session.desktop.background),
            output=result,
        )
        return result.strip()

    @property
    def functions(self) -> tuple[RoutedFunction, ...]:
        return self._functions


__all__ = ["Desktop", "ForegroundApplication"]
