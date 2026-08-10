# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select one foreground NAT function before making any model call."""

from __future__ import annotations

import json

from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import LLMRef

from ...agents.factory import build_routed_agent
from ...agents.invoke import invoke_with_tool_retry
from ...agents.prompts import HUMAN
from ...runtime.events import emit
from ...runtime.scope import current_invocation
from ...runtime.state import Session
from .runtime import ApplicationOwnership
from .spec import ApplicationCatalog
from .turn import ApplicationTurn, add_application_turn
from .types import RoutedFunction


class ApplicationManager:
    def __init__(self, spec: ApplicationCatalog, ownership: ApplicationOwnership) -> None:
        self.spec = spec
        self.ownership = ownership
        self._turn: Function | None = None
        self._root: Function | None = None
        self._root_agent: Function | None = None
        self._foreground: dict[str, Function] = {}
        self._functions: tuple[RoutedFunction, ...] = ()

    def register_foreground(self, app_id: str, application: Function) -> None:
        if self.spec.application(app_id).mode != "foreground":
            raise ValueError(f"{app_id} is not a foreground application")
        if not isinstance(application, Function):
            raise TypeError("foreground applications must be NAT functions")
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
        self._root_agent = await build_routed_agent(
            builder,
            name="foreground_root",
            llm_ref=llm_ref,
            prompt=prompt,
            functions=functions,
        )
        self._root = await add_application_turn(
            builder,
            name="application_manager__root_turn",
            description="Handle one root-assistant turn.",
            handler=self._route_root,
        )
        self._turn = await add_application_turn(
            builder,
            name="application_manager__turn",
            description="Dispatch one turn to the current foreground application.",
            handler=self._dispatch,
        )

    async def _dispatch(self, session: Session, request: str, trace_id: str) -> str:
        foreground = self.ownership.current(session)
        emit(
            "application_manager.route",
            participant_id=session.participant_id,
            trace_id=trace_id,
            foreground=foreground,
            background=sorted(session.applications.background),
        )
        application = self._root if foreground == self.ownership.ROOT else self._foreground[foreground]
        if application is None:
            raise RuntimeError("application manager has not been built")
        result = await application.ainvoke(ApplicationTurn(request=request), to_type=str)
        return result.strip()

    async def _route_root(self, session: Session, request: str, trace_id: str) -> str:
        if self._root_agent is None:
            raise RuntimeError("application manager has not been built")
        foreground = self.ownership.ROOT
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

        result = await invoke_with_tool_retry(self._root_agent, payload, retry=retry)
        emit(
            "agent.foreground.response",
            participant_id=session.participant_id,
            step=None,
            trace_id=trace_id,
            foreground=foreground,
            operation=call.route_operation or "answer",
            next_foreground=self.ownership.current(session),
            background=sorted(session.applications.background),
            output=result,
        )
        return result

    @property
    def functions(self) -> tuple[RoutedFunction, ...]:
        return self._functions

    @property
    def function(self) -> Function:
        if self._turn is None:
            raise RuntimeError("application manager has not been built")
        return self._turn


__all__ = ["ApplicationManager"]
