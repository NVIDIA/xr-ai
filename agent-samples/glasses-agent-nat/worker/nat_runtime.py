# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared NeMo Agent Toolkit runtime for the glasses worker."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nat.builder.function import FunctionGroup
from nat.builder.workflow_builder import WorkflowBuilder
from nat.runtime.loader import load_config
from nat.runtime.session import SessionManager

log = logging.getLogger("glasses_agent_nat.nat_runtime")


class NatRuntime:
    """Owns one NAT workflow instance and exposes a small worker-facing API."""

    def __init__(
        self,
        *,
        config_file: Path,
        workflow_builder_cm,
        session_manager: SessionManager,
    ) -> None:
        self._config_file = config_file
        self._workflow_builder_cm = workflow_builder_cm
        self._session_manager = session_manager

    @classmethod
    async def create(cls, config_file: Path) -> "NatRuntime":
        config_file = config_file.resolve()
        config = load_config(config_file)
        workflow_builder_cm = WorkflowBuilder.from_config(config=config)
        workflow_builder = await workflow_builder_cm.__aenter__()
        try:
            session_manager = await SessionManager.create(
                config=config,
                shared_builder=workflow_builder,
                max_concurrency=8,
            )
        except Exception:
            await workflow_builder_cm.__aexit__(None, None, None)
            raise
        log.info("loaded NAT workflow config %s", config_file)
        return cls(
            config_file=config_file,
            workflow_builder_cm=workflow_builder_cm,
            session_manager=session_manager,
        )

    async def run_agent(self, prompt: str, *, participant_id: str = "") -> str:
        """Run the configured NAT workflow for one user-facing request."""
        async with self._session_manager.session(user_id=participant_id or None) as session:
            async with session.run(prompt) as runner:
                result = await runner.result(to_type=str)
        text = self._normalize(result)
        return text.strip() if isinstance(text, str) and text.strip() else "Done."

    async def call_tool(
        self,
        group: str,
        tool: str,
        args: dict[str, Any],
        *,
        participant_id: str = "",
    ) -> dict[str, Any] | list[Any] | str | None:
        """Call one function exposed by a NAT MCP client function group."""
        full_name = f"{group}{FunctionGroup.SEPARATOR}{tool}"
        async with self._session_manager.session(user_id=participant_id or None) as session:
            try:
                function_group = session.workflow.function_groups[group]
                functions = await function_group.get_accessible_functions()
                function = functions[full_name]
            except KeyError as exc:
                raise ValueError(f"NAT tool {full_name!r} is not configured") from exc
            result = await function.ainvoke(args)
        return self._normalize(result)

    async def close(self) -> None:
        await self._session_manager.shutdown()
        await self._workflow_builder_cm.__aexit__(None, None, None)

    @staticmethod
    def _normalize(value: Any) -> dict[str, Any] | list[Any] | str | None:
        """Normalize NAT/MCP outputs to the shapes used by the existing worker."""
        if value is None or isinstance(value, dict | list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value

        structured = getattr(value, "structured_content", None)
        if structured is not None:
            return structured

        content = getattr(value, "content", None)
        if content:
            first = content[0]
            text = getattr(first, "text", None)
            if text is not None:
                try:
                    return json.loads(text)
                except Exception:
                    return text

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump()

        return str(value)
