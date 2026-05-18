# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native NeMo Agent Toolkit wrapper for the glasses request-time agent loop."""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from nat.builder.function import LambdaFunction
from nat.builder.function_info import FunctionInfo
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from config import WorkerConfig

log = logging.getLogger("glasses_agent_nat.nat_agent")
_trace_log = logging.getLogger("glasses_agent_nat.trace")

_MAX_LOOP = 8

ExecuteToolCb = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any] | str | None]]


class GlassesNatLoopConfig(FunctionBaseConfig, name="glasses_agent_nat_loop"):
    """NAT component config for the glasses request-time tool loop."""


class GlassesNatRequest(BaseModel):
    system_prompt: str = Field(description="System prompt for this turn.")
    context: str = Field(description="Memory, camera, participant, and history context.")
    transcript: str = Field(description="User request text.")
    needs_thinking: bool = Field(default=False, description="Enable private model thinking.")


class NatAgentRunner:
    """Runs the glasses tool-calling loop as a native NAT LambdaFunction."""

    def __init__(
        self,
        *,
        cfg: WorkerConfig,
        tools_openai: list[dict[str, Any]],
        execute_tool: ExecuteToolCb,
        http: httpx.AsyncClient,
    ) -> None:
        self._cfg = cfg
        self._tools_openai = tools_openai
        self._execute_tool = execute_tool
        self._http = http

        async def _run(request: GlassesNatRequest) -> str:
            return await self._run_tool_loop(request)

        info = FunctionInfo.from_fn(
            _run,
            input_schema=GlassesNatRequest,
            description=(
                "Answer one smart-glasses user request by calling the configured "
                "VLM and video tools through the worker's MCP clients."
            ),
        )
        self._function = LambdaFunction.from_info(
            config=GlassesNatLoopConfig(),
            info=info,
            instance_name="glasses_agent_nat_loop",
        )

    async def run(
        self,
        *,
        system_prompt: str,
        context: str,
        transcript: str,
        needs_thinking: bool,
    ) -> str:
        request = GlassesNatRequest(
            system_prompt=system_prompt,
            context=context,
            transcript=transcript,
            needs_thinking=needs_thinking,
        )
        result = await self._function.ainvoke(request, str)
        return result.strip() or "Done."

    async def _run_tool_loop(self, request: GlassesNatRequest) -> str:
        needs_thinking = request.needs_thinking
        system_content = request.system_prompt
        if needs_thinking:
            system_content = (
                "Use your private <think> block to reason through the question. "
                "NEVER output these thoughts in your final response - "
                "only output a concise 1-3 sentence answer for the wearer.\n\n"
                + system_content
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": (
                    f"[Context - use this before calling tools]\n"
                    f"{request.context}\n\n"
                    f"[User request]\n{request.transcript}"
                ),
            },
        ]

        for iteration in range(_MAX_LOOP):
            body = {
                "model": "llm",
                "messages": messages,
                "tools": self._tools_openai,
                "max_tokens": 1024 if needs_thinking else 512,
                "temperature": 0.0,
                "chat_template_kwargs": {
                    "enable_thinking": needs_thinking,
                    **({"thinking_budget": 1024} if needs_thinking else {}),
                },
            }
            try:
                resp = await self._http.post(
                    self._cfg.agent_llm_server.rstrip("/") + "/v1/chat/completions",
                    json=body,
                )
                if resp.is_error:
                    log.error("agent-llm %s: %s", resp.status_code, resp.text[:300])
                    break
            except Exception:
                log.exception("agent-llm call failed on iteration %d", iteration)
                break

            choice = resp.json()["choices"][0]
            message = choice["message"]
            finish = choice.get("finish_reason", "")
            tool_calls = message.get("tool_calls") or []
            content = (message.get("content") or "").strip()

            log.info(
                "nat agent iter=%d  finish=%s  tool_calls=%d  content=%r",
                iteration, finish, len(tool_calls), content[:200],
            )

            if not tool_calls:
                if finish == "length" and needs_thinking:
                    log.warning(
                        "iter=%d hit length limit during thinking; retrying without thinking",
                        iteration,
                    )
                    needs_thinking = False
                    continue
                _trace_log.info("RESP  %s", content or "Done.")
                return content or "Done."

            messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                name = tc["function"]["name"]
                args_str = tc["function"].get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}

                log.info("nat tool call  iter=%d  tool=%s  args=%s", iteration, name, args)
                _trace_log.info(
                    "TOOL  [nat:%d] %s(%s)", iteration, name,
                    ", ".join(f"{k}={v}" for k, v in args.items()),
                )
                result = await self._execute_tool(name, args)
                result_str = json.dumps(result, default=str)
                log.info("nat tool result  tool=%s  %s", name, result_str[:200])
                _trace_log.info("RES   [nat:%d] %s -> %s", iteration, name, result_str[:300])

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        return "Done."
