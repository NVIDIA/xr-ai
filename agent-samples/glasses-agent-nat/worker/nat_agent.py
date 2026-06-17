# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-time wrapper around the configured NeMo Agent Toolkit workflow."""

import logging

from nat_runtime import NatRuntime

log = logging.getLogger("glasses_agent_nat.nat_agent")
_trace_log = logging.getLogger("glasses_agent_nat.trace")


class NatAgentRunner:
    """Formats one glasses turn and invokes the NAT tool-calling workflow."""

    def __init__(self, runtime: NatRuntime) -> None:
        self._runtime = runtime

    async def run(
        self,
        *,
        context: str,
        transcript: str,
        needs_thinking: bool,
        participant_id: str,
    ) -> str:
        thinking_prefix = ""
        if needs_thinking:
            thinking_prefix = (
                "This request may require visual or temporal care. Use the "
                "provided context and tools as needed, then return only a "
                "concise answer for the wearer.\n\n"
            )

        prompt = (
            f"{thinking_prefix}"
            f"[Context - use this before calling tools]\n{context}\n\n"
            f"[User request]\n{transcript}"
        )
        response = await self._runtime.run_agent(prompt, participant_id=participant_id)
        _trace_log.info("RESP  %s", response)
        return response
