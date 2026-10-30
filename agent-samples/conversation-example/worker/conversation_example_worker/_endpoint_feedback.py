# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send sample-owned endpoint provenance over the existing client data path."""

from loguru import logger
from xr_ai_hub import DataMessage
from xr_ai_interaction import UserQuery
from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_voice import HubVoiceTransport

from ._semantic_voice import _ClefVoicePolicy
from ._topics import USER_QUERY_TOPIC


class _EndpointFeedback(Agent):
    def __init__(self, policy: _ClefVoicePolicy, transport: HubVoiceTransport) -> None:
        super().__init__()
        self._policy = policy
        self._transport = transport

    @subscribe(USER_QUERY_TOPIC)
    async def query(self, query: UserQuery, ctx: RuntimeContext) -> None:
        pid = ctx.metadata.participant_id
        if pid is None:
            return
        source = self._policy.take_admission_source(pid, query.request_id)
        if source is None:
            return
        try:
            await self._transport.send_return_data(
                DataMessage(
                    participant_id=pid,
                    topic="voice.user-query",
                    pts_us=query.timestamp_us,
                    data=f"User [{source}]: {query.text}".encode("utf-8"),
                )
            )
        except Exception:  # noqa: BLE001 - diagnostic delivery must not fail the admitted query
            logger.exception("voice endpoint feedback delivery failed pid={!r}", pid)
