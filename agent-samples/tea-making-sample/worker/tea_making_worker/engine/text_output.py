# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish labeled application text without entering the voice pipeline."""

import time

from xr_ai_hub import DataMessage
from xr_ai_voice import VoiceSession

from ..runtime.events import emit


class TextOutputBridge:
    def __init__(self, session: VoiceSession) -> None:
        self._session = session

    async def send(self, participant_id: str, label: str, message: str) -> None:
        text = message.strip()
        if not text:
            return
        await self._session.transport.send_return_data(
            DataMessage(
                participant_id=participant_id,
                topic=label,
                pts_us=time.time_ns() // 1_000,
                data=text.encode(),
            )
        )
        emit(
            "application.text_output",
            participant_id=participant_id,
            application=label,
            message=text,
        )


__all__ = ["TextOutputBridge"]
