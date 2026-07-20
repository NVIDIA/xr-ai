# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Track live hub frame signals and fetch pixels on demand."""

from xr_ai_agent import FrameData, FrameSignal, ProcessorEndpoint


class LiveFrameProvider:
    def __init__(self, endpoint: ProcessorEndpoint) -> None:
        self._endpoint = endpoint
        self._latest: dict[str, FrameSignal] = {}
        endpoint.on_frame(self._on_frame)

    async def _on_frame(self, signal: FrameSignal) -> None:
        previous = self._latest.get(signal.participant_id)
        if previous is None or signal.pts_us >= previous.pts_us:
            self._latest[signal.participant_id] = signal

    def participants(self) -> list[str]:
        return sorted(self._endpoint.connected_participants)

    async def fetch_latest(self, participant_id: str) -> FrameData | None:
        signal = self._latest.get(participant_id)
        if signal is None:
            return None
        return await self._endpoint.request_frame(signal)
