"""
cloudxr-agent worker — connects to the hub via IPC.

A starting point for CloudXR-enabled agents.  This worker receives media from
XR participants over LiveKit/IPC and can send data back via the hub.

The CloudXR process independently streams simulation/render content directly
to XR devices over WSS (port 48322) — it does not pass through this worker.
See cloudxr_runtime.yaml for CloudXR configuration.

Launched as a subprocess by ``uv run cloudxr_agent`` (the orchestrator).
Do not run this directly.
"""
from __future__ import annotations

import asyncio
import logging
import signal
from collections import defaultdict

from xr_ai_agent import AudioChunk, DataMessage, FrameSignal, ParticipantEvent, ProcessorEndpoint

log = logging.getLogger("cloudxr_agent")

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


class CloudXRAgent:
    def __init__(self) -> None:
        self._ep = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
        self._ep.on_frame(self._on_frame)
        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_participant(self._on_participant)
        self._frames: dict[str, int] = defaultdict(int)

    async def _on_frame(self, sig: FrameSignal) -> None:
        self._frames[sig.participant_id] += 1

    async def _on_audio(self, chunk: AudioChunk) -> None:
        pass  # TODO: add audio processing

    async def _on_data(self, msg: DataMessage) -> None:
        log.info("data from %r  topic=%r  bytes=%d", msg.participant_id, msg.topic, len(msg.data))

    async def _on_participant(self, event: ParticipantEvent) -> None:
        pid = event.participant_id
        if event.joined:
            log.info("participant joined: %r", pid)
        else:
            log.info("participant left: %r  frames_received=%d",
                     pid, self._frames.pop(pid, 0))

    async def run(self) -> None:
        await self._ep.run()

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    agent = CloudXRAgent()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    log.info("cloudxr-agent connecting  sub=%s  push=%s", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        agent.shutdown()

    log.info("cloudxr-agent stopped")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
