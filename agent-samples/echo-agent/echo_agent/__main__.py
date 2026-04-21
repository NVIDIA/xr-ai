"""
Echo agent for XR-Media-Hub.

Starts the hub as a subprocess, connects to it via IPC, echoes every audio
chunk back to the originating participant, and sends a JSON stats ping to each
connected participant every 5 seconds.

How to run (from agent-samples/echo-agent/ or xr-ai/):
    uv run echo_agent

What you should see:
    - [hub] prefixed log lines from the hub process.
    - Stats lines printed every 5 s showing audio chunk / data message counters.
    - Clients receive JSON pings on their data channel (topic "agent.stats").
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from collections import defaultdict

from xr_media_hub.ipc import ProcessorEndpoint
from xr_media_hub.ipc._types import AudioChunk, DataMessage, FrameSignal, ParticipantEvent
from xr_ai_launcher import HubLauncher

log = logging.getLogger("echo_agent")

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"
_STATS_INTERVAL_S = 5.0


def _now_us() -> int:
    return time.time_ns() // 1_000


class EchoAgent:
    """
    Stateful echo agent.

    Tracks per-participant counters and periodically emits stats back to each
    connected participant via the hub's return-data path.
    """

    def __init__(self) -> None:
        self._ep = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
        self._ep.on_frame(self._on_frame)
        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_participant(self._on_participant)

        # participant_id → counter value
        self._video_frames:  dict[str, int]   = defaultdict(int)
        self._audio_chunks:  dict[str, int]   = defaultdict(int)
        self._data_msgs:     dict[str, int]   = defaultdict(int)
        self._audio_bytes:   dict[str, int]   = defaultdict(int)
        self._join_time:     dict[str, float] = {}

        self._stats_task: asyncio.Task | None = None
        self._start_time = time.monotonic()

    # ── callbacks ─────────────────────────────────────────────────────────────

    async def _on_frame(self, signal: FrameSignal) -> None:
        pid = signal.participant_id
        self._video_frames[pid] += 1
        # Sample pixel data at ~1 fps (every 30th frame) to show request_frame usage.
        if self._video_frames[pid] % 30 == 0:
            frame = await self._ep.request_frame(signal)
            if frame:
                log.debug("frame sample: %s  %dx%d  %d bytes",
                          pid, frame.width, frame.height, len(frame.data))

    async def _on_audio(self, chunk: AudioChunk) -> None:
        pid = chunk.participant_id
        self._audio_chunks[pid] += 1
        self._audio_bytes[pid] += len(chunk.data)
        await self._ep.send_return_audio(chunk)

    async def _on_data(self, msg: DataMessage) -> None:
        pid = msg.participant_id
        self._data_msgs[pid] += 1
        log.info(
            "data from %r  topic=%r  bytes=%d",
            pid, msg.topic, len(msg.data),
        )

    async def _on_participant(self, event: ParticipantEvent) -> None:
        pid = event.participant_id
        if event.joined:
            log.info("participant joined: %r", pid)
            self._join_time[pid] = time.monotonic()
        else:
            log.info("participant left: %r", pid)
            self._join_time.pop(pid, None)
            self._video_frames.pop(pid, None)
            self._audio_chunks.pop(pid, None)
            self._data_msgs.pop(pid, None)
            self._audio_bytes.pop(pid, None)

    # ── stats loop ────────────────────────────────────────────────────────────

    async def _stats_loop(self) -> None:
        while True:
            await asyncio.sleep(_STATS_INTERVAL_S)
            now = time.monotonic()
            for pid in self._ep.connected_participants:
                joined_at = self._join_time.get(pid, now)
                stats = {
                    "agent": "echo-agent",
                    "participant": pid,
                    "video_frames_received": self._video_frames[pid],
                    "audio_chunks_received": self._audio_chunks[pid],
                    "data_msgs_received": self._data_msgs[pid],
                    "audio_bytes_received": self._audio_bytes[pid],
                    "uptime_s": round(now - joined_at, 1),
                }
                log.info("stats: %s", json.dumps(stats))
                await self._ep.send_return_data(
                    DataMessage(
                        participant_id=pid,
                        topic="agent.stats",
                        pts_us=_now_us(),
                        data=json.dumps(stats).encode(),
                    )
                )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._stats_task = asyncio.create_task(self._stats_loop(), name="stats-loop")
        await self._ep.run()

    def shutdown(self) -> None:
        if self._stats_task and not self._stats_task.done():
            self._stats_task.cancel()
        self._ep.stop()
        self._ep.close()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    async with HubLauncher():
        agent = EchoAgent()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, agent.shutdown)

        log.info("echo-agent starting  sub=%s  push=%s", _HUB_PUB, _HUB_PUSH)
        try:
            await agent.run()
        finally:
            agent.shutdown()

    log.info("echo-agent stopped")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
