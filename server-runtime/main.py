"""XR-Media-Hub entry point (stub)."""
from __future__ import annotations

import asyncio
import logging

from xr_media_hub.ipc import HubEndpoint, ParticipantEvent, SlotView, AudioChunk

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PULL_ADDR = "ipc:///tmp/xr_hub_in"
PUB_ADDR  = "ipc:///tmp/xr_hub_pub"


async def on_frame(view: SlotView) -> None:
    sig = view.signal
    log.info("frame  participant=%s  track=%s  seq=%d  %dx%d  fmt=%s",
             sig.participant_id, sig.track_id, sig.seq, sig.width, sig.height, sig.fmt.name)
    # TODO: upload view.data to GPU (CuPy H2D), feed rolling window.


async def on_audio(chunk: AudioChunk) -> None:
    log.info("audio  participant=%s  track=%s  pts=%d  %dHz  %dch",
             chunk.participant_id, chunk.track_id, chunk.pts_us, chunk.sample_rate, chunk.channels)
    # TODO: feed real-time audio pipeline.


async def on_participant(event: ParticipantEvent) -> None:
    action = "joined" if event.joined else "left"
    log.info("participant %s %s (connector=%s)", event.participant_id, action, event.connector_id)


async def main() -> None:
    hub = HubEndpoint(pull_addr=PULL_ADDR, pub_addr=PUB_ADDR)
    hub.on_frame(on_frame)
    hub.on_audio(on_audio)
    hub.on_participant(on_participant)

    log.info("XR-Media-Hub listening on %s", PULL_ADDR)
    try:
        await hub.run()
    finally:
        hub.close()


if __name__ == "__main__":
    asyncio.run(main())
