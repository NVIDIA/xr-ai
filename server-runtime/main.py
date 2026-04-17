"""XR-Media-Hub entry point (stub)."""
from __future__ import annotations

import asyncio
import logging

from xr_media_hub.ipc import HubEndpoint, PixelFormat, SlotView, AudioChunk

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SHM_NAME        = "xr_hub_frames"
NUM_SLOTS       = 10
MAX_FRAME_BYTES = 12_441_600  # 4K NV12 worst-case
PULL_ADDR       = "ipc:///tmp/xr_hub_in"
PUB_ADDR        = "ipc:///tmp/xr_hub_pub"


async def on_frame(view: SlotView) -> None:
    sig = view.signal
    log.info("frame  seq=%d  %dx%d  fmt=%s  pts=%d",
             sig.seq, sig.width, sig.height, sig.fmt.name, sig.pts_us)
    # TODO: upload view.data to GPU (CuPy H2D), feed rolling window.


async def on_audio(chunk: AudioChunk) -> None:
    log.info("audio  pts=%d  %dHz  %dch  %d samples",
             chunk.pts_us, chunk.sample_rate, chunk.channels, chunk.samples)
    # TODO: feed real-time audio pipeline.


async def main() -> None:
    hub = HubEndpoint(
        shm_name=SHM_NAME,
        num_slots=NUM_SLOTS,
        max_frame_bytes=MAX_FRAME_BYTES,
        pull_addr=PULL_ADDR,
        pub_addr=PUB_ADDR,
    )
    hub.on_frame(on_frame)
    hub.on_audio(on_audio)

    log.info("XR-Media-Hub listening on %s", PULL_ADDR)
    try:
        await hub.run()
    finally:
        hub.close()
        hub.unlink()


if __name__ == "__main__":
    asyncio.run(main())
