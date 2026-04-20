"""
Entry point for xr_media_hub.

    uv run xr_media_hub          # via pyproject.toml script
    python -m xr_media_hub       # direct module invocation
"""
from __future__ import annotations

import asyncio
import logging
import signal

from xr_media_hub._config_loader import load_config
from xr_media_hub.ipc import AudioChunk, HubEndpoint, ParticipantEvent, SlotView
from xr_media_hub.transport.livekit import LiveKitConnector, make_client_token

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

    cfg = load_config()
    connector = LiveKitConnector(cfg)
    await connector.start()

    token = make_client_token(cfg, identity="ios-client")
    print(f"\n  LiveKit URL : ws://0.0.0.0:{cfg.lk_port_ws}")
    print(f"  Room        : {cfg.room_name}")
    print(f"  Token       : {token}")
    if cfg.enable_web_server:
        print(f"  Web client  : http://localhost:{cfg.web_server_port}")
    print()

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    log.info("XR-Media-Hub running — press Ctrl-C to exit")
    hub_task  = asyncio.create_task(hub.run(),       name="hub")
    conn_task = asyncio.create_task(connector.run(), name="connector")

    await stop.wait()
    log.info("Shutting down…")

    hub.stop()
    hub.close()
    await connector.stop()

    await asyncio.gather(hub_task, conn_task, return_exceptions=True)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
