# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Entry point for xr_media_hub.

    uv run xr_media_hub          # via pyproject.toml script
    python -m xr_media_hub       # direct module invocation
"""
from __future__ import annotations

import asyncio
import collections
import logging
import signal
import time

from xr_media_hub._config_loader import load_config
from xr_media_hub.ipc import AudioChunk, DataMessage, HubEndpoint, ParticipantEvent, SlotView
from xr_media_hub.transport.livekit import LiveKitConnector, make_client_token

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PULL_ADDR      = "ipc:///tmp/xr_hub_in"
PUB_ADDR       = "ipc:///tmp/xr_hub_pub"
STATS_INTERVAL = 5.0

_frame_counts: dict[str, int] = collections.defaultdict(int)
_audio_counts: dict[str, int] = collections.defaultdict(int)
_data_counts:  dict[str, int] = collections.defaultdict(int)
_participants: set[str] = set()

_recorder = None


async def on_frame(view: SlotView) -> None:
    _frame_counts[view.signal.participant_id] += 1
    if _recorder is not None:
        await _recorder.on_frame(view)


async def on_audio(chunk: AudioChunk) -> None:
    _audio_counts[chunk.participant_id] += 1


async def on_data(msg: DataMessage) -> None:
    text = msg.data.decode("utf-8", errors="replace") if msg.data else ""
    log.info("data  participant=%s  topic=%r  %r", msg.participant_id, msg.topic, text[:120])
    _data_counts[msg.participant_id] += 1


async def on_participant(event: ParticipantEvent) -> None:
    action = "joined" if event.joined else "left"
    log.info("participant %s %s", event.participant_id, action)
    if event.joined:
        _participants.add(event.participant_id)
    else:
        _participants.discard(event.participant_id)
        _frame_counts.pop(event.participant_id, None)
        _audio_counts.pop(event.participant_id, None)
        _data_counts.pop(event.participant_id, None)
        if _recorder is not None:
            _recorder.close_participant(event.participant_id)


async def _stats_loop() -> None:
    while True:
        await asyncio.sleep(STATS_INTERVAL)
        if not _participants:
            continue
        parts = []
        for pid in sorted(_participants):
            fps   = _frame_counts.pop(pid, 0) / STATS_INTERVAL
            achps = _audio_counts.pop(pid, 0) / STATS_INTERVAL
            dps   = _data_counts.pop(pid, 0)  / STATS_INTERVAL
            parts.append(f"{pid}  video={fps:.1f}fps  audio={achps:.1f}ch/s  data={dps:.1f}msg/s")
        log.info("stats ─ %s", " │ ".join(parts))


async def main() -> None:
    global _recorder

    hub = HubEndpoint(pull_addr=PULL_ADDR, pub_addr=PUB_ADDR)
    hub.on_frame(on_frame)
    hub.on_audio(on_audio)
    hub.on_data(on_data)
    hub.on_participant(on_participant)

    cfg = load_config()
    connector = LiveKitConnector(cfg)
    await connector.start()

    vr_cfg = cfg.video_recording or {}
    if vr_cfg.get("enabled"):
        from xr_media_hub.video import VideoRecorder, VideoRecorderConfig
        rc_defaults = VideoRecorderConfig()
        rc = VideoRecorderConfig(
            out_dir         = vr_cfg.get("out_dir",         rc_defaults.out_dir),
            chunk_frames    = int(vr_cfg.get("chunk_frames",    rc_defaults.chunk_frames)),
            max_total_bytes = int(vr_cfg.get("max_total_bytes", rc_defaults.max_total_bytes)),
            sample_fps      = float(vr_cfg.get("sample_fps",    rc_defaults.sample_fps)),
            bitrate         = int(vr_cfg.get("bitrate",         rc_defaults.bitrate)),
            gpu_id          = int(vr_cfg.get("gpu_id",          rc_defaults.gpu_id)),
        )
        _recorder = VideoRecorder(rc)
        log.info("Video recording enabled  out_dir=%s", rc.out_dir)

    token = make_client_token(cfg, identity="ios-client")
    web_scheme = "https" if cfg.web_server_tls else "http"
    print(f"\n  LiveKit URL : ws://0.0.0.0:{cfg.lk_port_ws}  (plain ws — no TLS)", flush=True)
    print(f"  Room        : {cfg.room_name}", flush=True)
    print(f"  Token       : {token}", flush=True)
    if cfg.enable_web_server:
        print(f"  Web client  : {web_scheme}://localhost:{cfg.web_server_port}", flush=True)
    if _recorder is not None:
        print(f"  Recording   : {rc.out_dir}", flush=True)
    print(flush=True)

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    log.info("XR-Media-Hub running — press Ctrl-C to exit")
    hub_task   = asyncio.create_task(hub.run(),       name="hub")
    conn_task  = asyncio.create_task(connector.run(), name="connector")
    stats_task = asyncio.create_task(_stats_loop(),   name="stats")

    await stop.wait()
    log.info("Shutting down…")

    stats_task.cancel()
    hub.stop()
    hub.close()
    await connector.stop()

    await asyncio.gather(hub_task, conn_task, stats_task, return_exceptions=True)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
