"""
Client-control MCP worker.

Connects to the XR-Media-Hub via ProcessorEndpoint and exposes FastMCP tools
for sending messages and control commands to XR clients over the data channel.

This is a full processor-endpoint worker — it mirrors participant join/leave
events from the hub so tools always reflect live connection state.

Tools
-----
  list_connected_participants()                   — all active participant IDs
  send_to_client(participant_id, topic, message)  — send arbitrary text to a client
  start_camera(participant_id)                    — request client to start camera
  stop_camera(participant_id)                     — request client to stop camera

Standalone usage
----------------
Run as a standalone server with its own ProcessorEndpoint:

    uv run client_control_mcp_server --config client_control_mcp_server.yaml

Config (client_control_mcp_server.yaml)
---------------------------------------
    hub_pub:  ipc:///tmp/xr_hub_pub   # hub PUB socket (subscribe)
    hub_push: ipc:///tmp/xr_hub_in    # hub PUSH socket (send)
    host:     0.0.0.0
    port:     8201

MCP endpoint (StreamableHTTP): http://localhost:8201/mcp

REST convenience endpoints (for non-MCP callers)
-------------------------------------------------
  GET  /health
  POST /send              {"participant_id","topic","message"}
  POST /camera/start      {"participant_id"}
  POST /camera/stop       {"participant_id"}

Embedded usage
--------------
Create a ``ProcessorEndpoint``, wrap it with ``ClientControlBridge``, then
mount the FastMCP skill.  The caller owns the endpoint lifecycle::

    ep = ProcessorEndpoint(sub_addr=hub_pub, push_addr=hub_push,
                           topics=(b"participant",))
    bridge = ClientControlBridge(ep)
    mcp.mount(build_mcp(bridge), namespace="client_control")
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from pydantic import BaseModel

from xr_ai_agent import DataMessage, ProcessorEndpoint

log = logging.getLogger("client_control_mcp")

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


def _now_us() -> int:
    return time.time_ns() // 1_000


# ── bridge ────────────────────────────────────────────────────────────────────

class ClientControlBridge:
    """
    Self-contained client-control worker with its own ``ProcessorEndpoint``.

    The hub connection lifecycle is managed by FastMCP via ``build_mcp()`` —
    nothing for the caller to wire up. Multiple independent instances can
    connect to the same hub simultaneously (ZMQ PUB/SUB fanout).
    """

    def __init__(
        self,
        hub_pub:  str = _HUB_PUB,
        hub_push: str = _HUB_PUSH,
    ) -> None:
        self._ep = ProcessorEndpoint(
            sub_addr=hub_pub, push_addr=hub_push,
            topics=(b"participant",),  # only need join/leave for connected_participants
        )

    @asynccontextmanager
    async def lifespan(self, _server: object = None) -> AsyncIterator[None]:
        task = asyncio.create_task(self._ep.run())
        try:
            yield
        finally:
            self._ep.stop()
            task.cancel()

    @property
    def connected_participants(self) -> frozenset[str]:
        return self._ep.connected_participants

    async def send(self, participant_id: str, topic: str, data: bytes) -> None:
        """Send raw bytes to *participant_id* on *topic*."""
        await self._ep.send_return_data(DataMessage(
            participant_id=participant_id,
            topic=topic,
            pts_us=_now_us(),
            data=data,
        ))

    async def send_text(self, participant_id: str, topic: str, message: str) -> None:
        """Send a UTF-8 string to *participant_id* on *topic*."""
        await self.send(participant_id, topic, message.encode())

    async def start_camera(self, participant_id: str) -> None:
        """Request *participant_id* to start their camera (Camera On Demand)."""
        await self.send(participant_id, "clientControl",
                        json.dumps({"action": "startCamera"}).encode())

    async def stop_camera(self, participant_id: str) -> None:
        """Request *participant_id* to stop their camera."""
        await self.send(participant_id, "clientControl",
                        json.dumps({"action": "stopCamera"}).encode())


# ── skill builder ─────────────────────────────────────────────────────────────

def build_mcp(bridge: ClientControlBridge) -> FastMCP:
    """Return a FastMCP with all client-control tools bound to *bridge*.

    The bridge's hub connection rides on the FastMCP lifespan, so no separate
    start/stop is needed by the caller — mounting this MCP (or running it
    standalone) is enough.
    """
    mcp = FastMCP("client-control-mcp", lifespan=bridge.lifespan)

    @mcp.tool()
    async def list_connected_participants() -> list[str]:
        """
        Return all participant IDs currently connected to the hub.

        Reflects live join/leave events — always up to date.
        """
        return sorted(bridge.connected_participants)

    @mcp.tool()
    async def send_to_client(
        participant_id: str,
        topic: str,
        message: str,
    ) -> dict:
        """
        Send a text message to a specific client on the given topic.

        The client receives the payload on its data channel. Use any topic
        that is meaningful to your application; reserved topics such as
        'clientControl' and '_agent.status' have special client-side handling.
        """
        await bridge.send_text(participant_id, topic, message)
        log.info("send  pid=%r  topic=%r  %d bytes", participant_id, topic, len(message))
        return {"ok": True, "participant_id": participant_id, "topic": topic}

    @mcp.tool()
    async def start_camera(participant_id: str) -> dict:
        """
        Request a client to start their camera.

        The client must have Camera On Demand mode enabled. Once it complies,
        a video track will begin publishing to the hub.
        Returns immediately — does not wait for the camera to start.
        """
        await bridge.start_camera(participant_id)
        log.info("start_camera  pid=%r", participant_id)
        return {"ok": True, "participant_id": participant_id, "action": "startCamera"}

    @mcp.tool()
    async def stop_camera(participant_id: str) -> dict:
        """
        Request a client to stop their camera.

        Returns immediately — does not wait for confirmation.
        """
        await bridge.stop_camera(participant_id)
        log.info("stop_camera  pid=%r", participant_id)
        return {"ok": True, "participant_id": participant_id, "action": "stopCamera"}

    return mcp


# ── standalone entry point ────────────────────────────────────────────────────

class SendRequest(BaseModel):
    participant_id: str
    topic:          str
    message:        str


class ParticipantRequest(BaseModel):
    participant_id: str


def build_app(bridge: ClientControlBridge) -> FastAPI:
    mcp_app = build_mcp(bridge).http_app(path="/")
    app = FastAPI(title="Client-Control MCP Server",
                  docs_url=None, redoc_url=None, lifespan=mcp_app.lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok",
                "participants": sorted(bridge.connected_participants)}

    @app.post("/send")
    async def send(req: SendRequest) -> JSONResponse:
        await bridge.send_text(req.participant_id, req.topic, req.message)
        return JSONResponse({"ok": True})

    @app.post("/camera/start")
    async def camera_start(req: ParticipantRequest) -> JSONResponse:
        await bridge.start_camera(req.participant_id)
        return JSONResponse({"ok": True})

    @app.post("/camera/stop")
    async def camera_stop(req: ParticipantRequest) -> JSONResponse:
        await bridge.stop_camera(req.participant_id)
        return JSONResponse({"ok": True})

    app.mount("/mcp", mcp_app)
    return app


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    hub_pub  = cfg.get("hub_pub",  _HUB_PUB).strip()
    hub_push = cfg.get("hub_push", _HUB_PUSH).strip()
    host     = cfg.get("host",     "0.0.0.0").strip()
    port     = int(cfg.get("port", 8201))

    bridge = ClientControlBridge(hub_pub=hub_pub, hub_push=hub_push)
    app    = build_app(bridge)

    log.info("client-control-mcp  mcp=http://%s:%d/mcp", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()
