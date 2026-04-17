"""
xr_media_hub.transport.livekit — LiveKit connector for XR-Media-Hub.

The LiveKit connector is an internal transport implementation detail.
It is not exposed to the agent or MCP layer — only to the server-runtime.

Quick start
-----------
    from xr_media_hub.transport.livekit import LiveKitConnector, LiveKitConnectorConfig

    cfg  = LiveKitConnectorConfig(room_name="xr-room", api_key="...", api_secret="...")
    conn = LiveKitConnector(cfg)
    await conn.start()
    try:
        await conn.run()
    finally:
        await conn.stop()
"""

from .config import LiveKitConnectorConfig
from .connector import LiveKitConnector
from ._token import make_client_token

__all__ = ["LiveKitConnector", "LiveKitConnectorConfig", "make_client_token"]
