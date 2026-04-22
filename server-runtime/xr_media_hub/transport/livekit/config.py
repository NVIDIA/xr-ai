"""Configuration for the LiveKit connector."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LiveKitConnectorConfig:
    # ── LiveKit server credentials ────────────────────────────────────────────
    api_key:    str = "devkey"
    api_secret: str = "devsecret-xr-livekit-prototype-2026"
    room_name:  str = "xr-room"

    # ── LiveKit server ports (used by docker and room client) ─────────────────
    lk_port_ws:  int = 7880   # signaling WebSocket
    lk_port_tcp: int = 7881   # WebRTC TCP
    lk_port_udp: int = 7882   # WebRTC UDP

    # ── Internal URL for the Python room client (direct WS, no proxy) ─────────
    lk_internal_url: str = "ws://127.0.0.1:7880"

    # ── Identity used when the connector joins the room ────────────────────────
    identity: str = "xr-hub-connector"

    # ── Token server (browser-facing HTTPS proxy) ─────────────────────────────
    token_server_host: str = "0.0.0.0"
    token_server_port: int = 8000
    # URL returned in token responses so the browser knows where to connect.
    token_server_url:  str = "ws://localhost:8000"
    # Leave empty for plain HTTP (camera blocked on remote without HTTPS).
    cert_file: str = ""
    key_file:  str = ""
    # Absolute path to browser static files. Empty = no static serving.
    browser_dir: str = ""

    # ── Token server (opt-in, only needed for HTTPS browser clients) ──────────
    # On a local/HTTP network clients connect directly to ws://<host>:lk_port_ws
    # using a pre-generated token — no proxy needed.
    enable_token_server: bool = False

    # ── IPC hub ZMQ addresses ─────────────────────────────────────────────────
    hub_push_addr: str = "ipc:///tmp/xr_hub_in"
    hub_sub_addr:  str = "ipc:///tmp/xr_hub_pub"

    # ── Web server (serves a static web client + /token endpoint) ────────────
    enable_web_server: bool = False
    web_server_host:   str  = "0.0.0.0"
    web_server_port:   int  = 8080
    # Absolute path to the web client directory. Set via xr_media_hub.yaml.
    web_client_dir:    str  = ""
    # Enable HTTPS. A self-signed cert is auto-generated in
    # ~/.local/share/xr-ai/ on first run. Supply cert_file/key_file to use
    # your own certificate instead.
    web_server_tls:    bool = False

    # ── Shared-memory ring buffer ──────────────────────────────────────────────
    shm_num_slots:       int = 10
    shm_max_frame_bytes: int = 12_441_600   # 4K NV12
