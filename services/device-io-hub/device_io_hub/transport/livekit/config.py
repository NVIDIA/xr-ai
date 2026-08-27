# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the LiveKit connector."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_RETURN_AUDIO_MAX_BUFFER_S = 3.0


def _validate_return_audio_max_buffer_s(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(
            "return_audio_max_buffer_s must be a finite number greater than 0, "
            f"got {value!r}"
        )
    try:
        max_buffer_s = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "return_audio_max_buffer_s must be a finite number greater than 0, "
            f"got {value!r}"
        ) from exc
    if not math.isfinite(max_buffer_s) or max_buffer_s <= 0:
        raise ValueError(
            "return_audio_max_buffer_s must be a finite number greater than 0, "
            f"got {value!r}"
        )
    return max_buffer_s


@dataclass
class LiveKitConnectorConfig:
    # ── LiveKit server credentials ────────────────────────────────────────────
    api_key:    str
    api_secret: str
    room_name:  str = "xr-room"

    # ── LiveKit server ports (used by docker and room client) ─────────────────
    lk_port_ws:  int = 7880   # signaling WebSocket
    lk_port_tcp: int = 7881   # WebRTC TCP
    lk_port_udp: int = 7882   # WebRTC UDP

    # Discover and advertise the host's public IP for clients outside a NAT.
    lk_use_external_ip: bool = False
    # Some cloud NATs do not support the self-ping LiveKit uses to validate the
    # discovered IP. This setting has an effect only with external IP enabled.
    lk_skip_external_ip_validation: bool = False

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
    # Public root CA returned by /cert for externally supplied certificates.
    # Private keys are never returned by the web server.
    root_ca_file: str = ""
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
    # Absolute path to the web client directory. Set via device_io_hub.yaml.
    web_client_dir:    str  = ""
    # HTTPS is on by default — required for camera access from any device that
    # isn't localhost, and required so the same-origin /rtc proxy can carry
    # LiveKit signaling as wss:// without browser mixed-content blocks.
    # A development root CA and signed server leaf are auto-generated in
    # ~/.local/share/xr-ai/ on first run. Supply cert_file/key_file to use your
    # own, and optionally root_ca_file to make its public root available at
    # /cert for client installation.
    # Set to False for the two cases where the hub should *not* terminate TLS
    # itself: (a) a TLS-terminating reverse proxy (nginx, Caddy, Cloudflare
    # Tunnel) sits in front and speaks plain http:// + ws:// to the hub on the
    # loopback; (b) localhost-only dev where browsers grant camera/mic on
    # http://localhost and the cert dance adds friction with no benefit.
    web_server_tls:    bool = True
    # Extra hostnames/IPs added to the auto-generated cert's SAN: addresses
    # clients dial that are on no local interface (a NAT'd cloud VM's public
    # IP, a forwarding proxy's address, or a DNS name).
    web_server_extra_sans: list[str] = field(default_factory=list)

    # ── Shared-memory ring buffer ──────────────────────────────────────────────
    shm_num_slots:       int = 10
    shm_max_frame_bytes: int = 12_441_600   # 4K NV12

    # ── Return audio pacing ───────────────────────────────────────────────────
    # Maximum queued TTS audio duration per participant. The oldest queued
    # frames are dropped when a producer exceeds this hard bound.
    return_audio_max_buffer_s: float = _DEFAULT_RETURN_AUDIO_MAX_BUFFER_S
    """Maximum seconds of queued return audio retained per participant."""

    # ── Video recording (NVENC, optional) ─────────────────────────────────────
    # Set video_recording.enabled: true in device_io_hub.yaml to activate.
    # Frames are encoded via NVENC (pynvvideocodec) and written as H.264
    # Annex B chunks to video_recording.out_dir.
    video_recording: Any = field(default=None)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.api_key, str)
            or not self.api_key.strip()
            or not isinstance(self.api_secret, str)
            or not self.api_secret.strip()
        ):
            raise ValueError(
                "LiveKit credentials are required; set non-empty api_key and "
                "api_secret values in device_io_hub.yaml, or set "
                "LIVEKIT_API_KEY and LIVEKIT_API_SECRET"
            )
        self.return_audio_max_buffer_s = _validate_return_audio_max_buffer_s(
            self.return_audio_max_buffer_s
        )
