# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token generation helpers — used by both the room client and external callers."""
from __future__ import annotations

from datetime import timedelta

from livekit.api import AccessToken, VideoGrants

from .config import LiveKitConnectorConfig


def make_client_token(
    cfg: LiveKitConnectorConfig,
    identity: str = "client",
    ttl: int = 3600 * 24,   # 24 h — long enough for dev
) -> str:
    """
    Generate a signed LiveKit JWT for a browser or mobile client.

    On a local/HTTP network pass this token directly to the livekit-client SDK
    along with ws://<host>:<lk_port_ws> — no token server needed.

        token = make_client_token(cfg, identity="alice")
        # hand token + ws://10.x.x.x:7880 to the browser client
    """
    return (
        AccessToken(cfg.api_key, cfg.api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(room_join=True, room=cfg.room_name))
        .with_ttl(timedelta(seconds=ttl))
        .to_jwt()
    )
