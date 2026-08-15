# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for LiveKit client data ingress."""

from xr_media_hub.transport.livekit._room_client import _normalize_client_topic


def test_untopiced_client_data_uses_private_text_topic() -> None:
    assert _normalize_client_topic(None) == "_client.text"
    assert _normalize_client_topic("") == "_client.text"
    assert _normalize_client_topic("xr.session.started") == "xr.session.started"
