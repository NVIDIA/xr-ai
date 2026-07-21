# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire encoding for the private service RPC protocol."""

from typing import Any

import msgpack

PROTOCOL_VERSION = 1


class RPCError(RuntimeError):
    """A transport, protocol, or remote execution failure."""

    def __init__(self, message: str, *, code: str = "rpc_error") -> None:
        super().__init__(message)
        self.code = code


def pack(value: dict[str, Any]) -> bytes:
    encoded = msgpack.packb(value, use_bin_type=True)
    if not isinstance(encoded, bytes):
        raise RPCError("msgpack encoder returned no bytes", code="encoding_error")
    return encoded


def unpack(value: bytes) -> dict[str, Any]:
    decoded = msgpack.unpackb(value, raw=False)
    if not isinstance(decoded, dict):
        raise RPCError("RPC frame must contain a map", code="invalid_frame")
    return decoded
