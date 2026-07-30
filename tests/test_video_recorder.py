# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from xr_media_hub.video._recorder import _append_encoded_packets


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            [
                {"data": b"\x02", "timestamp": 0, "picture_type": 1},
                {"data": b"\x03\x04", "timestamp": 1, "picture_type": 2},
            ],
            b"\x02\x03\x04",
        ),
        ([], b""),
    ],
)
def test_append_encoded_packets(
    output: list[dict[str, object]],
    expected: bytes,
) -> None:
    buffer = bytearray(b"\xff")

    _append_encoded_packets(buffer, output)

    assert buffer == b"\xff" + expected


@pytest.mark.parametrize(
    "output",
    [
        b"pre-2.2-byte-output",
        [b"not-a-packet"],
        [{"timestamp": 0, "picture_type": 1}],
        [{"data": "not-bytes", "timestamp": 0, "picture_type": 1}],
    ],
)
def test_append_encoded_packets_rejects_unknown_contracts(output: object) -> None:
    with pytest.raises(TypeError, match="unexpected PyNvVideoCodec"):
        _append_encoded_packets(bytearray(), output)
