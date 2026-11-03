# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate transport-neutral interaction messages without the voice media stack."""

import pytest
from pydantic import ValidationError
from xr_ai_interaction import ResponseOutput


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"kind": "reply", "request_id": "r1", "final": False}, "response_id"),
        ({"kind": "reply", "request_id": "r1"}, "contain text"),
        ({"kind": "update"}, "contain text"),
        ({"kind": "reply", "text": "hi"}, "needs a request_id"),
        ({"kind": "update", "text": "hi", "request_id": "r1"}, "cannot have a request_id"),
        ({"kind": "update", "text": "hi", "response_id": "u1"}, "must be finite"),
        (
            {"kind": "update", "text": "hi", "response_id": "u1", "final": False},
            "must be finite",
        ),
        ({"kind": "reply", "text": "hi", "request_id": "r1", "urgent": True}, "only an update"),
        ({"kind": "withdraw", "text": "hi"}, "carries no text"),
        ({"kind": "withdraw", "request_id": "r1"}, "carries no text"),
        ({"kind": "withdraw", "response_id": "s"}, "carries no text"),
        ({"kind": "withdraw", "urgent": True}, "final and not urgent"),
        ({"text": "hi"}, "kind"),
        ({"kind": "reply", "text": "hi", "request_id": ""}, "request_id"),
    ],
)
def test_response_output_validation(fields: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ResponseOutput(**fields)


def test_response_output_accepts_each_kind() -> None:
    ResponseOutput(kind="update", text="hi", urgent=True)
    ResponseOutput(kind="reply", text="hi", request_id="r1")
    ResponseOutput(kind="reply", text="hi", request_id="r1", response_id="s", final=False)
    ResponseOutput(kind="reply", request_id="r1", response_id="s")
    ResponseOutput(kind="withdraw")
    with pytest.raises(ValidationError, match="interrupt"):
        ResponseOutput(kind="update", text="hi", interrupt=True)  # type: ignore[call-arg]


def test_response_output_does_not_track_identifier_reuse() -> None:
    first = ResponseOutput(kind="reply", text="First response.", request_id="r1", response_id="turn")
    second = ResponseOutput(kind="reply", text="Second response.", request_id="r1", response_id="turn")
    assert first.text == "First response."
    assert second.text == "Second response."
