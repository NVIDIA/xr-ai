# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate transport-neutral interaction messages without the voice media stack."""

import pytest
from pydantic import ValidationError
from xr_ai_interaction import ResponseOutput


def test_response_output_rejects_ambiguous_empty_messages() -> None:
    with pytest.raises(ValidationError, match="response_id"):
        ResponseOutput(final=False)
    with pytest.raises(ValidationError, match="contain text"):
        ResponseOutput()
    with pytest.raises(ValidationError, match="cannot interrupt"):
        ResponseOutput(response_id="turn", interrupt=True)


def test_response_output_does_not_track_identifier_reuse() -> None:
    first = ResponseOutput(text="First response.", response_id="turn")
    second = ResponseOutput(text="Second response.", response_id="turn")
    assert first.text == "First response."
    assert second.text == "Second response."
