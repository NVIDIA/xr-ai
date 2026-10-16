# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public text payload used by named conversation-context topics."""

from pydantic import BaseModel, ConfigDict, Field


class ConversationContextUpdate(BaseModel):
    """Text published on a configured conversation-context topic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    """Non-empty source-owned context text; handling comes from topic configuration."""


__all__ = ["ConversationContextUpdate"]
