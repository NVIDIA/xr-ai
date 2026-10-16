# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-record definition used within ``xr_ai_conversation``.

Other packages import the public record from the package root.
"""

from pydantic import BaseModel, ConfigDict, Field


class ConversationContextUpdate(BaseModel):
    """Text published on a configured conversation-context topic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    """Non-empty source-owned context text; handling comes from topic configuration."""
