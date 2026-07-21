# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private model defaults shared by native function contracts."""

from pydantic import BaseModel, ConfigDict


class _StrictRequest(BaseModel):
    """Reject unknown arguments instead of silently discarding them."""

    model_config = ConfigDict(extra="forbid")


__all__: list[str] = []
