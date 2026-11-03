# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accumulate distinct visual evidence under an explicit guide confirmation policy.

The backend selects and deduplicates frames. This component owns only the
bounded evidence window; it cannot commit progress or execute controls.
"""

from collections import deque
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfirmationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    required: int = Field(gt=0)
    window: int = Field(gt=0, le=100)
    max_age_s: float = Field(gt=0)
    reset_on_negative: bool = True

    @model_validator(mode="after")
    def bounds(self):
        if self.required > self.window:
            raise ValueError("confirmation.required must fit inside confirmation.window")
        return self


@dataclass
class ConfirmationWindow:
    observations: deque = field(default_factory=deque)

    def observe(self, verdict, now, policy):
        while self.observations and now - self.observations[0][0] > policy.max_age_s:
            self.observations.popleft()
        if verdict is None or (verdict == "rejected" and policy.reset_on_negative):
            self.observations.clear()
        if verdict in {"accepted", "rejected"}:
            self.observations.append((now, verdict))
        while len(self.observations) > policy.window:
            self.observations.popleft()
        return sum(value == "accepted" for _, value in self.observations)
