# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the execution contract shared by procedure verification flows.

A flow performs one bounded observation at a time and returns typed evidence.
The step runner owns scheduling and task lifetime; flows never advance a guide
or publish speech. Unavailable evidence is distinct from a negative judgment.
"""

from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FlowUpdate(Generic[T]):
    status: Literal["pending", "ready", "unavailable"]
    value: T
    delay_s: float


class Flow(Protocol[T]):
    async def poll(self) -> FlowUpdate[T]:
        """Observe once; await all child work and propagate cancellation/errors."""
        ...
