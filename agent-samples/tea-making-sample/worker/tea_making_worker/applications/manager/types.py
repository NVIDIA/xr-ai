# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing metadata for root-visible NAT functions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from nat.plugin_api import FunctionRef


class InvocationEffect(StrEnum):
    INLINE = "inline"
    FOREGROUND = "foreground"
    BACKGROUND = "background"


@dataclass(frozen=True, slots=True)
class RoutedFunction:
    name: str
    route: str
    effect: InvocationEffect = InvocationEffect.INLINE
    return_direct: bool = False

    @property
    def ref(self) -> FunctionRef:
        return FunctionRef(self.name)

    def catalog_entry(self) -> str:
        capture = "[foreground]" if self.effect == InvocationEffect.FOREGROUND else ""
        return f"{self.name}{capture}={self.route}"


__all__ = ["InvocationEffect", "RoutedFunction"]
