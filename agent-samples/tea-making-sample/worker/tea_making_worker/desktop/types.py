# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing metadata shared by every root-visible NAT function."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from nat.plugin_api import FunctionRef


class FunctionEffect(StrEnum):
    INLINE = "inline"
    FOREGROUND = "foreground"
    BACKGROUND = "background"


@dataclass(frozen=True, slots=True)
class RoutedFunction:
    name: str
    route: str
    effect: FunctionEffect = FunctionEffect.INLINE
    return_direct: bool = False

    @property
    def ref(self) -> FunctionRef:
        return FunctionRef(self.name)

    def catalog_entry(self) -> str:
        return f"{self.name}[{self.effect}]={self.route}"


__all__ = ["FunctionEffect", "RoutedFunction"]
