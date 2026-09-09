# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared policy for release tags published in the documentation site."""
from __future__ import annotations

import os
import re

_WITHDRAWN_TAGS_ENV = "XR_AI_DOCS_WITHDRAWN_TAGS"


def withdrawn_tags() -> frozenset[str]:
    """Return release tags excluded from documentation publication."""
    value = os.environ.get(_WITHDRAWN_TAGS_ENV, "")
    return frozenset(filter(None, re.split(r"[\s,]+", value)))


def tag_whitelist(semver_pattern: str) -> str:
    """Return an anchored semantic-version pattern excluding withdrawn tags."""
    withdrawn = "|".join(re.escape(tag) for tag in sorted(withdrawn_tags()))
    exclusion = rf"(?!(?:{withdrawn})$)" if withdrawn else ""
    return rf"^{exclusion}{semver_pattern}$"
