# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared exception types for hub startup.

``StartupError`` is raised by pre-flight checks and lifecycle code that
fails with a *user-facing* problem the operator must fix (missing system
library, daemon not reachable, container exited early, etc.). Its
``str()`` is a pre-formatted banner intended for direct display.

The entry point catches it and exits the process cleanly — no traceback
— so the operator sees only the banner. Anything that actually warrants
a traceback (programming error, unexpected internal state) should raise
a different exception type.
"""
from __future__ import annotations


class StartupError(RuntimeError):
    """Pre-formatted, user-facing startup failure. Caught at the entry point."""
