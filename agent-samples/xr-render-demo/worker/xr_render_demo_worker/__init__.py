# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr-render-demo worker — voice-driven XR scene control.

``app`` owns process startup and pipeline assembly; ``xr_session`` owns the XR /
LOVR session lifecycle; ``scene`` builds the per-turn scene context;
``processors`` runs the agentic turn; ``capabilities`` assembles the native NAT
toolbox. The console script entry point is :func:`run`.
"""

from .app import main, run

__all__ = ["main", "run"]
