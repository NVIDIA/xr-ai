# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr-render-demo worker — voice-driven XR scene control.

``app`` owns process startup and pipeline assembly; ``xr_session`` owns the XR /
LOVR session lifecycle; ``scene`` builds the per-turn scene context;
``processors`` runs the agentic turn; ``capabilities`` assembles the native NAT
toolbox. The console-script entry point is ``xr_render_demo_worker.__main__:run``.

This module stays import-light on purpose: it declares the bundled prompt paths
and imports no submodule. Importing one seam (``scene``, say) must not drag in
Pipecat, NAT, and the model clients — otherwise the seams are not actually
separable and cannot be exercised on their own.
"""

from pathlib import Path

#: Prompts shipped inside the package. Consumers — the worker, the eval harness,
#: tests — resolve them from here instead of re-deriving a relative path, so a
#: future relocation cannot leave a stale literal behind.
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT = PROMPTS_DIR / "system.txt"

__all__ = ["PROMPTS_DIR", "SYSTEM_PROMPT"]
