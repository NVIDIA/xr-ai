# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent Python async flows; callbacks report state to their caller.

Call a function directly for sequential execution, or schedule that same
coroutine in a caller-owned task group for concurrent/spanning work. Every
argument is supplied at call time; no flow loads YAML or connects to an agent.
"""

from ._identity import (
    IdentificationObservation,
    IdentificationState,
    SettingSource,
    run_identification_flow,
)
from ._timer import TimerState, run_timer_flow
from ._visual import VisualObservation, VisualVerificationState, run_visual_verification_flow

__all__ = [
    "IdentificationObservation",
    "IdentificationState",
    "SettingSource",
    "TimerState",
    "VisualObservation",
    "VisualVerificationState",
    "run_identification_flow",
    "run_timer_flow",
    "run_visual_verification_flow",
]
