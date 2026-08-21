# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn-scoped trace id shared between the supervisor and its subagents.

A context variable, not a SubagentTask field: the task schema is filled by
the supervisor LLM, which must not be trusted to copy identifiers.
"""

from contextvars import ContextVar

current_trace_id: ContextVar[str] = ContextVar("current_trace_id", default="")
