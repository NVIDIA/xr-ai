# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture narrated demonstrations as local evidence packets.

agent.py handles spoken commands and participant connections. capturing.py
samples images, obtains captions, and saves narration and packet summaries.
captioning.py supplies caption requests and validation; config.py loads the
capture settings. Shared infrastructure is supplied by the worker's app.py.
"""

from .agent import CaptureAgent

__all__ = ["CaptureAgent"]
