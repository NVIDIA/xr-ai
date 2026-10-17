# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay approved procedure guides with spoken guidance and evidence checks.

agent.py handles conversation and participant connections. replaying.py tracks
progress and verifies steps using camera observations and timers. guides.py
selects the approved guide, _workflow_spec.py validates its structure, and
config.py loads replay settings. Shared voice and image services come from
the worker's app.py.
"""
