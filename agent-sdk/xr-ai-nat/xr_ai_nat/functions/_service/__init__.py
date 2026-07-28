# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private correlated msgpack/ZMQ transport shared by service-backed functions.

Owns only the RPC transport (``rpc``); the shared value models live in the
capability-neutral :mod:`xr_ai_nat.functions.types`.
"""
