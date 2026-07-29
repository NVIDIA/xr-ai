# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allow ``python -m xr_render_demo_worker`` alongside the console script."""

from .app import run

if __name__ == "__main__":
    run()
