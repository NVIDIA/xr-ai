# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker application for capturing demonstrations and guiding approved procedures.

Start with __main__.py for the process entry point, then app.py for shared
infrastructure and agent wiring. The capture package owns demonstration
commands, snapshots, captions, and saved evidence packets. The sibling guide
package owns guide selection, spoken guidance, and step verification.
"""
