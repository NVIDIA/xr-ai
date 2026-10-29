.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Command-line reference
======================

Overview
--------

Adapting a sample to your own application starts with being able to run it
repeatably. These entry points let you launch a working baseline and select
the sample's supported modes as you try the interaction you want to build on.

Sample commands
---------------

This reference is generated from the installed commands in each top-level
sample's ``pyproject.toml`` under ``agent-samples/``,
and the literal ``argparse`` declarations in its
entry-point module. Worker and service commands are launcher-managed process
protocols, so they are intentionally excluded from the user-facing catalog.

.. xr-ai-cli-reference::
