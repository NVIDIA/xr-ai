<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# XR AI Experiences

Run and switch XR AI samples through a web UI on one Linux GPU host. Unlike a
direct sample command, this launcher manages background lifecycle, model stacks,
credentials, logs, and client-facing network configuration.

Recommended for cloud use:
[Launch XR AI Experiences on Brev](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3I1iZdrqmkxKSgsIsMXvqu8xrp7).

Developers can run the same entry point on a prepared local GPU host to exercise
the single-host deployment workflow. Normal setup does not change system settings
or reboot. Follow the local quickstart before running `./deployments/brev/launch.sh setup`.

Refer to [Deploying on Brev or your own host](https://nvidia.github.io/xr-ai/latest/guides/deploying-on-brev.html)
for prerequisites, authentication, release selection, and networking.
