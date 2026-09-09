.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Sample configuration reference
==============================

This reference is generated directly from configuration files under each
installed top-level sample's ``yaml/`` directory and from configuration files
beside a direct capability subproject. Adding a file in either location enrolls
it automatically. Values are sample- and hardware-specific examples, not
universal defaults. Keep field guidance beside the value as a source comment;
the generated page preserves those comments verbatim.

Refer to the generated
:doc:`Python API reference <python/xr_ai_models/index>` for public typed model
configuration fields. Operational choices, credentials, and deployment
workflows remain in the handwritten guides.

To change a sample parameter:

1. Start in ``agent-samples/<sample>/`` and find the owning file in that
   sample's README or published guide.
2. Edit the checked-in YAML or JSON value, preserving its documented type.
   Resolve relative paths from the file that declares them unless the sample
   guide documents different precedence.
3. Restart the sample process that owns the file. Sample configuration is not
   hot-reloaded.
4. If the edit changes a persistent model server, stop and restart the shared
   model stack. Refer to :doc:`/guides/customizing-model-servers` for that
   workflow.

.. xr-ai-config-reference::
