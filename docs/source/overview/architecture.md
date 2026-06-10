<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Architecture

XR-AI follows a **one hub, many clients, many agents** model. The
**XR-Media-Hub** (server runtime) fans each client's inbound media to every
agent and routes return traffic back to the originating client only. Clients
reach the hub over a transport (LiveKit internally), agents attach over an IPC
boundary, and MCP servers are the agent's only interface to XR data and
rendering.

This page is a placeholder seeded by the docs scaffold; the full architecture
write-up is ported from the repository's existing architecture notes.
