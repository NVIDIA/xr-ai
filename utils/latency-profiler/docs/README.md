<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Static analyses

`cost-cards/` holds one card per SDK component, written by reading the source on `origin/main` at `ee2682be9`
rather than by running anything. Every millisecond figure in them is code-derived or a guess unless it says
otherwise. Where a benchmark later contradicted a card, the card carries a "Measured correction" note at the top.
The generated report is the source of truth for measured numbers; the cards are most useful for call structure,
fixed waits and timers, and cost drivers.
