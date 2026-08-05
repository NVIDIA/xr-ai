<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Guidance evals

`cases.yaml` covers each router action and every step policy with facts that do
not appear as worked examples in the prompts. `check.py` validates schema,
coverage, and prompt budgets without model servers. During a model-backed human
test, use the same cases as probes and compare the logged tool call or commit to
the expected fields. Visual fixtures are ordinary captions, including one
negative water-visibility case that must fail the deterministic evidence gate.
Routing cases distinguish explicit workflow management from task questions,
action reports, current readings, and timer questions.
