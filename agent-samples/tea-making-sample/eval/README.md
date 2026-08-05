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

Corner-case probes cover atomic identification readiness, irrelevant retrieval
results, missing temperature units, unconfirmed immersion, a running timer, and
voice questions that require fresh vision or timer tools. These fixtures use
different labels and readings from the prompt rules so they remain behavioral
checks rather than worked-example recall. A repeated below-target reading also
guards against routine observation messages being spoken every cycle.
Identification probes include a visible variety paired with a different
retrieved variety; that mismatch must remain unready and commit no values.
The contract probe checks that prior completion status has a distinct name and
that field meanings and completion values remain visible to the small model.
The progress probe requires a brief message for a real non-completing state
change while the repeated-reading probe remains silent. Retrieval chunk
settings are pinned to the compact original configuration.
