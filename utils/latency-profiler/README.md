<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-latency-profiler

Benchmarks the model services a stack is serving, the `AgentRouter` paths, and streaming choices. It fits the
measurements into a model that prices common agent patterns, and it compares the same stack across machines.

Refer to the [latency profiling guide](../../docs/source/guides/latency-profiling.md) for commands and how to read
the report.

- `xr_latency/bench/` holds one module per benchmark; `xr_latency/suites.py` registers them and defines the
  `quick` and `full` suites.
- `xr_latency/fit.py` turns raw samples into `model.json`; `estimate.py` prices stages from it, and
  `xr_latency/report/calc.js` is its in-page port (`tests/test_parity.py` keeps the two equal).
- `patterns/` holds the common turn shapes the estimator prices. Samples carry no profiling code: components
  are measured through the SDK, and a sample's latency is read off the patterns it resembles.
- `stacks/` holds model profiles. `docs/cost-cards/` holds static, code-read analyses; they are estimates.
