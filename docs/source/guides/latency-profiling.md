<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Profiling and estimating latency

`utils/latency-profiler` measures the model services a stack is serving, the
`AgentRouter` paths, and the streaming choices an agent can make. It fits those
measurements into a small model that prices a planned agent before you build it,
and it renders everything as one self-contained HTML report.

The profiler calls services only through the typed `xr_ai_models` factories, so
numbers include the SDK's request building and streaming overhead. It never
starts, restarts, or reconfigures a model server. Start the stack with its
sample orchestrator first, then point the profiler at it.

## Quick start

```bash
cd utils/latency-profiler
uv sync --extra router --extra speech --extra vad
uv run xr-latency probe --stack my-stack          # health check only; sends no inference
uv run xr-latency run quick --stack my-stack      # small sample counts; every estimator term
uv run xr-latency report results/*-quick-* --out report/
uv run xr-latency patterns --model report/model.json
```

A stack is a models profile in `stacks/<name>.json`, the same shape the samples use in `yaml/models.json`, plus an
optional top-level `services` map (`speaker_stt` is the speaker-ASR URL). The LLM benchmark expects an extra role
named `llm_raw` with the same endpoint as `llm` but no `max_tokens`, `temperature`, or reasoning defaults. The
section on profile defaults below explains why. A benchmark whose role or server is missing is skipped, so a
partial stack still produces a partial model.

Each run writes `results/<timestamp>-<label>/samples.jsonl` and a `meta.json` with a machine fingerprint (CPU,
memory, GPUs and driver, served model ids, git SHA). `run full` uses each benchmark's own default plan and adds the
router acknowledgment race, VAD, and contention.

### Profiling from another machine

`--host` replaces `localhost` in the profile, so a laptop can profile a stack served by a Spark, though network
latency is then part of every number. For hardware comparisons run the profiler on the machine that serves the
models.

```bash
uv run xr-latency run quick --stack my-stack --label spark
uv run xr-latency report results/*-spark-* --out report-spark/
uv run xr-latency compare report-a/model.json report-spark/model.json
```

## Benchmarks

| Command | What it measures |
|---|---|
| `run llm` | Time to first token against prompt size, prefix-cache cold and warm, decode rate, thinking on and off, and concurrency. |
| `run decision` | Router decision latency against history size and label count, and how concurrent decisions queue. |
| `run router` | Every `AgentRouter` path with synthetic destinations, plus the acknowledgment race. |
| `run streaming` | `chat`, `stream`, and `stream_events` side by side, tool-call turns, very short outputs, and time to first audio under three delivery strategies. |
| `run vlm` | VLM latency against image count and resolution, streamed and not, cold and cached. |
| `run tts` | Batch and streamed text to speech against text length. |
| `run stt` | Speaker ASR end-of-speech to final result at real-time pace, and per-frame request cost. |
| `run vad` | Silero VAD firing time against input frame size. Runs on the CPU. |
| `run contention` | LLM latency while other load shares the server. |

Every sample is tagged with the number of requests other clients had running on the vLLM servers at that moment.
The report drops tagged samples, so run benchmarks while nothing else uses the stack.

## Patterns and estimates

`patterns/*.yaml` are common turn shapes: a plain voice question (batch or streamed), a routed question, a misroute,
look-and-answer, a tool loop, retrieval, and a spoken background alert. Each is a list of stages priced from the
fitted model. A stage carries a basis: `measured`, `fitted`, or `assumed` for a number written into the pattern
(tool time, retrieval, frame fetch). Stage p90 values are summed, which overstates the true p90 of a turn.

```yaml
stages:
  - {kind: endpoint}
  - {kind: route, path: multi_route, history_bytes: 6000}
  - {kind: speak, prompt_tokens: 4000, cached: true, out_tokens: 80,
     first_sentence_tokens: 20, strategy: sentence, playback_ms: 140}
```

`xr-latency estimate routed-qa --model model.json` prints the stages of one pattern (a path to your own YAML works
too), and `xr-latency patterns --model model.json [--thinking]` prices all of them. To price your own agent, copy the
closest pattern and change its token counts. The report's validation section predicts the composed results it also
measured and prints the error, so you can judge how far to trust an estimate.

## Profile defaults that override callers

`OpenAICompatLLM` writes the profile's `default_extras` into the request after
the per-call arguments, so a `max_tokens` or `temperature` in `default_extras`
silently replaces whatever the caller passed. Benchmarks that need to control
output length therefore use a profile without those defaults.

## Limits

- Transport latency (CloudXR, WebRTC, the network, device capture and display)
  is not measured. The `playback_ms` stage parameter is an assumption.
- Results describe the models, hardware, and quantization of the stack that was
  running, not the repository defaults.
- Static cost cards for each SDK component are in `utils/latency-profiler/docs/`.
  They are read from the source, not measured. The report marks which numbers
  have a benchmark behind them.
