<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Visual task guide

This focused sample guides a ten-step hand-counting task with an explicit NAT
state machine and on-demand current-frame vision. The web client's **Agent
output** shows the current step and each requested validation, for example:

```text
Show three — Yes, I see 3 extended fingers.
```

Vision never advances the task. Only explicit `start task`, `next step`, and
`reset task` commands change state. `task status` reads it. These controls,
on-demand vision, RAG, and the root workflow are native NAT functions.

## Bundled task

`tasks/hand-counting/workflow.yaml` orders ten separate step YAML files. Each
step declares its instruction, visible criterion, expected finger count, and
expected visible-hand count. Deterministic validation reads those fields rather
than inferring the answer from the step number. The RAG service is intentionally
configured for this sample's bundled `knowledge/` directory; this PR does not
claim a general copy-and-retarget task-folder contract.

## Run

```bash
cd agent-samples/model-servers
uv sync
uv run model_servers
```

In another terminal:

```bash
cd agent-samples/visual-task-guide
uv sync
cd worker && uv sync && cd ..
uv run visual_task_guide
```

The launcher requires the shared VLM, Nemotron-3-Nano guide LLM, STT, and
embedding endpoints before it starts the hub, Piper TTS, RAG service, and
worker. This sample does not launch video memory or record historical video.

Open `https://localhost:8080`, connect, and start the camera. Start the
microphone for voice interaction or use the text box:

- `start task` starts at **Show one**.
- `next step` advances exactly once.
- “What’s the next step?” reports the following step without advancing.
- `task status` prints the current step.
- `reset task` returns to **Show one / not started**.
- “Did I do the step correctly?” captures one fresh frame and compares its
  reliable count with the current step without invoking RAG.
- “How many fingers do you see?” captures one fresh frame.
- “How should I position both hands?” uses dense retrieval over the bundled task documents.

Voice and typed commands are both dispatched directly; this focused demo does
not require a wake phrase, but accepts an optional “agent” or “hey agent”
prefix. A vision request runs only when the user asks a visual question. The
workflow uses a neutral count query with no target answer, then captures one
latest frame. Validation parses the VLM's compact count contract and compares
it deterministically with the trusted step. Direct count questions bypass the
guide LLM; other questions combine the fresh visual result with bounded dense
retrieval in one 128-token pass.

The worker console logs task transitions, RAG citations, and total workflow
latency. Model prompts and full payloads are not logged.

The reusable boundaries are `StreamingVisionConfig`, `RAGFunctionsConfig`, and
`ModelsLLMConfig`. The sample owns the session-local state machine, task
workflow, and focused guide agent.
Progress resets whenever the worker starts or the participant reconnects.

## Evaluate deployed prompts

With the shared model servers and this sample's RAG service running:

```bash
uv run --project eval visual_task_guide_eval
```

The harness calls both deployed models, checks concise output, verifies native
dense RAG retrieval, and audits fixture leakage. See
[`eval/README.md`](eval/README.md).
