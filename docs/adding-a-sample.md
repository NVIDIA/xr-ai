<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Adding a new sample

Read this when scaffolding a new agent sample. For a working reference, see
`agent-samples/simple-vlm-example/`. Hard rules and the checklist live in
`AGENTS.md`; this file holds the boilerplate templates.

## Naming conventions

Choose a kebab-case sample name (e.g. `simple-vlm-example`).
Derive all other names from it mechanically:

| Thing | Convention | Example |
|---|---|---|
| Sample directory | `agent-samples/<kebab-name>/` | `simple-vlm-example/` |
| Orchestrator module | `<snake_name>.py` | `simple_vlm_example.py` |
| Orchestrator entry point | `<snake_name>` | `simple_vlm_example` |
| Worker entry module | `<snake_name>_worker.py` | `simple_vlm_example_worker.py` |
| Worker entry point | `<snake_name>_worker` | `simple_vlm_example_worker` |
| Agent class | `<CamelName>Agent` | `SimpleVlmAgent` |
| Logger name | `"<snake_name>"` | `"simple_vlm_example"` |
| pyproject name (orch) | `"<kebab-name>"` | `"simple-vlm-example"` |
| pyproject name (worker) | `"<kebab-name>-worker"` | `"simple-vlm-example-worker"` |

## Directory layout

Both sub-projects use **flat module layouts** — no nested package
directories, no `__main__.py`, no `__init__.py`. Hatchling ships the
listed `.py` files as top-level modules in each sub-project's isolated
venv.

```
agent-samples/<name>/
├── pyproject.toml                  ← orchestrator project
├── main.py                         ← orchestrator (declare PROCESSES, call run_stack)
├── yaml/                           ← all YAML configs for this sample
│   ├── xr_media_hub.yaml
│   ├── <command>.yaml              ← one per launchable process
│   └── …
└── worker/
    ├── pyproject.toml              ← worker project
    ├── <snake_name>_worker.py      ← entry point: parses config, runs main loop
    ├── agent.py                    ← <CamelName>Agent class
    └── …                           ← split helpers as needed (audio.py, services.py, …)
```

When the worker is small (≲ 100 lines) keep it as a single file —
`worker/<snake_name>_worker.py` containing everything. Only split once
the file makes the agent logic harder to read; aim for a few focused
modules over one monolith *or* a swarm of tiny files.

Suggested split (used by `simple-vlm-example`):

| File | Responsibility |
|---|---|
| `<snake>_worker.py` | Entry point: config parsing, signal handling, lifecycle |
| `agent.py` | The agent class — IPC callbacks and orchestration |
| `audio.py` | WAV/PCM helpers, RMS, pixel-format conversion (rename if not audio) |
| `services.py` | Thin HTTP/MCP clients for external services + readiness probe |
| `voice.py` | Per-participant VAD/streaming-STT bookkeeping (when applicable) |

## Orchestrator `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "<kebab-name>"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = ["xr-ai-launcher"]

[tool.uv.sources]
xr-ai-launcher = { path = "../../utils/xr-ai-launcher", editable = true }

[project.scripts]
<snake_name> = "main:run"

[tool.hatch.build.targets.wheel]
only-include = ["main.py"]
```

## Worker `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "<kebab-name>-worker"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "xr-ai-agent",
    # add task-specific deps here: numpy, torch, etc.
]

[tool.uv.sources]
xr-ai-agent = { path = "../../../agent-sdk", editable = true }

[project.scripts]
<snake_name>_worker = "<snake_name>_worker:run"

[tool.hatch.build.targets.wheel]
only-include = [
    "<snake_name>_worker.py",
    "agent.py",
    # add other split modules here
]
```

When the worker is a single file, drop the extra entries:

```toml
[tool.hatch.build.targets.wheel]
only-include = ["<snake_name>_worker.py"]
```

## Orchestrator `main.py`

Exact boilerplate — do not add logic here:

```python
"""
<Name> agent orchestrator.  Runs the process stack for this sample.

How to run (from agent-samples/<name>/):
    uv sync && uv run <snake_name>
"""
from pathlib import Path

from xr_ai_launcher import Process, run_stack

_BASE = Path(__file__).resolve().parent

PROCESSES = [
    Process("hub",    "../../server-runtime", "xr_media_hub"),
    Process("worker", "worker",               "<snake_name>_worker"),
]


def run() -> None:
    run_stack(PROCESSES, _BASE)


if __name__ == "__main__":
    run()
```

## Worker `<snake_name>_worker.py`

Follow this structure exactly. Fill in the sections marked `# ← FILL IN`.

```python
"""
<Name> agent worker — <one-line description>.

Launched as a subprocess by ``uv run <snake_name>`` (the orchestrator).
Do not run this directly.

Protocol                            # ← include only if the worker sends/receives data msgs
--------
Client → agent  (topic "<in.topic>"):
    <description>

Agent → client  (topic "<out.topic>"):
    <description>

Environment                         # ← include only if env vars are read
-----------
    ENV_VAR   description (default: value)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from pathlib import Path

from xr_ai_agent import (          # ← import only what you use
    AudioChunk, DataMessage, FrameSignal, ParticipantEvent, ProcessorEndpoint,
    Subscribe,                     # ← only needed when scoping subscriptions
)

log = logging.getLogger("<snake_name>")

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


class <CamelName>Agent:            # ← FILL IN agent logic

    def __init__(self) -> None:
        self._ep = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
        self._ep.on_frame(self._on_frame)       # ← remove callbacks you don't use
        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_participant(self._on_participant)

    # ── callbacks ─────────────────────────────────────────────────────────────

    async def _on_frame(self, sig: FrameSignal) -> None: ...
    async def _on_audio(self, chunk: AudioChunk) -> None: ...
    async def _on_data(self, msg: DataMessage) -> None: ...
    async def _on_participant(self, event: ParticipantEvent) -> None: ...

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self._ep.run()        # ← start any background tasks before this line

    def shutdown(self) -> None:
        # ← cancel any background tasks here first
        self._ep.stop()
        self._ep.close()


async def main(ready_file: Path | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    agent = <CamelName>Agent()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    if ready_file:
        ready_file.touch()

    log.info("<snake_name> connecting  sub=%s  push=%s", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        agent.shutdown()


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ready-file", type=Path, default=None)
    ns, _ = p.parse_known_args()
    asyncio.run(main(ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
```
