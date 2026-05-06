<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Process model

Every sample is self-contained: running it starts the hub and all required
processes automatically. No separate server launch step.

Each sample has **two sub-projects**:

| Sub-project | Role | Dependencies |
|---|---|---|
| `<sample>/` | Orchestrator — declares process list in code, launches all | `xr-ai-launcher` only (stdlib) |
| `<sample>/worker/` | Agent worker — connects to hub via IPC, runs agent logic | `xr-ai-agent`, numpy, etc. |

**Config convention** — the YAML config path for each process is declared
explicitly in the orchestrator's `PROCESSES` list via the `config=` field of
`Process`. The launcher passes it as `--config <path>` to the subprocess.
All sample configs live in the `yaml/` directory. Omit `config=` for
processes that use their own internal defaults.

The orchestrator declares the process sequence in code:

```python
_BASE = Path(__file__).resolve().parent   # sample root

PROCESSES = [
    Process("hub",    "../../server-runtime", "xr_media_hub",
            config="yaml/xr_media_hub.yaml"),
    Process("worker", "worker",               "my_agent_worker",
            config="yaml/my_agent_worker.yaml"),
    # Optional shared components — add as needed:
    # Process("cloudxr", "../../cloudxr-runtime",       "cloudxr_runtime",
    #         config="yaml/cloudxr_runtime.yaml"),
    # Process("mcp",     "../../agent-mcp-servers/oxr-mcp", "oxr_mcp_server",
    #         config="yaml/oxr_mcp_server.yaml"),
]

def run() -> None:
    run_stack(PROCESSES, _BASE)
```

## Rules

- **Processes start serially** — each process must create its `--ready-file`
  before the next one starts. Declare processes in dependency order (hub
  before workers, cloudxr before MCP servers that open OpenXR sessions, etc.).
- **Every process accepts `--ready-file <path>`** and must `Path(path).touch()`
  when it is fully initialized and ready to serve requests.
- `xr_media_hub` always runs as its own process — never embedded in-process.
- The worker never imports anything from `server-runtime` or `utils/xr-ai-launcher/`.
- Process management lives in `utils/xr-ai-launcher/`, not inside any process it manages.
- `run_stack` is fail-fast: if any process exits, the rest are terminated.

## Adding a new managed process type

Add `utils/xr-ai-launcher/xr_ai_launcher/_<name>.py` following the pattern in `_hub.py`.
Use `ManagedProcess` as the base. Export from `__init__.py`.
