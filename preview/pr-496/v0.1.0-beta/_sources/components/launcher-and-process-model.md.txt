<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Launcher and process model

Every sample is self-contained: running it starts the XR-Media-Hub and all required
processes automatically. No separate server launch step.

Each sample has **two sub-projects**:

| Sub-project | Role | Dependencies |
|---|---|---|
| `<sample>/` | Orchestrator — declares process list in code, launches all | `xr-ai-launcher` only (stdlib) |
| `<sample>/worker/` | Agent worker — connects to hub via IPC, runs agent logic | `xr-ai-agent`, numpy, etc. |

**Configuration convention** — the YAML configuration path for each process is declared
explicitly in the orchestrator's `PROCESSES` list via the `config=` field of
`Process`. The launcher passes it as `--config <path>` to the subprocess.
All sample configuration files live in the `yaml/` directory. Omit `config=` for
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

## Serial and parallel items

The stack is declared as a sequence of `Process` or `Parallel` items:

- `Process` — started alone; the launcher waits for it to signal ready before
  moving on.
- `Parallel([p1, p2, ...])` — all processes in the group are started at once;
  the launcher waits for *every* member to signal ready before the next item
  in the sequence begins. If any member exits before signaling ready, the
  launcher shuts everything down, just as it would for a serial process.

```python
PROCESSES = [
    Process("hub",    "../../server-runtime", "xr_media_hub"),
    Parallel([
        Process("stt", "../../ai-services/stt-server", "stt_server"),
        Process("tts", "../../ai-services/tts/piper",  "piper_tts_server"),
    ]),
    Process("worker", "worker", "my_agent_worker"),
]
```

## How `run_stack` works

For each process the launcher:

1. Resolves the project directory and YAML configuration from the sample root (`base`
   — all relative paths in `Process.project` and `Process.config` are resolved
   against it).
2. Spawns `uv run --project <dir> <command> --config <yaml> --ready-file <f>`
   in a new process group, so the whole group (`uv` plus its children) can be
   torn down together rather than leaving orphans.
3. Waits for the process to create *<f>* (the ready file), printing a progress
   line every five seconds so slow starts remain visible.
4. Once all processes are ready, monitors them: any exit triggers a graceful
   shutdown of the rest (SIGTERM, escalating to SIGKILL after a timeout).

Each process is responsible for creating its own ready file at the moment it
is fully initialized and able to serve requests — after model warm-up, after
the IPC socket connects, after the HTTP server starts listening, etc.

Pass `exit_after_ready=True` to `run_stack` to return immediately once
everything is ready instead of monitoring — useful for launchers whose
processes are all `launch_mode="persist"` and should outlive the orchestrator
(e.g. `model-servers`).

### The `--ready-file` protocol

The launcher injects `--ready-file <path>` into every spawned command. The
process must `Path(path).touch()` the moment it is fully initialized and able
to serve requests. The launcher blocks on the file's existence; if the process
exits before creating it, startup is aborted and the whole stack is torn down.
This makes readiness explicit and process-defined: a model server signals ready
after weights load, an HTTP server after it starts listening, a worker after
its IPC socket connects.

### `launch_mode`: own, persist, reuse

`Process.launch_mode` controls spawn and shutdown behaviour:

- `"own"` (default) — the launcher spawns this process and kills it on
  shutdown.
- `"persist"` — the launcher spawns this process but leaves it running on
  shutdown. Use for heavy model servers that should survive stack restarts
  (e.g. vLLM containers). Cleanup is the caller's responsibility. The optional
  `port` field is used to stop such persistent services.
- `"reuse"` — the launcher does **not** spawn this process; it is assumed to be
  already running (e.g. started by `model-servers`). The entry in the process
  list documents the dependency; the launcher skips it entirely and does not
  kill it on shutdown.

On a clean ready-exit, `persist` and `reuse` processes are left running. On an
abort during startup (Ctrl-C, or a process exiting before it signals ready)
the launcher tears down **everything**, including `persist` processes, so no
half-started service is left behind.

## Adding a new managed process

There is no per-process launcher module to write — the launcher spawns any uv
sub-project generically. To add a new process to a stack:

1. Make the sub-project's entry-point command accept `--ready-file <path>`
   (touch it once ready) and, if it takes configuration, `--config <path>`.
2. Add a `Process` (or `Parallel`) entry to the orchestrator's `PROCESSES`
   list, in dependency order, pointing at the sub-project directory and its
   entry-point command — exactly like the `hub` and `worker` entries above.

## Shared `utils/` packages

The orchestrator's process management and several cross-cutting concerns live
in single-purpose packages under `utils/`. Each is small and narrowly scoped,
and most are deliberately dependency-light so they can be added to any
sub-project without dragging in a heavy dependency chain.

**`xr-ai-launcher`** — process management for the xr-ai stack: the `Process`,
`Parallel`, and `run_stack` API described above, plus helpers for CloudXR
environment setup, credential loading, and GPU detection. Intentionally
stdlib-only so it can be added to any sample without pulling in the dependency
chain of the processes it manages.

**`xr-ai-logging`** — shared loguru setup for the monorepo. Every process calls
`setup_logging()` once at startup to get a unified logging stack: a stderr sink
(level controlled by `XR_AI_VERBOSE`), a DEBUG file sink under
`/tmp/log_<namespace>_<timestamp>/`, and a stdlib bridge that routes records
emitted via `logging.getLogger(...)` into loguru — so stdlib-only packages
(`xr-ai-launcher`) and the agent SDK end up in the same sinks. The orchestrator
stamps namespace, timestamp, and root env vars so all spawned subprocesses write into
the same per-run folder.

**`xr-ai-vad`** — shared Silero-VAD utterance detector for agent workers. It
consumes int16 LE PCM audio and emits int16 PCM utterance bytes via an async
callback when speech ends, so workers get a single, consistent voice-activity
boundary without each re-implementing VAD.

**`xr-ai-voicegate`** — the speech-only opt-in gate shared by agent workers.
It owns the magic-phrase, follow-up, and STOP ladder, the lazy listening chime,
and the participant-joined greeting hook. Workers feed STT transcripts via
`feed` and register handlers for the events it emits (query, stop, phrase-only,
drop, participant-joined).

**`xr-ai-vllm`** — pluggable vLLM backend for inference services. Each
vLLM-backed service can host vllm via `pip` (the pip-installed `vllm` CLI in
the wrapper's venv, the default) or `docker` (`docker run nvcr.io/nvidia/vllm`,
an NGC container), chosen per-server via `vllm_backend: pip|docker` in the
service YAML. Both paths honor identical configuration keys; only the runtime hosting
vllm differs. Stdlib-only by contract, so the docker path stays light even when
pip vllm is not installed.

**`xr-ai-nemo-runtime`** — opt-in NGC NeMo container backend for the in-process
NeMo servers (the STT server and the Magpie TTS server). The NeMo servers load
torch+NeMo in the wrapper's venv and inherit the host's cuDNN/CUDA via
`LD_LIBRARY_PATH`, which aborts at torch import on hosts whose system cuDNN
differs from torch's bundled one. This runs the same FastAPI server inside an
NGC NeMo image instead — so torch, NeMo, and cuDNN all come from the container
and the host's libraries are irrelevant — opted into per-server via
`backend: docker` in the service YAML (default `pip` keeps the in-venv
behavior). It is a bespoke NeMo docker path, deliberately separate from
`xr-ai-vllm`, and is stdlib-only by contract.
