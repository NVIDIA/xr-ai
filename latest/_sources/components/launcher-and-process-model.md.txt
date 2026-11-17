<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Launcher and process model

## Overview

An XR AI application runs a hub, agent workers, and any additional services it
needs. The launcher starts those processes in dependency order, waits for
readiness, and stops the application stack if a monitored process exits. Use
it to make your application repeatable to launch and operate.

## Declare the application stack

Application samples own their DeviceIOHub and worker processes. Model services may
instead be declared with `launch_mode="reuse"` and started separately through the
sample's separate `model-servers/` convenience launcher.

An application sample has at least these two sub-projects:

| Sub-project | Role | Dependencies |
|---|---|---|
| `<sample>/` | Orchestrator — declares owned and reused processes in code | `xr-ai-launcher`, `xr-ai-logging`, and sample-specific orchestration dependencies |
| `<sample>/worker/` | Agent worker — connects to hub via IPC, runs agent logic | `xr-ai-hub-client`, numpy, etc. |

Samples may add capability or eval sub-projects. Each sample's `model-servers/`
project is a separate inference convenience launcher with no worker. It combines
each service's own YAML under `services/` with sample-owned GPU
profiles. Refer to {doc}`/guides/customizing-model-servers` for configuration and
startup, shutdown, and sample-switching workflows.

**Configuration convention** — the YAML configuration path for each process is declared
explicitly in the orchestrator's `PROCESSES` list via the `config=` field of
`Process`. The launcher passes it as `--config <path>` to the subprocess.
Application and service configuration normally lives in the sample's `yaml/`
directory. A capability sub-project may keep its configuration beside that
project, such as `xr-render-demo/scene/scene_service.yaml`. Omit `config=` for
processes that use their own internal defaults.

Samples that support interchangeable local and hosted models may set
`models_config` in the worker YAML. The worker SDK's `load_models_config()`
accepts omitted `deployment` metadata and defaults it to external ownership.
Consumer samples declare only application processes and let workers connect to
the configured endpoints; their launchers do not read model profiles.

Launcher profile loaders require the wrapped JSON form with `adapter`,
`endpoint`, and `deployment` objects for each model. `load_model_deployment()`
reads the profile selected by the worker YAML, while `load_deployment_profile()`
reads a profile path directly. Both use only the standard library and expose
deployment metadata and credential requirements. The sample-local model
launchers declare their service lists directly and do not use these loaders.
Explicit reused and external deployment entries remain accepted by the loaders. Refer to
{ref}`model profile formats <deployment-profiles>` for the worker and launcher
requirements.

The orchestrator declares the process sequence in code:

```python
_BASE = Path(__file__).resolve().parent   # sample root

PROCESSES = [
    Process("hub",    "../../services/device-io-hub", "device_io_hub",
            config="yaml/device_io_hub.yaml"),
    Process("worker", "worker",               "my_agent_worker",
            config="yaml/my_agent_worker.yaml"),
    # Optional shared components — add as needed:
    # Process("cloudxr", "../../services/cloudxr-runtime", "cloudxr_runtime",
    #         config="yaml/cloudxr_runtime.yaml"),
]

def run() -> None:
    run_stack(PROCESSES, _BASE)
```

## Rules

- **Spawned stack items start in declaration order** — non-`reuse` members of a
  `Parallel` item start concurrently, and the launcher waits for each spawned
  `Process` or `Parallel` member to create its `--ready-file` before starting
  the next item. Declare items in dependency order (hub before workers and
  application processes after the services they call).
- **Every spawned process accepts `--ready-file <path>`** and must `Path(path).touch()`
  when it is fully initialized and ready to serve requests.
- **Native voice workers** pass the ready file to `VoiceAgent`; its private
  media session touches it only after the input transport's hub IPC
  receive loop has started.
- `device_io_hub` always runs as its own process — never embedded in-process.
- The worker never imports anything from `device_io_hub` or `xr_ai_launcher`.
- Process management lives in `utils/xr-ai-launcher/`, not inside any process it manages.
- `run_stack` is fail-fast while monitoring: if any spawned process exits, the
  rest are terminated.

(consumer-model-readiness)=
## Consumer model readiness

Worker readiness reports local initialization, not general model availability.
Batch STT, TTS, LLM, and VLM health are not polled automatically. Start the
sample's model-server stack and wait for its launcher to return before starting
the sample, or provision and validate external endpoints first. Server wrappers
perform their own startup and reuse checks.

`VoiceAgent` waits for explicitly supplied `probes`. It also selects the
configured speaker ASR backend; `speaker.backend: required` waits for that
service before worker readiness. Endpoint `readiness` and `health_path` settings
control explicit `health()` calls and do not enable automatic worker polling.
Applications that need batch STT and TTS health checks can pass
`probes={"stt": stt.health, "tts": tts.health}` to `VoiceAgent`.

{doc}`Simple VLM </reference/simple-vlm-example>` performs a streaming image
warmup, and {doc}`XR Render </reference/xr-render-demo>` performs an LLM warmup,
including for hosted LLMs. Tea making checks its RAG capability and
{ref}`embedding health <rag-embedding-health>`. These application-specific
checks can delay readiness independently of the voice runtime.

## Serial and parallel items

The stack is declared as a sequence of `Process` or `Parallel` items:

- `Process` — when not configured with `launch_mode="reuse"`, started alone;
  the launcher waits for it to signal ready before moving on.
- `Parallel([p1, p2, ...])` — all non-`reuse` processes in the group are started
  at once; the launcher waits for every spawned member to signal ready before
  the next item in the sequence begins. If any spawned member exits before
  signaling ready, the launcher stops owned processes and unfinished persistent
  wrappers, just as it would for a serial process. Persistent services that
  had signaled readiness when cleanup began stay running.

```python
PROCESSES = [
    Process("vlm", "../../services/vlm-server", "vlm_server",
            launch_mode="reuse"),
    Process(
        "embedding", "../../services/embedding-server", "embedding_server",
        launch_mode="reuse",
    ),
    Process("hub", "../../services/device-io-hub", "device_io_hub",
            config="yaml/device_io_hub.yaml"),
    Parallel([
        Process("video-memory", "../../services/video-memory-service",
                "video_memory_service", config="yaml/video_memory_service.yaml"),
        Process("rag", "../../services/rag-service", "rag_service",
                config="yaml/rag_service.yaml"),
    ]),
    Process("worker", "worker", "my_agent_worker"),
]
```

## How `run_stack` works

For each spawned process (an entry not configured with `launch_mode="reuse"`),
the launcher:

1. Resolves the project directory and YAML configuration from the sample root (`base`
   — all relative paths in `Process.project` and `Process.config` are resolved
   against it).
2. Spawns `uv run --project <dir> <command> --config <yaml> --ready-file <f>`
   in a new process group, so the whole group (`uv` plus its children) can be
   torn down together rather than leaving orphans.
3. Waits for the process to create *<f>* (the ready file), recording a DEBUG
   progress line every five seconds. It is visible with `XR_AI_VERBOSE` and in
   the per-run log.
4. Once all processes are ready, monitors them: any exit triggers a graceful
   shutdown of the rest (SIGTERM, escalating to SIGKILL after a timeout).

Each spawned process is responsible for creating its own ready file at the
moment it is fully initialized and able to serve requests — after model warm-up,
after the IPC socket connects, after the HTTP server starts listening, etc.

Pass `exit_after_ready=True` to `run_stack` to return immediately once
everything is ready instead of monitoring — useful for launchers whose
processes are all `launch_mode="persist"` and are designed to outlive the
orchestrator (e.g. `model-servers`). For a persistent process, this also tells
a bootstrap that reused an already-running service that it may exit after
signaling readiness. Without `exit_after_ready=True`, the bootstrap remains
alive so the launcher can continue monitoring it.

### The `--ready-file` protocol

The launcher injects `--ready-file <path>` into every spawned command. The
process must `Path(path).touch()` the moment it is fully initialized and able
to serve requests. The launcher blocks on the file's existence; if the process
exits before creating it, startup is aborted. Owned processes and unfinished
persistent wrappers are stopped; persistent services that had signaled readiness
when cleanup began stay running. A service in a separate session needs its
wrapper to forward shutdown signals; persistent pip-backend vLLM does not
forward them before readiness and can keep loading.
This makes readiness explicit and process-defined: a model server signals ready
after weights load, an HTTP server after it starts listening, a worker after
its IPC receive loop is active.

### Foreground stack health

Applications using `run_stack` can pass `health_port=8093` to expose process
health to a local supervisor or monitoring script. Repository samples expose
this as `--health-port PORT`. The listener binds only to loopback, separately
from the worker's optional event viewer. For example, from
`agent-samples/simple-vlm-example/`:

```bash
uv run simple_vlm_example --health-port 8093
```

Query `http://127.0.0.1:8093/healthz` locally. It returns 503 while startup or
cleanup is in progress, and 200 after startup completes while every spawned
process is still running. Every request checks the child-process handles.
The JSON response contains `status` (`ready` or `not_ready`), `processes`
(spawned names mapped to `running` or `exited`), and `unchecked_reuse` (names
of external services that are not checked). HEAD returns the same status
without a body. Shutdown signals immediately clear readiness.
An unreachable endpoint means the orchestrator cannot report health; it does
not establish whether its child processes or external services have stopped.
Keep the endpoint private; use SSH forwarding for remote inspection.

This is process health, not an inference or dependency probe. Services declared
with `launch_mode="reuse"` are not checked, and live child processes may still
be unresponsive. The listener is disabled by default and for
`exit_after_ready=True`. Invalid or occupied ports fail before launching
processes. Refer to `run_stack` for the API contract.

### `launch_mode`: own, persist, reuse

`Process.launch_mode` controls spawn and shutdown behaviour:

- `"own"` (default) — the launcher spawns this process and kills it on
  shutdown.
- `"persist"` — the launcher spawns this process but leaves it running on
  shutdown. Use for heavy model servers that need to survive stack restarts
  (e.g. vLLM containers). Cleanup is the caller's responsibility. The optional
  `port` field is metadata; `model-servers` uses it to select cleanup targets,
  while generic `run_stack` does not inspect it.
- `"reuse"` — the launcher does **not** spawn this process; it is assumed to be
  already running (e.g. started by `model-servers`). The entry in the process
  list documents the dependency; the launcher skips it entirely and does not
  kill it on shutdown.

On a clean ready-exit, `persist` and `reuse` processes are left running. On an
abort during startup (SIGINT, SIGTERM, a process exiting before it signals ready,
or an unexpected exit while other members of its parallel group start), the launcher
stops `own` processes and `persist` wrappers that have not signaled readiness.
Persistent services that had signaled readiness when cleanup began stay running,
including ready members of a parallel group whose other members failed. For
model-server stacks, use the sample's model command with `--stop` to clear
retained models. Externally
started `reuse` services remain untouched. Signal handling stays active through
cleanup; repeated SIGINT or SIGTERM does not interrupt cleanup or force early
SIGKILL. A wrapper that ignores SIGTERM can take the full 20-second shutdown
timeout plus up to 5 seconds waiting after SIGKILL. SIGKILL and launcher crashes
cannot run this cleanup.

## Process failures

When a managed process cannot start, exits before readiness, or exits while
the stack is being monitored, the launcher prints the failed process,
command, project, configuration, and exit status or spawn error. When the
configured run-log directory exists, the summary includes its path. Inspect
the service's terminal output and run logs for the underlying error and any
recovery advice. Each service controls its own log filename. Before reporting
a terminated child's failure, the launcher waits up to one second for its
pending terminal output to be forwarded. A descendant that keeps an output
pipe open cannot hold up this wait indefinitely.

An unexpected monitored exit is a stack failure even when the child exits
with code 0; the launcher returns 1 in that case. Other child exit codes are
preserved, and signal termination maps to `128 + signal number`. Startup
cancellation returns `128 + signal number` (130 for SIGINT, 143 for SIGTERM);
cancellation after readiness follows normal shutdown. Summary errors do not
replace the original failure. Members of a
parallel startup group remain monitored until the entire group is ready. An
unexpected exit during that wait triggers startup cleanup even if the exiting
process had already signaled ready. Other persistent services that had signaled
readiness when cleanup began stay running. With `exit_after_ready=True`, a
`persist` wrapper may exit successfully after signaling ready while its service
keeps running.

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
sub-project without dragging in a heavy dependency chain. Their public names,
signatures, types, defaults, fields, and method behavior are generated in the
{doc}`Python API reference </reference/python/index>`.

**`xr-ai-launcher`** — process management for the xr-ai stack: the `Process`,
`Parallel`, and `run_stack` API described above, plus helpers for CloudXR
environment setup, credential loading, and GPU detection. Intentionally
based on the standard library and the dependency-free `xr-ai-gpu` utility so
it can be added to any sample without pulling in the dependency
chain of the processes it manages.

**`xr-ai-logging`** — shared loguru setup for the monorepo. Every process calls
`setup_logging()` once at startup to get a unified logging stack: a stderr sink
(level controlled by `XR_AI_VERBOSE`), a DEBUG file sink under
`/tmp/log_<namespace>_<timestamp>/`, and a stdlib bridge that routes records
emitted via `logging.getLogger(...)` into loguru. The launcher depends only on
`xr-ai-gpu` and uses standard-library logging; its records and the agent SDK's
records end up in the same sinks without either package importing loguru. The orchestrator
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
the wrapper's venv, the default) or `docker` (the image selected by
`vllm_image`), chosen per-server via `vllm_backend: pip|docker` in the service
YAML. Both paths honor identical configuration keys; only the runtime hosting
vllm differs. Stdlib-only by contract, so the docker path stays light even when
pip vllm is not installed.

GPU hardware facts live in `xr-ai-gpu`, independently of application
orchestration. Launcher profile matching uses its strict full inventory with no
driver deadline. CloudXR uses its narrow PCI lookup with a five-second deadline
and continues without pinning on failure. vLLM retains its existing selected-GPU
queries and fallback behavior. Profile recommendations and environment
construction remain with their consuming packages.
