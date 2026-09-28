<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Launcher and process model

Application samples own their DeviceIOHub and worker processes. Model services may
instead be declared with `launch_mode="reuse"` and started separately through the
shared `model-servers` stack.

An application sample has at least these two sub-projects:

| Sub-project | Role | Dependencies |
|---|---|---|
| `<sample>/` | Orchestrator — declares owned and reused processes in code | `xr-ai-launcher`, `xr-ai-logging`, and sample-specific orchestration dependencies |
| `<sample>/worker/` | Agent worker — connects to hub via IPC, runs agent logic | `xr-ai-hub-client`, numpy, etc. |

Samples may add capability or eval sub-projects. `model-servers` is a dedicated
orchestrator and has no worker sub-project.

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
Consumer samples declare only application processes; their launchers do not read
worker model profiles automatically.

The launcher profile loaders require the wrapped JSON form with `adapter` and
`endpoint` objects for each model, while `deployment` remains optional.
`load_model_deployment()` reads the profile selected by the worker YAML, and
`load_deployment_profile()` reads a profile path directly. Calling either loader
explicitly lets an orchestrator pass resolved deployment metadata and credential
requirements to preflight. The shared `model-servers` sample uses the direct
loader and managed entries to select servers. Refer to
{ref}`model profile formats <deployment-profiles>` for the worker and launcher
requirements.

## Declaring the process sequence

The orchestrator declares the process sequence in code:

```python
_BASE = Path(__file__).resolve().parent   # sample root

PROCESSES = [
    Process("hub",    "../../services/device-io-hub", "device_io_hub",
            config="yaml/device_io_hub.yaml"),
    Process("worker", "worker",               "my_agent_worker",
            config="yaml/my_agent_worker.yaml"),
    # Optional shared components (add as needed):
    # Process("cloudxr", "../../services/cloudxr-runtime", "cloudxr_runtime",
    #         config="yaml/cloudxr_runtime.yaml"),
]

def run() -> None:
    run_stack(PROCESSES, _BASE)
```

## Service artifact preparation

Download-capable service entry points accept `--prepare` with their normal
`--config <path>` argument. This mode resolves or downloads the service's
artifacts and exits without starting its server, opening listeners, or reporting
ready. Run it through the service project so preparation uses the same
dependency environment as a normal start:

```bash
uv run --project services/vlm-server vlm_server \
  --config services/vlm-server/vlm_server.yaml --prepare
```

The supported services prepare container images, Hugging Face snapshots, NIM
profiles, speech assets, DeviceIOHub browser artifacts, or the pinned LOVR
executable as applicable. Warm preparation checks recorded files, selected
versions, or the pinned LOVR checksum. Invalid Hugging Face manifests trigger
fresh downloads; other artifacts follow their service-specific cache checks.
Preparation failures include the service name. The `[prepare]` status lines are
human-readable progress output.

## Dependency contracts and preflight

A sample can declare host and endpoint prerequisites in `requirements.json`
beside its orchestrator. The
[`requirements.json` JSON Schema](https://github.com/NVIDIA/xr-ai/blob/main/utils/xr-ai-launcher/requirements.schema.json)
defines the machine-readable format. Every field is optional:

| Field | Value | Check |
|---|---|---|
| `$schema` | String | Optional JSON Schema identifier used by editors and validators |
| `os` | String or list of strings | Operating system name |
| `arch` | String, list, or `{allowed, unless_env_present?, unless_env_truthy?, unless_config?}` | Machine architecture, with optional custom-binary environment or configuration overrides |
| `python`, `node` | Version range or `{version, unless_env_truthy?}` | Installed runtime version |
| `cuda` | Boolean or version range | Available CUDA driver capability |
| `nvidia_driver`, `docker` | Minimum version or `{version, unless_env_truthy?}` | Installed driver or Docker version |
| `nvidia_container_toolkit` | Boolean | NVIDIA Docker runtime availability |
| `vulkan`, `nvenc` | Boolean | Usable Vulkan device or NVIDIA encoder |
| `disk_gb_free` | `{config_keys, runtime_gb, preparation}` | Runtime headroom and optional preparation-space checks for launcher-owned process caches |
| `ports` | List of `{port, name?, proto?, health?, config_key?, enabled_config_key?, unless_env_truthy?, bind_host?, bind_config_key?}` objects | Availability or expected endpoint health for TCP and UDP ports |
| `env` | List of `{name, required, docs?}` objects | Required or recommended environment credentials |
| `commands` | List of executable names or `{name, unless_env_truthy?}` objects | Commands available on `PATH` |

For each owned process, a disk policy selects the first configured key from that
process's YAML, expands environment variables and `~`, and resolves a relative
value from the YAML's directory. Missing configured cache keys are contract
errors. Runtime checks group cache paths by backing filesystem and apply the
largest declared headroom once per filesystem. The model-server contract
reserves 5 GB for manifests, temporary files, logs, and runtime cache growth
after its artifacts are present.

When `preparation` is true, callers supply
`PreparationSpace(process, cache_path, remaining_bytes)` entries for every
effective cache target after applying service-specific environment and
configuration precedence. A process may have multiple target paths. Preflight
groups them by backing filesystem and adds their remaining artifact bytes to
that filesystem's runtime headroom; zero remaining bytes means the caller
verified that the selected artifacts are fully prepared. Without inventory for
an owned process, preparation capacity is reported with `status: "deferred"`,
`ok: null`, and `skipped: true`. Deferred results do not fail the overall
report.

A port `name` that matches a `Process.name` lets preflight apply that process's
ownership rule. When the process does not declare `Process.port`, an explicit
`config_key` takes precedence; preflight then checks an `endpoint` URL in the
process YAML and finally falls back to the contract's declared `port`.
`enabled_config_key` omits a disabled listener, and `bind_config_key` reads its
bind address; `bind_host` supplies a fixed address when no configuration key is
needed. `unless_env_truthy` omits a version, command, or port only when the
named environment value is `true`, `yes`, `on`, or `1`. The architecture rule
also supports `unless_env_present`, which omits the requirement when the
variable has a nonempty value. The optional `health` value is an HTTP path such
as `/health`.

Profiles add `requirements.<profile>.json` beside the base contract. Objects
merge recursively in selection order; arrays and scalar values replace the base
value. Unknown fields and invalid values fail contract loading.

Call `preflight()` to resolve the contract against a set of processes and an
optional `ModelDeployment`. `base` resolves relative process and caller-supplied
preparation paths, `force_expensive` makes a missing local container image an
immediate probe failure, and `preparation_space` carries the selected-artifact
inventory described above. Invalid contracts, configuration values, and
process-to-inventory associations raise `ContractError`; invalid model profiles
raise `ValueError`, and unreadable process configuration discovered from the
process list can raise `OSError`. Otherwise, preflight returns a
`PreflightResult` without printing. The result classifies services as owned,
reused, or external:

- An owned service needs its local commands, devices, disk, and configured
  ports. An occupied port is accepted only when the process's ownership probe
  verifies configuration compatibility and readiness. A successful health
  response alone is insufficient.
- A reused service must answer at the endpoint selected by the model deployment
  profile. Local GPU, Docker, disk, and bind-port requirements do not apply to
  that service.
- An external service is checked at its configured endpoint when the profile
  enables health readiness. Credential values remain redacted. External services
  do not trigger model-side local requirements, whether their endpoint is
  loopback or remote.

Owned-port preflight uses the Linux iproute2 `ss` command to inspect TCP
listeners and all UDP sockets. When `ss` is missing, fails, or prints
unparseable output, an ownership probe that verifies the expected managed
service turns the failure into a warning. A timeout or an invalid or
unresolvable bind host always fails.

A passing inspection does not guarantee that the service can bind. The TCP
check can miss a socket that is bound but not listening or a connected socket
that holds the port, and sockets in another network namespace may not be
visible. The service's own bind at launch is authoritative. For observed IPv6
sockets, `ss` renders a dual-stack wildcard as `*` and an IPv6-only wildcard as
`[::]`; an IPv6-only socket does not block an IPv4 bind. An IPv6 target is
checked against observed IPv6 sockets only.

Preflight also compares every resolved owned TCP and UDP port with the host's
`ip_local_port_range` and `ip_local_reserved_ports`. An unreserved service port
inside the inclusive ephemeral range produces a warning because the kernel can
automatically select it for a connection before the service binds. A reservation
prevents future automatic selection; it does not release an existing connection
that already holds the port. Refer to the
[Linux IP sysctl documentation](https://docs.kernel.org/networking/ip-sysctl.html#ip-variables)
for the kernel contract.

If either kernel setting is unreadable or malformed, preflight emits one
nonblocking warning because it cannot determine the policy.

Cheap checks cover versions, commands, credentials, disk, ports, and endpoint
health. Expensive GPU-container, NVENC, and Vulkan probes run directly after
applicable cheap checks pass; failed cheap checks leave them visibly blocked. If
the container GPU probe cannot run before its local image is prepared, it is
deferred. After preparation, call `rerun_deferred_checks(result)` to replace
that result without repeating the completed checks. The caller decides how to
report the returned `PreflightResult` and whether to continue to preparation or
launch.

Set `Process.ownership_probe` to a callable that receives the environment
preflight would use for the child process.
`managed_service_matches(..., project=...)` provides the standard model-service
probe. For Docker services, `project` is required so the probe can run
`uv run --quiet --offline --no-sync --project <project> <entrypoint> --config <config> --describe-launch`
and calculate the expected effective launch identity without starting or
stopping the service. A missing project, failed or malformed description, Docker
inspection error, timeout, or ambiguous set of labelled containers leaves
ownership unverified. A definite service or effective-identity mismatch raises
`OwnershipProbeMismatch` with the configured remediation.

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

## Serial and parallel items

The stack is declared as a sequence of `Process` or `Parallel` items:

- `Process` — when not configured with `launch_mode="reuse"`, started alone;
  the launcher waits for it to signal ready before moving on.
- `Parallel([p1, p2, ...])` — all non-`reuse` processes in the group are started
  at once; the launcher waits for every spawned member to signal ready before
  the next item in the sequence begins. If any spawned member exits before
  signaling ready, the launcher shuts everything down, just as it would for a
  serial process.

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
exits before creating it, startup is aborted and the whole stack is torn down.
This makes readiness explicit and process-defined: a model server signals ready
after weights load, an HTTP server after it starts listening, a worker after
its IPC receive loop is active.

### `launch_mode`: own, persist, reuse

`Process.launch_mode` controls spawn and shutdown behaviour:

- `"own"` (default): the launcher spawns this process and kills it on shutdown.
- `"persist"`: the launcher spawns this process but leaves it running on
  shutdown. Use this mode for heavy model servers that need to survive stack
  restarts (for example, vLLM containers). Cleanup is the caller's
  responsibility. The optional `port` field supplies service metadata to
  preflight and model-server cleanup.
- `"reuse"`: the launcher assumes this process is already running (for example,
  when started by `model-servers`) and skips it. The entry documents the
  dependency and can be passed to `preflight()` to require readiness before
  launching the stack.

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
sub-project without dragging in a heavy dependency chain. Their public names,
signatures, types, defaults, fields, and method behavior are generated in the
{doc}`Python API reference </reference/python/index>`.

**`xr-ai-launcher`**: process management for the xr-ai stack: the `Process`,
`Parallel`, and `run_stack` API described above, dependency-contract and
preflight APIs, plus helpers for CloudXR environment setup, credential loading,
GPU detection, artifact manifests, preparation status, and preparation error
handling. Intentionally stdlib-only so it can be added to any sample without
pulling in the dependency chain of the processes it manages.

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

**`xr-ai-vllm`**: pluggable vLLM backend for inference services. Each
vLLM-backed service can host vllm via `pip` (the pip-installed `vllm` CLI in the
wrapper's venv, the default) or `docker` (the image selected by `vllm_image`),
chosen per-server via `vllm_backend: pip|docker` in the service YAML. Both paths
honor identical configuration keys; only the runtime hosting vllm differs.
Services can also use the package to prepare vLLM or NIM images and model
artifacts without starting a server and verify persistent launch identity before
reuse. Its only runtime dependency is the stdlib-only `xr-ai-launcher` package,
so the docker path stays light even when pip vllm is not installed.
