<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Decisions & change log

Significant decisions, in reverse-chronological order. Update this whenever a
non-trivial architectural or design decision is made so the rationale is
preserved and not re-litigated.

### 2026-05-20 — Hub releases held ring-buffer slots on participant leave (#143)

`HubEndpoint` holds the latest SHM ring slot per `(participant_id,
track_id)` so processors can fetch pixels on demand without an eager
copy. The slot was only released when the *next* FRAME_SIGNAL for the
same key arrived — so when a track ended (LiveKit `track_unsubscribed`
or `participant_disconnected`), its last slot stayed held forever.
After enough connect/publish/disconnect cycles the ring filled with
abandoned slots and every subsequent frame from any participant was
dropped at the connector with `Ring buffer full — dropped frame`.
A new participant could publish video and the worker would log
`tracks_seen=0` until the hub was restarted.

**Fix.** The hub now releases every slot keyed by a participant when
that participant's `PARTICIPANT_EVENT(joined=False)` arrives. The
`notify_participant_left` path already fires on both `track_unsubscribed`
+ `participant_disconnected` (it is called from `_room_client._handle_left`),
so no new message type is required. Reuses the established
"release_slot without `view.data.release()`" pattern from the FRAME_SIGNAL
branch — the connector's ring is still live at this point, so the
memoryview does not need an explicit release.

Also bumped `_DEFAULT_NUM_SLOTS` from 10 → 16 so a single ill-timed
reconnect within the in-flight window between last frame and disconnect
event can't still hit the ceiling. The slot is 12.4 MiB at the 4K NV12
ceiling, so six extra slots adds ~75 MiB to the worst-case per-connector
shm footprint — cheap insurance.

**Out of scope.** Connector crash / OOM still leaks slots — `joined=False`
is only emitted by the live `notify_participant_left` path, not by an
unclean exit. A heartbeat or TTL scan would close that gap; deferred
until there's evidence it matters in practice. The issue's reproduction
is wholly covered by participant disconnect.

### 2026-05-18 — `nightly XR AI test` workflow on self-hosted `gpu` runner

The `gpu`-marked pytest suite (`tests/test_gpu_*.py`,
`tests/test_integration_livekit.py`, `tests/test_local_render_mcp.py`)
is filtered out of the default `tests` workflow because it needs real
GPU / Docker / NVENC hardware that the `ubuntu-latest` runners don't
provide — see the 2026-05-12 entry for why the marker was introduced.
Until now the suite ran only via `tests/run_local_gpu_tests.sh` on
developer boxes, which meant regressions could slip into `main` between
ad-hoc local runs.

A new `.github/workflows/nightly-xr-ai-test.yml` runs the same suite at
04:00 UTC every day (and on-demand via `workflow_dispatch`) on a
self-hosted runner registered with the `gpu` label. The job mirrors
`tests.yml` — `uv sync` + `pytest -m gpu` from `tests/` — at Python
3.12 only, since these tests are GPU-bound rather than
Python-version-bound and doubling the matrix would just double GPU-hour
cost without new coverage. Concurrency is set to queue (not cancel)
overlapping runs so a long nightly finishes before the next cron fires.

**Runner hygiene.** The `gpu` label points at a persistent host, so
state leaks across runs. Two layers of cleanup keep the suite robust:

1. A workflow-level pre/post step force-removes every container named
   `xr-ai-vllm-*` (the prefix the launcher uses) and asserts ≥ 30 GiB
   of GPU memory is free at start — a clear hard error here beats a
   confusing downstream vLLM OOM.
2. An autouse pytest fixture in `tests/conftest.py` does the same scrub
   around each `@pytest.mark.gpu` test. Per-test `finally` blocks
   already call `stop_persistent_servers`, but those don't run if
   pytest itself is killed or a fixture errors out — and any leak
   between LLM tests on a single 46 GiB GPU OOMs the next one.

**CUDA toolkit discovery.** A discovery step picks the toolkit from a
priority list (`/usr/local/cuda-13.0`, `…-13`, `…`, `$CUDA_HOME`, then
`which nvcc`) and exports `CUDA_HOME` plus `CUDACXX` via `$GITHUB_ENV`
so downstream JIT compilers (FlashInfer, torch.cpp_extension, cmake)
stop guessing. A follow-up step asserts the chosen `nvcc` supports
`compute_89`, failing the run early on a misconfigured host.

**vLLM tests use the docker backend.** Hosting vLLM in
`nvcr.io/nvidia/vllm:26.04-py3` instead of pip means the host's CUDA /
FlashInfer JIT toolchain (or its absence) no longer affects the tests
— the container ships nvcc and a working FlashInfer build. The image's
JIT cache lives at the runner's `/ephemeral/cache/flashinfer/...`,
which is invisible to per-test setup; switching to the container side-
steps it entirely.

**`extra_pip` seam in `xr_ai_vllm.serve`.** The launcher already pip-
installed `hf_transfer` into the NGC container before `vllm serve` ran
(the image hard-errors with `HF_HUB_ENABLE_HF_TRANSFER=1` otherwise).
That seam is generalised: `serve(..., extra_pip=[...])` is threaded
through `_docker.run` → `build_run_argv` and appended to the same
`pip install -q ... && vllm serve ...` shell line. `nemotron_omni`
defaults `extra_pip=["mamba-ssm", "causal-conv1d"]` so its hybrid SSM
backbone — which the NGC image doesn't bundle — loads cleanly. The
knob is `cfg["extra_pip"]`-overridable for version pinning. pip-mode
silently ignores it (deps belong in `pyproject.toml` there).

### 2026-05-14 — `xr-ai-models` seam adopted by migrated workers; one migration pending

`vlm-mcp` (#139), `xr-render-demo` (#140), and `xr-ai-pipecat` (#137) now
depend on `agent-sdk/xr-ai-models` and construct their LLM / VLM / STT / TTS
clients from a per-sample `yaml/models.yaml` via `make_llm` / `make_vlm` /
`make_stt` / `make_tts`.  Per-model quirks (`chat_template_kwargs.enable_thinking`,
`thinking_budget`, `reasoning` vs `reasoning_content` field naming,
served-model-name strings) live in built-in presets — no caller branches on
backend, and swapping a model is a `kind:` + `base_url:` YAML edit.

The seam is consumed by `vlm-mcp` (#139), `xr-render-demo` (#140), and
`xr-ai-pipecat` (#137); `simple-vlm-example`'s worker still uses inline
`httpx` callers and migrates in #138.  The AGENTS.md hard rule
"All HTTP calls to AI services go through `agent-sdk/xr-ai-models`"
becomes universally enforceable on review once #138 lands.

Top-level docs (`README.md`, `docs/ai-services.md`,
`docs/adding-a-sample.md`, `AGENTS.md`) surface the `models.yaml` convention
in the new-sample checklist and the per-service call examples.

See PR #135 (Unit 1, SDK) and Units 2–5 (consumer migrations) for the
individual diffs.

### 2026-05-14 — Introduce `agent-sdk/xr-ai-models` SDK; collapse hand-rolled httpx clients behind four protocols

Before this change, every consumer of an AI service rolled its own `httpx`
wrapper around `/v1/chat/completions` / `/v1/audio/transcriptions` /
`/v1/audio/speech` — `VlmClient` existed in three places (vlm-mcp,
simple-vlm-example/worker/services.py, xr-render-demo/worker/processors.py
where four inline `httpx.post(.../v1/chat/completions)` sites duplicated the
OpenAI request shape); `SttClient` / `TtsClient` lived in both
xr-ai-pipecat and simple-vlm-example.  Per-model quirks
(`chat_template_kwargs.enable_thinking`, `thinking_budget=1024` in the
agentic loop, the `reasoning` vs `reasoning_content` field-name difference
between vLLM's `nano_v3` and `nemotron_v3` reasoning parsers) leaked into
every caller.  Swapping a model meant editing N files.

The new `agent-sdk/xr-ai-models/` package introduces four service
protocols — `LLMService`, `VLMService`, `STTService`, `TTSService` — and
one `OpenAICompat*` implementation per protocol that covers every in-tree
backend (vLLM-served VLM/LLMs, NeMo Parakeet STT, Piper/Magpie TTS) and
any future OpenAI-compatible endpoint.  Worker code constructs services
from a per-sample `yaml/models.yaml` via `make_llm` / `make_vlm` /
`make_stt` / `make_tts`; built-in presets (`cosmos_vlm`,
`llama_nemotron`, `nemotron3_nano`, `nemotron_omni`, `parakeet_stt`,
`piper_tts`, `magpie_tts`) pre-fill the model-specific quirks so a sample
entry only needs `kind: preset:<name>` + `base_url:`.

`ChatResponse.reasoning` is the canonical reasoning surface; the
`reasoning_field` knob normalizes `reasoning_content` (nemotron_v3) into
that one name so callers do not branch.  `enable_thinking` and
`thinking_budget` are typed kwargs on `chat()` that flatten into
`chat_template_kwargs` on the wire — callers never construct that dict.

**Wire-format note (vlm-mcp migration, #139).** Pre-migration, an
explicit `enable_thinking=True` from a caller produced a request with no
`chat_template_kwargs` key at all (the legacy `VlmClient` only emitted
the key when *false*). Post-migration the SDK always emits
`chat_template_kwargs: {"enable_thinking": <bool>}`. Functionally
equivalent — the model still generates `<think>` tokens — but worth
recording for anyone bisecting wire traces across the migration boundary.

**Why not LiteLLM or any-llm-sdk.** Both are excellent for cross-vendor
fan-out but solve a problem we do not have yet — every in-tree backend
already speaks OpenAI-compatible HTTP, and both libraries pass our most
painful quirk (the reasoning-field name) straight through; we would still
write the normalization layer on top.  They would also pull `openai`,
`pydantic`, `tiktoken`, and friends into every worker venv.  The
`factory.py::make_*` `kind` dispatch is the seam where a `LiteLLMBackend`
slots in as a new `kind` later if/when Phase B brings true cross-vendor
needs — protocols and callers do not change.

This is Unit 1 of a multi-PR refactor.  Subsequent units migrate
vlm-mcp, simple-vlm-example, xr-render-demo, and xr-ai-pipecat to depend
on `xr-ai-models` instead of rolling their own clients.

`VLMService` also exposes `ask_video(video, question)`, mirroring
`ask_image`.  The wire format is a `{"type": "video_url", "video_url":
{...}}` content part — what vLLM's Qwen2.5-VL serving expects when
`--limit-mm-per-prompt {"video": >=1}` is set.  `cosmos_vlm` declares
`capabilities: { vision, video }` because Cosmos-Reason1-7B is a
Qwen2.5-VL fine-tune primarily designed for video reasoning; video is
opt-in at the server because vLLM reserves tens of GiB of activation
memory for it at startup.  Callers that haven't enabled video on their
spec get a `ValueError` from `ask_video` rather than a silent server
500.

### 2026-05-14 — CodeQL Advanced Setup (committed workflow) instead of Default Setup

`Analyze (python)` and `Analyze (javascript-typescript)` are required status
checks on `main`. With GitHub's Default Setup (the implicit
`dynamic/github-code-scanning/codeql` workflow), CodeQL only runs when a PR's
diff touches a configured language — so a PR whose diff is entirely C++/CMake
or docs silently skipped analysis and the required contexts never reported,
leaving the PR permanently `mergeStateStatus=BLOCKED` even with an approval
and all visible checks green (first hit by PR #131). We've committed
`.github/workflows/codeql.yml` so the matrix runs on every PR and posts both
required contexts unconditionally. Default Setup must stay disabled in
`Settings → Code security → Code scanning` — Advanced Setup and Default Setup
cannot coexist.

### 2026-05-12 — `FrameSink::InjectVideoFrame` gains a zero-copy overload

The span-only `InjectVideoFrame(std::span<const std::byte>, …)` forced backends to allocate + memcpy the entire pixel buffer on every frame before they could construct an owning `livekit::VideoFrame`. On an embedded 32-bit ARMv7-A target (720p I420 @ 30 fps) this dominated per-frame cost at ~63 ms — the camera HAL collapsed to ~6 fps. Hardware-validated A/B against a new `InjectVideoFrame(std::vector<std::uint8_t>&&, …)` overload that moves the buffer directly into the SDK: total per-frame cost drops from ~70 ms to ~5 ms (14× speedup), matching the in-house baseline publisher.

**Design**: the overload is strictly additive. Default impl forwards to the span overload, so any backend that only overrides one path keeps working. Callers with read-only / shared buffers continue to use the span overload; owning callers get the fast path.

The fix benefits every native C++ consumer (CloudXR, native game-engine plugins, embedded camera SDKs), not just embedded — the embedded case is just where the cost crosses from "wasteful" to "unusable". See finding #12 in [issue #134](https://github.com/NVIDIA/xr-ai/issues/134) for the diagnostic methodology, on-device numbers, and the cross-platform impact estimate.

### 2026-05-12 — `gpu` pytest marker + local-only dev script for hardware-bound tests

Some components (Docker-backed vLLM lifecycle, NVENC paths, anything that
binds a real GPU) cannot be exercised on the GitHub `ubuntu-latest`
runners. Rather than skip them at import time or hide them behind ad-hoc
environment flags, we registered a single `gpu` pytest marker in
`tests/pyproject.toml` (and defensively in `tests/conftest.py::pytest_configure`
so branches that haven't picked up the pyproject change yet don't emit
`PytestUnknownMarkWarning`). CI's pytest invocation in
`.github/workflows/tests.yml` now passes `-m "not gpu"`, and developers
run the hardware-bound suite via `tests/run_local_gpu_tests.sh`, which
just calls `uv sync` then `pytest -m gpu` on the local box. Subsequent
batches that add GPU / Docker / NVENC tests should decorate them with
`@pytest.mark.gpu`; no further wiring is required.

### 2026-05-11 — Native StreamKit `LiveKitBackend` implementation

The stub at `client-samples/native/StreamKit/src/Backends/LiveKit/LiveKitBackend.cpp` is replaced with a working implementation against the upstream LiveKit C++ SDK (`livekit::Room`, `LocalParticipant`, `AudioSource`, `VideoSource`). The backend now covers `Connect` / `Disconnect` / `Send` / data-channel `_agent.status` interception / `FrameSink::InjectVideoFrame` lazy publish. CMake gains `LIVEKIT_SDK_ROOT` + `LIVEKIT_LIB_DIR` cache vars; without `LIVEKIT_SDK_ROOT` the backend compiles header-only in stub mode so CI stays green.

**Design**: shared_ptr (not unique_ptr) holds the opaque LiveKit handles so the destructor is well-defined in translation units that don't include the SDK headers (stub mode). The `livekit::VideoSource(width, height)` ctor requires explicit dimensions, so the LiveKit-backed FrameSink can't honour the "track is published on `StartCamera`" interpretation — instead `StartCamera` arms state and the track is created + published on the first injected frame (consistent with FrameSink's documented contract).

**Left out (called out in the README's "Constraints" table)**: platform mic capture (no portable C++ API), platform camera open (same), `FetchToken` HTTP (host overrides), `AudioConfig::MicrophoneMode` AEC/AGC/NS mapping (would need `AudioProcessingModule`). See [issue #134](https://github.com/NVIDIA/xr-ai/issues/134) for the cross-SDK API friction surfaced during the integration.

### 2026-05-10 — TLS by default; canonicalize the same-origin wss:// proxy; drop the client-side `secure` toggle

`web_server_tls` now defaults to **true**. The hub web server terminates
HTTPS on `web_server_port` (8080) and exposes a same-origin
`wss://<host>:8080/rtc[/<version>]` route that proxies LiveKit signaling
to the internal plaintext 7880 (`_lk_proxy.py`); that proxy is now the
**only** client-facing signaling path. LiveKit's native 7880 stays on
`127.0.0.1` — no external client connects to it directly.

The proxy gained protocol-version-aware paths in the same change: the
LiveKit JS SDK v2.x appends `/rtc/v1` to the base URL it is given (see
`client-sdk-js/src/api/utils.ts::createRtcUrl`), so the proxy now matches
`/rtc/{tail:path}` and forwards the version segment verbatim. The
previous `/rtc`-only routes would have closed v2.x sessions through the
catch-all WebSocket handler.

The Android, iOS, and visionOS samples lost their "HTTPS token" / `secure`
toggle. The `secure` field is gone from `BackendConfiguration`
(`.swift`/`.kt`); the URL is unconditionally `wss://<host>:<port>` and the
default token endpoint is `https://<host>:<port>/token`. The web non-XR
client auto-detects from `window.location.protocol`. The default `port`
field on mobile changed from `7880` → `8080` (the hub web-server port,
*not* LiveKit's native port). iOS persists this via `UserDefaults`, so a
device that had connected with the old default will keep `7880` saved
and needs to be edited in the app — Android's Compose `mutableStateOf`
isn't persisted, so it picks up the new default on next launch.

**Why this over native LiveKit TLS.** `docs/architecture.md` previously
listed three workarounds and called native LiveKit TLS "the correct
long-term fix but has not been implemented yet". On audit, the proxy
(workaround #1) was already in place and exercised by `web-xr`; the
remaining bug was that mobile clients built their own `ws://host:7880`
URL instead of using the URL `/token` returns. Native LiveKit TLS would
have required Rust-side cert verification in `livekit-rtc` for the
internal Python connector talking to its own LiveKit instance — an
uncertain surface to depend on. Canonicalizing the proxy is a smaller
change with the same user-visible outcome.

**What stayed.** `web_server_tls: false` is still a valid escape hatch
for `localhost`-only dev; the same-origin proxy then serves plain `ws://`.
The internal Python connector still talks to LiveKit over plain `ws://`
on `127.0.0.1:7880`. WebRTC media on 7881/TCP fallback and 7882/UDP is
DTLS/SRTP regardless — those ports are unchanged.

**iOS / visionOS cert install is mandatory.** Investigating a real client
rejection: the LiveKit Swift SDK's `URLSession` (`WebSocket.swift`
`Delegate`) does not implement
`urlSession(_:didReceive:completionHandler:)` for server-trust
challenges, and ATS does not bypass cert-chain validation regardless of
`NSAllowsArbitraryLoads` — the previous code comment claiming otherwise
was wrong. The `TrustingSessionDelegate` inside `LiveKitBackend.swift`
only covers the `/token` HTTP fetch, and was additionally gated by
`#if DEBUG` (so Release builds rejected the cert there too). The
`#if DEBUG` gate is gone, and the hub now exposes `/cert` (MIME
`application/x-x509-ca-cert`) so iOS Safari can install the hub's
self-signed cert in one tap. The iOS README + `docs/networking.md` +
`docs/troubleshooting.md` document the install + Full Trust toggle.
Android also requires cert install; see the tech-debt cleanup sub-entry below.

**The Full Trust toggle did not appear in initial testing** because
`_tls.py` generated the cert with `BasicConstraints CA:FALSE`; iOS only
exposes the toggle for CA-marked certs. The generator now writes a
self-signed CA (the mkcert pattern: `CA:TRUE`, `KeyUsage` with both
`key_cert_sign` and `digital_signature` + `key_encipherment`, EKU
`SERVER_AUTH`) and `ensure_self_signed_cert()` detects a cached non-CA
cert from older builds and regenerates with a one-line log banner.
Devices that installed the old profile must remove it from VPN &
Device Management before reinstalling the new one.

**The cert's SAN missed the LAN IP** in a second round of iOS testing
(`errSSLBadCert` / NSURLErrorDomain `-1202`, "pretending to be
10.29.90.196"). The previous SAN-population path used
`socket.gethostbyname(socket.gethostname())`, which on Ubuntu returns
the `/etc/hosts` loopback alias `127.0.1.1` rather than the LAN IP.
`_tls.py` now enumerates routable IPv4 addresses via the UDP-connect
trick (open a non-blocking UDP socket to `8.8.8.8` / `1.1.1.1` /
`169.254.169.254` and ask the kernel which interface it would route
from) and `gethostbyname_ex` for /etc/hosts aliases, then includes every
non-loopback hit in the SAN. `ensure_self_signed_cert()` also detects a
SAN that's missing a currently-detected local IP and regenerates with a
log banner directing users to reinstall the profile.

**Third iOS issue: 401 from LiveKit after a successful TLS handshake.**
The LiveKit Swift SDK 2.13.0 puts the JWT in
`Authorization: Bearer <token>` (the JS SDK uses `?access_token=…` in
the query string instead). The same-origin proxy in `_lk_proxy.py`
forwarded the query string but dropped every request header, so
LiveKit-server saw an unauthenticated WSS upgrade from the Swift SDK
and 401'd it. The proxy now forwards end-to-end headers (everything
except a hop-by-hop allowlist) to both the `/rtc/validate` HTTP shim and
the `/rtc[/<version>]` WebSocket, via httpx and `websockets.connect`'s
`additional_headers`. The web client is unaffected — it puts the token
in the query string and worked before.

**Tech-debt cleanup: `TrustAllCerts.kt` removed; Android now requires cert
install like iOS.** `TrustAllCerts.kt` was not "trust our specific self-signed
cert" — it accepted any cert from any issuer, including legitimately-misissued
or attacker-controlled ones, defeating TLS entirely for the Android client.
The file is deleted. `LiveKitOverrides` (which existed solely to inject the
trust-all OkHttpClient) is dropped from `LiveKitBackend.kt` as well. Android
now validates against the system + user CA store identically to iOS. To
replace the manual URL-typing step, both apps gained an in-app **"Install hub
certificate"** button in the Connection section (enabled when Host is non-empty,
visible in the disconnected state). Android fetches the cert from
`https://<host>:<port>/cert` using a single-connection trust-all
`HttpsURLConnection` scoped to that one call only (the cert cannot be
pre-validated because it is what we are installing), then opens the system
`KeyChain` install dialog. iOS opens Safari at the same URL, which triggers
the existing profile-install flow. The `network_security_config.xml`
`<certificates src="user"/>` anchor — already present — is the mechanism that
makes Android accept the cert once it is OS-installed; no domain-config
exception was needed or added.

### 2026-05-05 — Docker backend option for vLLM-backed servers

All four vLLM-backed services (`vlm_server`, `llama_nemotron_llm_server`,
`nemotron3_nano_llm_server`, `nemotron_omni_llm_server`) gained an opt-in
`vllm_backend: docker` mode that hosts vLLM via the NVIDIA NGC container
(`nvcr.io/nvidia/vllm:26.04-py3` by default) instead of the pip-installed
`vllm` CLI. Goal: let users try NVIDIA's optimized vLLM build without
disturbing the pip path that everything ships on.

(Cleanup side-effect: `llama_nemotron` was already a vLLM launcher despite
being described as "transformers in-process (+ LMFE)" in the docs — the
description was stale from a prior implementation. `DEPENDENCIES.md`,
`docs/ai-services.md`, and the service's `README.md` were corrected in the
same change.)

**Design.** Backend selection is per-server via a new YAML key:

```yaml
vllm_backend: pip       # default — today's behavior
# or
vllm_backend: docker
vllm_image:   nvcr.io/nvidia/vllm:26.04-py3
```

The `Process(...)` declaration in each orchestrator stays identical; flipping
backends is one YAML edit. Both modes honor the same `model:`, `port:`, and
vLLM-flag config — only the runtime that interprets them differs.

A new stdlib-only utility package `utils/xr-ai-vllm/` exposes `serve(...)`
(dispatches pip vs docker) and `stop_persistent_servers(...)` (docker-aware
cleanup that tries `docker stop <container_name>` first, falls back to the
existing port → PID → SIGTERM/SIGKILL path for pip mode). Each of the four
wrapper `__main__.py` files now reads YAML, builds its model-specific
`extra_serve_args`, and calls `serve()` — the per-service logic
(`_gpu_compute_major()` quant selection, nano_v3 parser fetch,
`media_io_kwargs` for omni, `llama3_json` tool-call parser for
llama_nemotron) stays in the wrapper because it is not runtime-agnostic.

**Why a shared helper instead of triplicated logic.** Four concrete consumers
justify the abstraction per the AGENTS.md rule. Keeping it stdlib-only matters
because docker mode's whole point is to avoid pulling vllm/torch into the
wrapper's venv when it is not actually used; the helper sits beside
`xr-ai-launcher` (also stdlib-only) at `utils/xr-ai-vllm/`.

**Why per-YAML toggle, not separate sibling packages.** The user-visible
change is "the same vLLM, hosted differently". A YAML key matches that
exactly; sibling packages would have meant duplicating every service's
pyproject, README, and per-profile YAML for a runtime-only swap. The plain
boolean `Process(...)` line stays unchanged.

**Why detached docker containers (not foreground).** `vlm_server`,
`llama_nemotron_llm_server`, and `nemotron3_nano_llm_server` already implement
vLLM persistence in pip mode via `start_new_session=True` so the container
survives wrapper restarts — loading 16–30 GB of weights every dev iteration
would be a regression. Docker mode mirrors this with
`docker run -d --rm --name xr-ai-vllm-<service>`; cleanup is via
`xr_ai_vllm.stop_persistent_servers()` invoked from `xr_render_demo --stop`.
`nemotron_omni_llm_server` was non-persistent in pip mode (used `os.execvp`)
and is non-persistent in docker mode (foreground `docker run --rm`) — same
shutdown semantics for both runtimes.

**Why `--network host` + `--ipc host`.** Mirrors the LiveKit container in
`server-runtime/xr_media_hub/transport/livekit/_docker.py` (the only other
docker-managed subprocess in the repo) and gives vLLM workers the shared
memory region they need for KV cache shards. Linux-only, which the repo
already requires.

**NGC auth.** The wrapper auto-runs `docker login nvcr.io --password-stdin`
when `NGC_API_KEY` is in the environment (loaded by `load_credentials()` per
`docs/credentials.md`) and no existing auth is found in `~/.docker/config.json`.
Existing logins are not overwritten.

**Side-effect bug fix.** `nemotron_omni_llm_server` previously used
`os.execvp` which never touched the launcher's `--ready-file`. Routing it
through `xr_ai_vllm.serve(persistent=False, ...)` adds the standard
`/health` poll + ready-file touch path, so the launcher's `_wait_ready` no
longer hangs when this service is selected.

**As-was.** Defaults, ports, model IDs, `Process(...)` lines, persistence
semantics, and the `--stop` UX are all unchanged for users who do not flip
`vllm_backend:`. Existing pip-mode `model_cache` weights are reused by docker
mode (mounted at the same path inside the container) and vice versa.

### 2026-05-05 — Unified loguru stack; `launcher/` and `xr-ai-logging/` consolidated under `utils/`

Two related infrastructure changes shipped together.

**Loguru migration.** New `xr-ai-logging` package wraps loguru with a single
`setup_logging(name, namespace=...)` entry point that every process calls
once at startup. Installs a stderr sink (INFO by default, DEBUG when
`XR_AI_VERBOSE` is truthy), an always-DEBUG file sink at
`/tmp/log_<namespace>_<YYYY-MM-DD_HH-MM-SS>/<process>.log`, and a stdlib
`logging` -> loguru bridge so `utils/xr-ai-launcher/` (stdlib-only by
contract) and `agent-sdk/xr_ai_agent/` (pyzmq+msgpack-only by contract)
participate without importing loguru. Subprocess coordination uses three
stamped env vars (`XR_AI_LOG_NAMESPACE` / `XR_AI_LOG_TIMESTAMP` /
`XR_AI_LOG_ROOT`). Stderr-vs-file split lets the user keep a quiet console
while retaining full DEBUG history per run.

The launcher's child-stdout/stderr forwarder also moved from raw `print()`
to a level-aware `log.<level>(...)` (parses the loguru level from each
captured line and re-emits at that level), so library banners (NeMo,
OpenXR loader, LOVR Vulkan) stay out of the default console but are
preserved in the file sink. ~16 INFO calls were demoted to DEBUG (per-data-
message, per-NVENC-chunk, per-tool-call duplicates, per-VAD-false-positive,
etc.) so INFO is now strictly lifecycle / once-per-utterance / periodic
stats.

**`utils/` consolidation.** Both `launcher/` and `agent-sdk/xr-ai-logging/`
are pure infrastructure used by every process, not specific to agents.
Moved to `utils/xr-ai-launcher/` and `utils/xr-ai-logging/` so the layout
reflects actual scope. The `xr-ai-launcher` "stdlib-only" rule still
applies — `utils/xr-ai-launcher/pyproject.toml` keeps `dependencies = []`.
`utils/xr-ai-logging/` has its own pyproject (`loguru>=0.7`). Python import
paths (`xr_ai_launcher`, `xr_ai_logging`) are unchanged; only filesystem
paths in `[tool.uv.sources]` and doc references shifted.

### 2026-05-05 — vLLM model persistence across stack restarts

vLLM-backed servers (`vlm_server`, `llama_nemotron_llm_server`,
`nemotron3_nano_llm_server`) now survive stack shutdowns so model weights stay
loaded across worker crashes and debug restarts.

**Mechanism:** each wrapper checks its own `/health` endpoint before spawning
vLLM.  If already healthy the wrapper signals ready immediately and idles (exits
cleanly on SIGTERM without touching vLLM).  If not healthy it spawns vLLM
normally.  vLLM itself is started with `start_new_session=True` so the
launcher's `killpg()` does not reach it.

**Cleanup:** `uv run xr_render_demo --stop` from the sample directory hits each
server's `/health`, finds the PID via `ss`/`lsof`, and sends SIGTERM (escalates
to SIGKILL after 20 s).

**Why this approach over launcher-level `persistent=` flag:** keeps `main.py`
and `_stack.py` unchanged; persistence is a detail of each service's own
startup script, not the orchestrator.

### 2026-05-01 — visionOS Enterprise license bundling

Apple Vision Pro main-camera passthrough
(`com.apple.developer.arkit.main-camera-access.allow`) requires the entitlement
signed into the binary **and** a per-team `Enterprise.license` file bundled
into the `.app`. Without the license the API is a silent no-op
(`CameraVideoFormat.supportedVideoFormats(...)` returns `[]`, LiveKit AR camera
publish fails with `LiveKitError.invalidState`). visionOS auto-loads the file
from the app bundle.

The license file is per-team and Apple's terms restrict redistribution, so it
is gitignored (`**/Enterprise.license`) rather than committed. A placeholder
`App/Enterprise.license.sample` documents the path for new contributors. An
Xcode "Copy Enterprise.license" build phase copies the file into the `.app`
at build time; if missing, the build succeeds with a warning and the camera
path no-ops at runtime (audio + data + simulator GIF feed are unaffected).
Symlinks at the expected path are supported and still gitignored.

The sample's display name (`StreamKitSample`) is intentionally decoupled from
its Bundle ID (`com.nvidia.xr-ai-example`) so a fork that renames the Bundle
ID still ships under the same on-device app name.

### 2026-04-30 — Unified MCP IDs: identity sidecars, live vs recorded splits, transcript source_id

The MCP servers had two consistency gaps: (1) `list_*` tools returned
sanitized filesystem names rather than the original LiveKit identities,
so a caller round-tripping a value through `list_recording_participants`
→ `get_latest_frame` could miss; (2) the transcript store named its
key `participant_id` even though transcripts can come from non-
participant sources (e.g. an agent's own TTS).

Changes:

- **Recorder + stores write a `.identity` sidecar per source.** The
  hub's `_recorder.py` writes `<recordings_dir>/<safe>/.identity`;
  transcript-mcp writes `<transcripts_dir>/<safe>.identity` next to
  the JSONL. Sidecar contents are the raw caller-supplied ID verbatim.
  Collisions between distinct raw IDs that happen to share a
  `_safe_name` get a counter suffix (`alice_home`, `alice_home_2`, …).
- **List tools return raw IDs.** `list_recorded_participants` (renamed
  from `list_recording_participants`) and `list_sources` (renamed from
  `list_participants` on transcript-mcp) read sidecars and return
  exactly what the writer passed in.
- **New tool `list_live_participants`** on video-mcp — surfaces
  `ep.connected_participants` from the ProcessorEndpoint so callers
  can ask "who's actually live right now?". This is the only set
  `get_latest_frame` will succeed for.
- **Transcript-mcp renames `participant_id` → `source_id`** in tool
  signatures, response keys, and stored identity sidecars. The store
  treats `source_id` as opaque, allowing agents to write under
  internal names (`"agent-vlm"`, `"tts"`) alongside live participant
  records. video-mcp keeps `participant_id` since video really does
  come from real participants.
- mcp-agent worker updated to use `source_id` when calling transcript
  tools.

**Why:** the underlying storage was always string-keyed and didn't
care, but the API leaked sanitized filenames and overloaded
"participant" semantics onto things that aren't participants. The
sidecar lifts the raw name back out cleanly; the rename names the
field for what it actually is.

### 2026-04-29 — Video recording on tmpfs; video-mcp gains live-frame + frame-at-time

`server-runtime/xr_media_hub/video/_recorder.py`:
- Default `out_dir` flipped from `/tmp/xr_recordings` (disk) to
  `/dev/shm/xr-ai/recordings` (tmpfs — RAM-backed). Writes don't touch
  disk by default.
- Eviction policy is now **size-based, global**: `max_total_bytes`
  (default 500 MB) caps total chunk size across all participants.
  When the cap is exceeded, oldest chunks are evicted FIFO. Replaces
  the prior per-participant `max_chunks` count.

`agent-mcp-servers/video-mcp/`:
- Now connects to the hub as a `ProcessorEndpoint` with
  `filter=Subscribe.VIDEO`. A small `FrameProvider` tracks the most
  recent `FrameSignal` per pid; pixel bytes are pulled on demand via
  `request_frame()`. No on-disk side-channel — the live path is
  entirely IPC-based.
- New MCP tool `get_latest_frame(participant_id)` — calls into the
  provider, converts the returned `FrameData` to RGB, writes a PNG to
  `out_dir`, returns `{path, width, height, timestamp_us, track_id}`.
- New MCP tool `get_frame_at_time(participant_id, timestamp_us)` —
  finds the chunk covering the timestamp, decodes it with NVDEC via
  PyNvVideoCodec, picks the frame closest to the timestamp by linear
  interpolation across the chunk, encodes PNG, returns `{path, width,
  height, timestamp_us, chunk_path}`.
- video-mcp gains `xr-ai-agent`, `PyNvVideoCodec`, `Pillow`, and
  `numpy` runtime deps. mcp-agent's composed `mcp_server` adopts the
  same model — owns its own `ProcessorEndpoint` and lifecycle.

**Why:** disk IO was wasted overhead for chunks that almost always get
evicted within minutes; `/dev/shm` cuts the IO cost to RAM bandwidth
without changing the file-based interface that the video-mcp uses for
historical queries. Live frames bypass the chunk store entirely — the
hub already has the most recent SHM slot held open per (pid, track),
so a `request_frame()` is a single zero-copy memcpy at the hub plus a
pixel-format conversion at the consumer.

### 2026-04-29 — MCP servers go pure FastMCP

`transcript-mcp-server`, `video-mcp-server`, and the composed `mcp-server`
in `agent-samples/mcp-agent/` no longer wrap a FastAPI app. Each runs the
`FastMCP.http_app(path="/mcp")` Starlette app directly under uvicorn.

- All worker ingress is now an MCP tool call. Transcript ingest is the new
  `transcript_add_transcript` tool (replaces `POST /ingest`); stats fetches
  use the existing `transcript_get_transcript_stats` / `video_get_video_stats`
  tools (replace the `/transcript/stats/{pid}` and `/video/stats/{pid}` REST
  routes).
- The composed `mcp-server` no longer reads a `skills:` config block — both
  sub-servers are always mounted; their per-server config lives at the top
  level of `mcp_server.yaml` under `transcript:` and `video:`.
- `/health` is gone. The mcp-agent worker's readiness probe now uses
  `fastmcp.Client.list_tools()` against `/mcp` to confirm the server is
  serving (a stronger guarantee than a 200 from `/health` ever was).
- Drops the `fastapi` and `pydantic` runtime dependencies on transcript-mcp,
  video-mcp, and the composed mcp-server. Worker gains a `fastmcp>=0.4`
  dependency.

**Why:** the dual REST + MCP surface had no value once workers got an MCP
client. Two interfaces meant two contracts to keep in sync, two error-
handling paths, and two readiness checks. Pure FastMCP is one contract,
one error model, and a stronger readiness check via `list_tools()`.

### 2026-04-29 — Participant-keyed agent subscriptions

`ProcessorEndpoint` now models subscriptions as **participants**, not topic
prefixes. The unit of opt-in is "I want everything for participant X";
categories (data / audio / video) are an opt-out filter inside that.

- New `Subscribe` flag enum (`DATA`, `AUDIO`, `VIDEO`, `ALL`) replaces the
  prior raw-bytes `topics=` parameter.
- `ProcessorEndpoint(auto_subscribe=True, filter=Subscribe.ALL)` is the
  default. The endpoint installs an internal participant handler that
  calls `subscribe(pid)` on join and `unsubscribe(pid)` on leave —
  agents see every client's full inbound stream out of the box.
- `ep.subscribe(pid, filter=...)` and `ep.unsubscribe(pid)` are the
  primitives. Idempotent. Calling subscribe with a different filter
  diffs the active subscriptions. Subscribing before the pid joins is
  fine — ZMQ holds the SUBSCRIBE.
- `auto_subscribe=False` is the escape hatch for single-client agents:
  the agent only sees `participant` + `control` until it explicitly
  subscribes. Use `ep.subscribed_participants` to introspect live state.
- New `ROSTER_REQUEST` IPC type (`MsgType.ROSTER_REQUEST = 12`).
  `request_roster()` (called automatically once at the start of
  `run()` when auto-subscribe is on) makes the hub re-publish
  `PARTICIPANT_EVENT(joined=True)` for every current pid so endpoints
  started mid-session catch up. Replays go on the regular `participant`
  topic, so other endpoints' `on_participant` callbacks may fire again
  for known pids — keep them idempotent.
- Topic prefixes always include the trailing `.` so `data.alice.` does
  not bleed into `data.alice2.chat`. The helper centralises this so
  individual agents never type the prefix themselves.

**Why:** the previous bytes-tuple `topics=` parameter forced agents to
know hub-internal topic conventions, made per-pid scoping awkward
(write your own join handler, remember the trailing dot), and didn't
solve the mid-session catch-up problem. The new model treats the
participant as the unit of subscription, which is what every real
agent actually wants — most just take the default broadcast; the few
that need scoping flip `auto_subscribe=False` and call `subscribe`.

### 2026-04-29 — Multi-client / multi-agent isolation; topic surface; tests

The hub formally supports many clients and many agents at once. The IPC and
LiveKit transport layers were extended so:

- **Per-participant return audio** — `RoomClient` now publishes one
  `xr-hub-return-{pid}` audio track per active participant, with subscribe
  permissions restricted via `set_track_subscription_permissions` so a
  participant can only hear their own return audio. Tracks are unpublished
  on participant leave.
- **Targeted return data** — `RoomClient.send_return_data` passes
  `destination_identities=[participant_id]` so return text/binary is no
  longer broadcast to other participants in the room.
- **`ReturnAudioFlush` control message** (`MsgType.RETURN_AUDIO_FLUSH = 11`)
  added to `xr-ai-agent`. `ProcessorEndpoint.flush_return_audio(pid)`
  routes through the hub on `return_audio_flush.<pid>` to the connector,
  which calls `AudioSource.clear_queue()` for that pid only. Used to
  cleanly interrupt agent TTS playback when a new query arrives.
- **StreamKit `onDataReceived(topic, data)`** — the previously dropped
  data-channel `topic` is now surfaced to the application across web and
  iOS/visionOS. The reserved `_agent.status` topic is still intercepted
  internally and never reaches `onDataReceived`.
- **`tests/` top-level suite** — multi-client / multi-agent coverage over
  the real IPC layer (no Docker / LiveKit needed). CI workflow at
  `.github/workflows/tests.yml` runs the suite on every push and PR
  across Python 3.11 and 3.12.

### 2026-04-30 — LLM servers reorganized into per-model packages

`ai-services/llm-server/` (single package, single model) is split into two
sibling packages under `ai-services/llm/`, each with its own entry-point
command, YAML, default port, and dependency set. This lets a sample pick the
LLM that matches its tool-calling / reasoning / hardware requirements without
dragging in the dependencies of the others (notably vLLM and
`lm-format-enforcer`).

| New package | Command | Port | Model | Backend |
|---|---|---|---|---|
| `llm/llama_nemotron/` | `llama_nemotron_llm_server` | 8106 | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | HF transformers + `lm-format-enforcer` — native tool calls, reasoning toggle |
| `llm/nemotron3_nano/` | `nemotron3_nano_llm_server` | 8107 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | vLLM (execvp shim) — Blackwell FP4 MoE |

- **HTTP contract is identical** across both (OpenAI-compatible
  `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`). Workers point
  at a different port to swap backends — no worker-side code changes.
- **Ports** chosen to be non-overlapping so both LLM backends can coexist in
  the same stack if a sample actually wants that (unusual; typically pick one).
- **`llama_nemotron`** adds grammar-constrained tool-call decoding via
  `lm-format-enforcer`. When `tools=[...]` is present in the request, a
  `UnionParser([tool_call_grammar, free_text])` is fed as
  `prefix_allowed_tokens_fn` so the model's vocabulary is masked every step to
  either valid `<TOOLCALL>[{...}]</TOOLCALL>` JSON or plain assistant text.
  Rationale: the native Llama-3.1 chat template instructs the model to emit
  tool calls as JSON, but sampling noise / schema drift can still produce
  syntactically broken output. LMFE eliminates that entirely.
- **`nemotron3_nano`** is intentionally thin (~200 lines). vLLM already
  exposes the OpenAI API, parses Nemotron-3-Nano's XML tool-call format via
  `--tool-call-parser qwen3_coder`, and splits the `<think>…</think>` preamble
  via `--reasoning-parser nano_v3` (custom plugin auto-fetched from the model
  card into `model_cache`). The shim reads the YAML, sets
  `VLLM_USE_FLASHINFER_MOE_FP4=1`, and `os.execvp`s into `vllm serve` so the
  launcher's signals go straight to vLLM with no intermediate wiring.
- **`enforce_eager: true`** is the default for `nemotron3_nano` — CUDA graph
  capture plus FlashInfer FP4 MoE autotune are silent and take 3–8 min on
  first run, which is a bad UX for a voice agent waiting to become healthy.
  Eager mode starts in ~5 s after weight load and is 10–20% slower per token
  (imperceptible at <250 tokens/turn where STT+VAD+TTS already dominate).

Dependency fan-out stays contained: only `llama_nemotron` pulls
`lm-format-enforcer`, only `nemotron3_nano` pulls `vllm>=0.12.0`.

### 2026-04-29 — render-mcp + oxr-mcp added; xr-render-demo as integration

Two new MCP servers under `agent-mcp-servers/`, port-per-server, no LiveKit
dep. oxr-mcp is pure FastMCP; render-mcp mixes one streaming HTTP route
with FastMCP tools.

**render-mcp** (`agent-mcp-servers/render-mcp/`, port 8220) — owns the LOVR
child (the OpenXR rendering app) and is the only process that pushes ops
onto LOVR's `scene_socket` (msgpack over ZMQ PUSH).

- **`POST /sphere/radius` is a plain FastAPI route**, not an MCP tool.
  The worker hits it ~50 Hz from the audio path; routing a streaming
  control signal through FastMCP's per-request dispatch + JSON-RPC
  envelope is the wrong shape and makes the server log unreadably chatty.
  The discrete operations (`start_xr`, `set_sphere_color`, …) stay on
  `/mcp` where an LLM agent can discover and drive them.

- **`xr.session.started` gates LOVR spawn.** CloudXR returns
  `XR_ERROR_FORM_FACTOR_UNAVAILABLE` from `xrGetSystem` until a streaming
  client has actually connected. Spawning LOVR at process start lands it
  in the desktop simulator forever. The caller is expected to call
  `start_xr` only after seeing the streaming client come up.
- **`start_xr` returns immediately; caller polls `get_health.lovr_started`.**
  The cloudxr readiness wait can take a minute; matching a single tool
  call's timeout to it would couple two unrelated knobs. render-mcp spawns
  LOVR + waits for cloudxr in a background task, caches terminal failures
  so retries fail fast, and exposes progress through `get_health`.

**oxr-mcp** (`agent-mcp-servers/oxr-mcp/`, port 8230) — exposes head pose
through a `get_head_pose()` MCP tool.

- **Two OpenXR sessions, one CloudXR.** LOVR holds the rendering session;
  oxr-mcp opens a SECOND headless session (`XR_MND_HEADLESS`) for pose
  only. Verified empirically: pos/quat update from the headset while LOVR
  keeps streaming pixels, no contention. Session opens lazily on first
  `/pose` request, so it doesn't fight CloudXR's startup either.

**Shared infra** — `launcher/_cloudxr_env.py`. Both MCPs need to wait for
`cloudxr.env`, source it, and wait for `runtime_started` before opening
their OpenXR sessions; the launcher (which already manages the cloudxr
child) is the natural home.

**xr-render-demo** (`agent-samples/xr-render-demo/`) — integration sample.
Web client streams mic audio; the worker computes RMS → sphere radius
continuously and runs VAD → STT → LLM whose JSON action list it translates
into render-mcp HTTP calls.

- **User-frame coordinates with worker-side transform.** The LLM emits
  user-frame coordinates (`+x` user's right, `-z` in front of the user).
  The worker fetches head pose from oxr-mcp once per utterance, rotates by
  yaw + translates by head position before forwarding to render-mcp.
  Putting the transform in the worker keeps render-mcp transport-agnostic
  and means the LLM never has to learn vector math.

### 2026-04-27 — MCP example: transcript + video MCP servers; NVENC recording in hub

`agent-samples/mcp-agent/` added as a demonstration of MCP integration with XR data.

**Transcript MCP server** (`agent-mcp-servers/transcript-mcp/`, port 8200):
- Single FastAPI process hosts both the non-MCP HTTP ingest endpoint (`POST /ingest`)
  and the FastMCP tools (`/mcp`) so agents can query historical transcripts.
- Agent workers POST transcripts over plain HTTP; MCP is for LLM tool-use only.
- JSONL storage persists across server restarts; one file per participant.

**Video MCP server** (`agent-mcp-servers/video-mcp/`, port 8210):
- Thin FastMCP wrapper around the hub video HTTP API (`GET /video`).
  Fetches the concatenated H.264 byte stream, writes it to a temp file, returns path.
- Kept separate from the transcript server so either can be used independently.

**Hub NVENC video recording** (`server-runtime/xr_media_hub/video/_recorder.py`):
- Opt-in via `video_recording.enabled: true` in `xr_media_hub.yaml`.
- Uses `PyNvVideoCodec` (on PyPI) for NVENC encoding; included in the standard `uv sync`.
  The config guard (`enabled: true`) prevents instantiation when recording is not needed.
- VBR mode, no B-frames (`bf=0`), `repeat_sps_pps=1`.  Each chunk uses a fresh encoder
  session so it always begins with SPS+PPS+IDR and is independently decodable.
  Chunks are binary-concatenable with `cat`.
- Hub exposes a video query HTTP API on port 8090 (`GET /video?pid=&start_us=&end_us=`).

**PyNvVideoCodec pitfalls (hard-won)**:
- `Encode()` must receive a **2D numpy array** of shape `(H*3//2, W)` — do **not** call
  `.flatten()`.  NVENC reads the array using numpy strides to determine the row pitch.
  A 1D array causes NVENC to assume an internally aligned pitch (e.g. 512 for W=320),
  producing a circular horizontal shift in every decoded frame.
- `GetSequenceParams()` does not exist in PyNvVideoCodec 2.x.  Use `repeat_sps_pps=1`
  in `CreateEncoder` kwargs instead; it prepends SPS+PPS automatically before each IDR.
- WebRTC adaptive bitrate changes the frame resolution mid-stream.  The encoder must be
  recreated (and the current chunk flushed) whenever `width` or `height` changes.  Feeding
  wrong-sized frames to the encoder silently corrupts all subsequent output.
- There is no reliable option that forces repeated IDR frames in PyNvVideoCodec 2.x.
  `gopLength`, `gop`, `idrPeriod` were all tested — NVENC only emits one IDR at the start
  of a session regardless.  Use per-chunk fresh encoders (`EndEncode` → `CreateEncoder`)
  to guarantee IDR boundaries; each new encoder session always begins its output with IDR.

**mcp-agent worker** (`agent-samples/mcp-agent/worker/`):
- Runs continuous STT (same VAD logic as echo-agent).
- POSTs each final utterance to the transcript-mcp-server over HTTP.
- Does not speak TTS — pure observation/logging pipeline.

### 2026-04-24 — AI inference servers added; NVIDIA models; shared model cache

`ai-services/` added as a sibling of `server-runtime/`, containing three reusable
OpenAI-compatible HTTP inference servers.

Model choices — all NVIDIA:
- **vlm-server**: `nvidia/Cosmos-Reason1-7B` in-process via HuggingFace
  transformers (Qwen2.5-VL architecture).  Accepts base64 image_url in messages.
- **stt-server**: `nvidia/parakeet-tdt-0.6b-v3` in-process via NeMo ASR.
  English-only TDT model, CC-BY-4.0.  ~1.5 GB VRAM.
- **tts/magpie**: `nvidia/magpie_tts_multilingual_357m` in-process via NeMo TTS.
  Multilingual, NVIDIA Open Model License.  ~1 GB VRAM.
- **tts/piper**: any rhasspy/piper-voices ONNX voice; ~100 ms/sentence on CPU.

Shared model cache: all weights land in `models/` at the repo root (gitignored).
Each YAML configures `model_cache` (resolved relative to the YAML file) so the
same physical directory is used regardless of which sample root the YAML is in.

Sample YAMLs for all four services ship with `mcp-agent` as a template.

OpenAI-compatible APIs chosen so workers never need to know backend details —
swap models by changing the YAML only.

### 2026-04-22 — CloudXR runtime extracted to top-level shared component

`cloudxr-runtime/` added as a peer of `server-runtime/`, wrapping
`isaacteleop[cloudxr]` (NVIDIA IsaacTeleop SDK).  Samples opt-in by adding
`Process("cloudxr", "../../cloudxr-runtime", "cloudxr_runtime")` to their
`PROCESSES` list and providing a `cloudxr_runtime.yaml` in the sample root.

The native CloudXR service runs entirely as a local process (no Docker).
`isaacteleop`'s Python `wss_run()` provides a TLS WebSocket proxy on port 48322
required for `auto-webrtc` profile; `auto-native` does not need it.
CloudXR and the hub are fully independent: CloudXR streams rendered/sim content
to XR devices over WebRTC while the hub handles agent media via LiveKit.

### 2026-04-22 — Launchable convention + StackLauncher

Each runnable sub-project (hub, worker, future CloudXR runtime, MCP servers) is a
**launchable**: an entry-point command + an optional `<command>.yaml` config.
The launcher discovers YAML files automatically by convention — no separate
launcher config file (the previous `stack.toml` idea was dropped).

The orchestrator code declares the process sequence using `Process` + `run_stack`.
All processes start concurrently; startup order does not matter because every
launchable must be resilient to peers not being ready (ZMQ reconnects, etc.).
`run_stack` is fail-fast: any process exit terminates the whole stack.

`launcher/` gained `Process`, `StackLauncher`, and `run_stack` (all stdlib-only).
`HubLauncher` / `ProjectLauncher` remain as lower-level building blocks.

### 2026-04-21 — Agent-SDK extracted; samples use orchestrator + worker subprocess model

`agent-sdk/` (`xr-ai-agent`) was extracted as a standalone package with only
`pyzmq` + `msgpack` as runtime dependencies. The four IPC client modules
(`_types`, `_codec`, `_shm`, `_processor`) moved there from `server-runtime`.
`server-runtime/xr_media_hub/ipc/__init__` re-exports everything for backwards compat.

Each sample now has two entry points:
- **Orchestrator** (`<name>`): stdlib + `xr-ai-launcher` only. Uses `HubLauncher`
  (which runs the hub via `uv run --project server-runtime`) and `ProjectLauncher`
  (which runs the worker via `uv run --project .`). Waits for the worker to exit.
- **Worker** (`<name>_worker`): imports only from `xr_ai_agent`. Contains all
  agent logic. Launched as a subprocess by the orchestrator.

`launcher/` gained `ProjectLauncher` — a generic context manager that runs any
uv project command as a managed subprocess in its own isolated venv, yielding
the `asyncio.subprocess.Process` for lifecycle control.

**Why:** complete venv isolation between hub (server-runtime), agent (sample), and
orchestrator (launcher-only). No cross-contamination of server deps into agent
venvs and vice versa. `uv run --project` is the mechanism — uv resolves and caches
each project's venv independently.

### 2026-04-21 — VLM agent sample added

`agent-samples/vlm-agent/` — answers natural-language queries about live XR
video using a locally-hosted vision-language model.
**Model:** `nvidia/Cosmos-Reason1-7B` (NVIDIA Open Model License + Apache 2.0,
commercial use permitted; ~16 GB VRAM at BF16). Architecture:
`Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` + `qwen-vl-utils`.
**Protocol:** client sends `vlm.query` data message (raw text or
`{"query":"…","track_id":"…"}`); agent replies on `vlm.response`.
**Frame flow:** `on_frame()` tracks latest `FrameSignal` per (participant,
track); on query, `request_frame(signal)` pulls a pixel copy, converts to PIL
via numpy (I420/NV12/RGB24/RGBA/BGRA), then calls `_VlmBackend.infer()` in a
thread pool so the asyncio loop is not blocked. Model is loaded lazily on the
first query. Override model via `VLM_MODEL` env var.

### 2026-04-21 — Process management moved to `launcher/`

`HubLauncher` lives in `launcher/xr_ai_launcher/`, not in `server-runtime`.
**Why:** process management should not be part of the processes it manages.
The launcher will eventually start MCP servers, CloudXR runtime, and other
components — keeping it separate keeps dependency chains lean and the boundary
clean. `launcher/` has zero runtime dependencies (stdlib only).

### 2026-04-21 — NVDEC/NVENC required; OpenH264 must not be used

`LiveKitConnector.start()` calls `require_nvidia_video_codecs()` before doing
anything else. It checks for `libnvcuvid.so` (NVDEC) and `libnvidia-encode.so`
(NVENC) via ctypes and raises `RuntimeError` if either is absent (Linux only).
**Why:** `livekit-rtc` bundles `libwebrtc` which includes OpenH264 as a software
fallback. OpenH264 is royalty-bearing for end users and must not ship in this
product. The guard prevents silent fallback at the cost of a hard startup failure.
In Docker: `--gpus all` or `--device /dev/nvidia*` must be passed.

### 2026-04-21 — Video frame delivery: metadata push, pixel pull

Processors receive `FrameSignal` metadata at full frame rate via `on_frame()`.
Pixel data is only copied when the processor calls `await ep.request_frame(signal)`.
The hub holds one SHM slot per (participant, track) — always the latest frame.
The slot stays `_STATE_READY` (not released to the connector) until the next frame
arrives for the same track, so `bytes(view.data)` in FRAME_REQUEST is safe.
**Why:** avoids copying every frame over IPC; agents sample at their own rate.
Concurrent `request_frame()` calls for the same track are coalesced into one
FRAME_REQUEST; all waiters receive the same FRAME_DATA response.

### 2026-04-21 — `AgentEndpoint` + `ConsumerEndpoint` → `ProcessorEndpoint`

`ipc/_agent.py` and `ipc/_consumer.py` are deleted. Both are replaced by a
single `ProcessorEndpoint` in `ipc/_processor.py`.
**Why:** `ConsumerEndpoint` was unused scaffolding; `AgentEndpoint` was too
narrow a name (the endpoint suits analytics, recording, etc. — not just agents).
`ProcessorEndpoint` auto-maintains `connected_participants: frozenset[str]` so
processors always know who is present without manual event tracking.

### 2026-04-21 — Agent return path through hub

Agents push `RETURN_DATA`/`RETURN_AUDIO` on the hub's PULL socket.
The hub's `_dispatch` routes them to `send_return_data`/`send_return_audio`,
which PUBs them on `return_data.<pid>` / `return_audio.<pid>` topics.
The `ConnectorEndpoint` SUBs these topics and calls registered callbacks
→ `RoomClient` → LiveKit → client.
**Why:** closes the loop so agents can send audio and data back to participants.

### 2026-04-21 — Echo-agent sample added

`agent-samples/echo-agent/` — echoes audio back to the originating participant
and sends a JSON stats ping (`topic="agent.stats"`) every 5 s to each
connected participant. Demonstrates `ProcessorEndpoint` usage end-to-end.

### 2026-04-20 — Track task management keyed by track SID

`RoomClient._track_tasks` changed from `list[Task]` to `dict[str, Task]`
keyed by track SID. A `track_unsubscribed` handler cancels the exact task.
**Why:** without this, stop/start camera caused a new streaming task to start
while the old one kept running, doubling (then tripling) fps counts.

### 2026-04-20 — Audio format: float32 on the wire, int16 in LiveKit

LiveKit delivers audio as int16 PCM. The hub's IPC layer (`AudioChunk`) uses
float32 LE interleaved. Conversion happens in `_room_client.py`:
- Inbound: `int16 / 32768.0 → float32`
- Outbound (return audio): `clip(float32, -1, 1) * 32767 → int16`

### 2026-04-20 — `xr_media_hub.yaml` config file

Flat YAML at repo root. Fields map 1:1 to `LiveKitConnectorConfig` dataclass.
Relative paths (e.g. `web_client_dir`) resolve relative to the YAML file's
own directory, not CWD. `HubLauncher` searches upward from CWD to find it
automatically.
