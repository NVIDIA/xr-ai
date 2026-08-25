<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Troubleshooting

Common issues and their fixes. If you hit something not listed here, open an
issue on the [repository](https://github.com/NVIDIA/xr-ai).

## Setup-time issues

### DGX Spark — `uv sync` fails to build a wheel

**Symptom:** `uv sync` fails on a DGX Spark system while building NeMo or
vLLM wheels with errors mentioning missing `Python.h` or development
headers.

**Cause:** the system is missing CPython development headers.

**Fix:** install before running `uv sync`:

```bash
sudo apt install python3-dev
```

This applies to the `agent-samples/model-servers/yaml/spark/` profile.

### DGX Spark — LOVR auto-download is not supported

**Symptom:** `uv run --project agent-samples/xr-render-demo xr_render_demo`
exits at startup with:

```
xr-render-demo: LOVR auto-download is not supported on linux/aarch64.
```

**Cause:** upstream LOVR releases do not ship a prebuilt aarch64 Linux binary,
so the orchestrator cannot fetch one. Build LOVR from source on the Spark and
point `LOVR_BIN` at it.

**Fix:**

```bash
sudo apt install -y cmake build-essential \
                    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
                    libcurl4-openssl-dev libx11-xcb-dev

git clone --recursive https://github.com/bjornbytes/lovr.git ~/lovr
cd ~/lovr
mkdir build && cd build
cmake ..
make -j$(nproc)

export LOVR_BIN=~/lovr/build/bin/lovr
```

`export LOVR_BIN=…` only lasts for the current shell. To make it permanent,
append the line to `~/.bashrc`, or set `lovr_bin: ~/lovr/build/bin/lovr` in
`agent-samples/xr-render-demo/scene/scene_service.yaml` instead.

If `git clone` was run without `--recursive`, run
`git submodule update --init --recursive` inside `~/lovr` before `cmake ..`.

### Blackwell GPUs (B200, RTX PRO 6000) — VLM fails to start

**Symptom:** the VLM server logs FlashInfer or NVFP4 kernel errors and never
becomes healthy on a Blackwell-class system.

**Cause:** The Docker backend cannot expose the Blackwell GPU, or the selected
vLLM image does not contain compatible kernels. Any first-use kernel
compilation occurs inside the container; host NVCC is not required.

**Fix:** install the NVIDIA Container Toolkit, restart Docker as its installation
guide requires, and retain the vLLM image pinned by the reviewed hardware
profile:

[NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

This applies to the
`agent-samples/model-servers/yaml/96G_blackwell/` profile.

### GPU service aborts with `cuDNN version incompatibility`

**Symptom:** a GPU service (commonly the NeMo STT server) crashes at torch
import with:

```
RuntimeError: cuDNN version incompatibility: PyTorch was compiled against
(9, 20, 0) but found runtime version (9, 13, 1). ... Looks like your
LD_LIBRARY_PATH contains incompatible version of cudnn.
```

**Cause:** the host exports an `LD_LIBRARY_PATH` that points at a system cuDNN
(common on cloud GPU images). It shadows the cuDNN bundled in the service's
venv — the exact version that venv's PyTorch was compiled against — so torch
loads the wrong runtime and aborts.

**Fix:** the launcher handles this automatically — `model_servers` or
`xr_render_demo` strip any `libcudnn`-bearing directory from each child's
`LD_LIBRARY_PATH` before spawning (logged once as a WARNING), so the
venv-bundled cuDNN is used. If you hit this running a service **directly**
(outside the launcher), clear the conflicting path yourself first:

```bash
# Inspect what's on the path
echo "$LD_LIBRARY_PATH"
# Run the service without the host cuDNN shadowing the venv copy
env -u LD_LIBRARY_PATH uv run <command>
```

### `vllm_backend: docker` — image pull fails with "unauthorized" or "denied"

**Symptom:** the wrapper logs `[<service>] Launching vLLM (docker)` and then
`docker run` fails with one of:

- `Error response from daemon: pull access denied for nvcr.io/nvidia/vllm`
- `unauthorized: authentication required`
- `denied: requested access to the resource is denied`

**Cause:** docker is not authenticated to `nvcr.io`, so it cannot pull the
NGC vLLM container.

**Fix:** log in with your NGC API key once. Get a key from
https://ngc.nvidia.com/setup/api-key and run:

```bash
docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY
```

The credential is cached in `~/.docker/config.json` and reused on subsequent
runs. Alternatively, save the key into the xr-ai credential cache so the
wrapper can auto-login:

```bash
python3 -c "
import json, os, pathlib
p = pathlib.Path.home() / '.config/xr-ai/credentials.json'
d = json.loads(p.read_text()) if p.exists() else {}
d['NGC_API_KEY'] = os.environ['NGC_API_KEY']
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(d, indent=2))
"
```

The orchestrator's `load_credentials()` injects `NGC_API_KEY` into the
environment before each wrapper runs; the docker backend uses it to run
`docker login nvcr.io --password-stdin` when no existing auth is found.

### `vllm_backend: docker` — wrapper exits with `vLLM exited before /health became reachable`

**Symptom:** the launcher reports

```
[<service>] vLLM exited before /health became reachable
[<service>] exited (rc=1) before signaling ready
```

and the per-run log file under `/tmp/log_<sample>_<timestamp>/<service>.log`
contains only wrapper messages — nothing from inside the container.

**Health probe** — confirm vLLM never reached the `/health` endpoint:

```bash
curl -fsS http://127.0.0.1:8108/health   # nemotron_omni (default LLM)
curl -fsS http://127.0.0.1:8100/health   # vlm_server (default Cosmos VLM)
curl -fsS http://127.0.0.1:8107/health   # superseded nemotron3_nano
```

**Container post-mortem** — the wrapper streams `docker logs -f` into the
per-run log file, so on the next run the actual vLLM error lands next to the
wrapper messages. To inspect manually:

```bash
docker ps -a --filter name=xr-ai-vllm-
docker logs --tail=200 <container-name>
```

**Cause:** vLLM crashed during startup — common reasons: model weights
missing or inaccessible in the bind-mounted `model_cache`, GPU not visible to
the container (`nvidia-container-cli` or `--gpus`), HF token missing for a
gated model, or a reasoning-parser plugin file that is not present inside
the container.

**Fix:** read the container logs (the next run captures them automatically),
address the root cause shown there, and retry. If the container was
auto-removed by `--rm` before you could check, the next failed run will
have the streamed output in the per-run log file — just re-run.

### `vllm_backend: docker` — `docker run` fails with `could not select device driver`

**Symptom:** `docker run` exits with a message mentioning `nvidia-container-cli`
or "could not select device driver "" with capabilities: [[gpu]]".

**Cause:** the NVIDIA Container Toolkit is not installed (or the daemon was
not restarted after install), so docker cannot honor `--gpus`.

**Fix:** install the toolkit and restart docker:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Switch back to `vllm_backend: pip` in the service YAML if you only need the
local install.

(hub-fails-immediately-with-runtimeerror-missing-libnvcuvid-so-libnvidia-encode-so)=

### Hub fails immediately because NVIDIA codec libraries are missing

**Cause:** The hub raises
`RuntimeError: missing libnvcuvid.so or libnvidia-encode.so` because NVDEC
(`libnvcuvid.so`) and NVENC (`libnvidia-encode.so`) are required. The
DeviceIOHub refuses to start without them so it never silently falls back to
OpenH264, which is royalty-bearing.

**Fix:**
- **Bare metal:** install or repair the NVIDIA driver. The libraries ship with the
  driver, not with CUDA.
- **Docker:** pass `--gpus all` (or `--device /dev/nvidia*` plus the codec
  device nodes) when starting the container.

## Runtime and connection issues

### Voice session drops or agent goes silent after a few minutes idle

**Cause:** an idle-timeout that auto-cancels the voice pipeline after a stretch
with no user or bot speech.

**Status:** disabled by default. `VoiceAgent` leaves its private Pipecat idle
timeout disabled, so a quiet session stays connected indefinitely.

**If you want it:** set `idle_timeout_secs: <seconds>` (e.g. `300` for 5 min)
in the sample's worker YAML (`simple_vlm_example_worker.yaml` or
`xr_render_demo_worker.yaml`); `0` or unset keeps it disabled. The knob is
owned by `xr_ai_voice.VoiceAgent`.

### Browser client connects but no audio or no video

**Most common cause:** firewall blocking WebRTC media on UDP 7882 (LiveKit).

**Fix:** open ports per {doc}`/getting_started/networking`. The web client
will appear to connect (signaling on 7880 succeeds) but media frames are
silently dropped without 7882.

### HTTPS web client → `ws://` mixed-content warning

**Symptom:** the LiveKit JS SDK logs a mixed-content error connecting to
`ws://<host>:7880/…` from an HTTPS page.

**Cause:** a stale client build (or a hand-rolled configuration) is pointing at
LiveKit's native 7880 instead of the same-origin `wss://<host>:8080/rtc`
proxy the DeviceIOHub exposes.

**Fix:** rebuild against the current `client-samples/web` or `web-xr` —
both auto-detect the page's protocol and use the wss proxy. If you're
holding a `LiveKitConfig` directly, set `port` to the hub's
`web_server_port` (8080) and let the SDK build `wss://host:8080`. Android,
iOS, and visionOS use wss only; there is no `secure` toggle.

### Android — connection fails with TLS or certificate errors

**Symptom:** the Android sample fails to connect; the error shows an
`SSLHandshakeException` or similar TLS error.

**Cause:** the hub uses a self-signed certificate by default. The Android sample
validates against the system and user CA store, the same as iOS.

**Fix:** install the hub's certificate via the in-app button before connecting:

1. In the **Connection** section, tap **Install hub certificate** (enabled once
   **Host** is non-empty).
2. The app fetches the certificate from `https://<host>:<port>/cert` and opens
   the system certificate-install dialog.
3. Confirm the install. After install, tap **Connect** — validation succeeds
   automatically.

Repeat for each hub host. Replace the auto-generated certificate with one from a
public CA via `cert_file` or `key_file` in `device_io_hub.yaml` for
production.

### iOS and visionOS — connection fails with certificate-trust errors

**Symptom:** the iOS or visionOS sample fails to connect; the LiveKit
WebSocket reports a TLS error (e.g. `NSURLErrorServerCertificateUntrusted`,
`-1202`, "The certificate for this server is invalid").

**Cause:** the LiveKit Swift SDK cannot bypass certificate validation, so the
hub's self-signed certificate must be trusted at the OS level.

**Fix:** install the hub's certificate as a trusted profile on the device:

1. In Safari on the device, open `https://<host>:8080/cert` and tap
   **Show Details → visit this website** past the certificate warning.
2. Approve the **Download Configuration Profile** prompt.
3. Install via **Settings → General → VPN & Device Management**.
4. Toggle **Settings → General → About → Certificate Trust Settings →
   Enable Full Trust** for the new certificate.

If step 4 shows no toggle, the cached certificate on the hub is from an older
xr-ai build that wrote `BasicConstraints CA:FALSE` and iOS will not
expose the trust toggle for it. Remove the installed profile via
**VPN & Device Management** and restart the hub — it auto-detects the
stale certificate and regenerates it as a self-signed CA (logged as `TLS: cached
cert is not a CA cert — regenerating…`).

If the toggle was enabled but the wss handshake still fails with
`errSSLBadCert` or NSURLErrorDomain `-1202` and a message like *"pretending
to be 192.168.1.42"* (that is, the IP you typed into the app), the
certificate's SubjectAlternativeName doesn't cover that IP. The hub detects
local IPv4 addresses via a UDP-connect probe and auto-regenerates the
certificate whenever the SAN is missing one (logged as `TLS: cached cert
SAN is missing …; regenerating…`); just restart the hub and re-install the
profile on the device. If the dialed address is not on any of the hub's
interfaces, add it to `web_server_extra_sans` in `device_io_hub.yaml` and
restart. Refer to {doc}`Networking </getting_started/networking>` for details.
To force regeneration, delete `~/.local/share/xr-ai/web-server.crt` and
`web-server.key` before restarting.

If the certificate is trusted (no `-1202`) but the room connection still fails
with HTTP 401 or "no permissions to access the room", the hub's `/rtc` WSS
proxy is dropping the `Authorization: Bearer <token>` header the Swift
SDK sends. Update to the latest DeviceIOHub and restart; the proxy
forwards the `Authorization` header on `/rtc/validate` and the WebSocket.

Repeat the install step per hub host, or replace the auto-generated certificate
with a public-CA certificate via `cert_file` or `key_file` in `device_io_hub.yaml`
for production.

### iOS and visionOS — microphone or camera is interrupted

An occasional LiveKit microphone timeout means its recording engine did not
produce the first buffer before publication. The current client enables prepared
recording mode before publishing to make that first buffer available. Stopping
the microphone then disables prepared input while leaving output active, so the
orange microphone indicator clears without silencing agent audio.

Phone calls, Siri, route changes, media-service resets, another capture app, or
closing an XR space can interrupt audio or camera while the control still shows
the user's requested state. The client re-arms capture when the OS allows it.
If it does not recover, filter Console.app for the `MediaSession` category to
inspect the recorded interruption, route, and capture-session events. CoreAudio
`-50` and `FigAudioSession -19224` messages alone are not evidence of failure;
they can also appear on successful starts.

### Chrome — Immersive Web extension cannot be enabled

**Symptom:** the Immersive Web extension for Chrome cannot be enabled.

**Status:** known issue, no workaround currently.

**Workaround:** use a native client (Quest 3, Vision Pro) on the same LAN, or
the IWER emulator built into the web client itself for desktop dev.

### vLLM cold start takes 3–8 minutes

**Symptom:** a vLLM server's weight load is fast,
but the server then sits silent for several minutes before becoming healthy.

**Cause:** CUDA graph capture + FlashInfer FP4 MoE autotune happen on first
run after weight load. They are silent.

**Fix:** the default Omni profiles set `enforce_eager: false` to enable CUDA
graph capture and maximize steady-state throughput, so this startup delay is
expected. For development, set `enforce_eager: true` in the active model YAML
to skip CUDA graph capture. Eager mode starts faster but can reduce per-token
throughput; keep the default when steady-state performance matters more than
cold-start time.

### vLLM exits before readiness with insufficient GPU memory

**Symptom:** a vLLM container exits during startup with `CUDA out of memory`,
`No available memory for the cache blocks`, or a negative available KV-cache
value buried in its container log.

**Fix:** the wrapper classifies these signatures as `INSUFFICIENT GPU MEMORY`
and prints the configured utilization when available. Free memory held by other
processes, reduce model context or concurrency, or use a device with more
GPU-visible memory. The complete original error remains in the reported log file.

### `xr_render_demo` exits but VRAM is still pinned

**By design.** The vLLM-backed servers (`nemotron_omni_llm_server`,
`vlm_server`, and `nemotron3_nano_llm_server`) survive stack
restarts so model weights stay loaded across worker crashes and debug
restarts. Refer to {doc}`/components/ai-services` → *vLLM model
persistence*.

**Fix:** to fully release VRAM:

```bash
cd xr-ai
uv run --project agent-samples/model-servers model_servers --stop
```

For pip-mode servers this sends `SIGTERM` to each persisted process, waits up
to 20 s, then `SIGKILL`s. For docker-mode servers it runs
`docker stop <container_name>` (escalating to `docker kill` after 20 s). Safe
to run while the stack is down.

### First run downloads models silently

**Symptom:** `uv run --project agent-samples/model-servers model_servers`
appears to hang at startup the first time.

**Cause:** model weights are downloading from HuggingFace into `models/` at
the repository root (gitignored; the Cosmos3 checkpoint alone is tens of GB).

**Fix:** wait. Subsequent runs use the cached weights and start in
~30–60 s. If the download makes no progress, note that unauthenticated
downloads (model-server runs started with `--allow-anonymous`) are rate-limited and can
stall indefinitely; set `HF_TOKEN` and restart. Refer to
{doc}`Credentials </getting_started/credentials>`.
