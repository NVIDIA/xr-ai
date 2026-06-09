<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Troubleshooting

Known frictions and their fixes. If you hit something not listed, please add
it here in the same change that you understand it.

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

This applies to the `xr-render-demo/yaml/spark/` profile.

### DGX Spark — LOVR auto-download is not supported

**Symptom:** `uv run xr_render_demo` exits at startup with:

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
`agent-mcp-servers/render-mcp/render_mcp.yaml` instead.

If `git clone` was run without `--recursive`, run
`git submodule update --init --recursive` inside `~/lovr` before `cmake ..`.

### Blackwell GPUs (B200, RTX PRO 6000) — VLM fails to start

**Symptom:** the VLM server logs FlashInfer or NVFP4 kernel errors and never
becomes healthy on a Blackwell-class system.

**Cause:** Blackwell FP4 MoE kernels need both the **NVIDIA Container Toolkit**
and a working **CUDA NVCC** toolchain present on the host (the kernels are
JIT-compiled at first use).

**Fix:** install both before launching:

```bash
# NVIDIA Container Toolkit (covers both Docker and bare-metal CUDA driver bits)
# Follow the latest instructions at:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# CUDA NVCC — install the matching CUDA toolkit for your driver
sudo apt install nvidia-cuda-toolkit
```

This applies to the `xr-render-demo/yaml/96G_blackwell/` profile.

### `vllm_backend: docker` — image pull fails with "unauthorized" / "denied"

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
curl -fsS http://127.0.0.1:8107/health   # nemotron3_nano (agent-llm)
curl -fsS http://127.0.0.1:8100/health   # vlm_server
curl -fsS http://127.0.0.1:8106/health   # llama_nemotron (llm)
```

**Container post-mortem** — the wrapper now streams `docker logs -f` into the
per-run log file, so on the next run the actual vLLM error lands next to the
wrapper messages. To inspect manually:

```bash
docker ps -a --filter name=xr-ai-vllm-
docker logs --tail=200 <container-name>
```

**Cause:** vLLM crashed during startup — common reasons: model weights
missing/inaccessible in the bind-mounted `model_cache`, GPU not visible to
the container (`nvidia-container-cli` / `--gpus`), HF token missing for a
gated model, or a reasoning-parser plugin file that is not present inside
the container.

**Fix:** read the container logs (the next run captures them automatically),
address the root cause shown there, and retry. If the container was
auto-removed by `--rm` before you could check, the next failed run will
have the streamed output in the loguru log file — just re-run.

### `vllm_backend: docker` — `docker run` fails with `could not select device driver`

**Symptom:** `docker run` exits with a message mentioning `nvidia-container-cli`
or "could not select device driver "" with capabilities: [[gpu]]".

**Cause:** the NVIDIA Container Toolkit is not installed (or the daemon was
not restarted after install), so docker cannot honor `--gpus`.

**Fix:** install the toolkit and restart docker:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Switch back to `vllm_backend: pip` in the service YAML if you only need the
local install.

### Hub fails immediately with `RuntimeError: missing libnvcuvid.so / libnvidia-encode.so`

**Cause:** NVDEC (`libnvcuvid.so`) and NVENC (`libnvidia-encode.so`) are
required — the hub refuses to start without them so it never silently falls
back to OpenH264 (which is royalty-bearing). See
[`docs/changelog.md`](changelog.md) entry **2026-04-21 — NVDEC/NVENC required**.

**Fix:**
- **Bare metal:** install/repair the NVIDIA driver. The libs ship with the
  driver, not with CUDA.
- **Docker:** pass `--gpus all` (or `--device /dev/nvidia*` plus the codec
  device nodes) when starting the container.

## Runtime / connection issues

### Voice session drops / agent goes silent after a few minutes idle

**Cause:** an idle-timeout that auto-cancels the voice pipeline after a stretch
with no user/bot speech.

**Status:** disabled by default. `make_voice_pipeline` passes
`cancel_on_idle_timeout=False` (overriding pipecat's on-by-default
`IDLE_TIMEOUT_SECS`), so a quiet session stays connected indefinitely.

**If you want it:** set `idle_timeout_secs: <seconds>` (e.g. `300` for 5 min)
in the sample's worker YAML (`simple_vlm_example_worker.yaml` /
`xr_render_demo_worker.yaml`); `0` or unset keeps it disabled. The knob is
threaded to `xr_ai_pipecat.make_voice_pipeline`, where it's documented.

### Browser client connects but no audio / no video

**Most common cause:** firewall blocking WebRTC media on UDP 7882 (LiveKit).

**Fix:** open ports per [`docs/networking.md`](networking.md). The web client
will appear to connect (signaling on 7880 succeeds) but media frames are
silently dropped without 7882.

### HTTPS web client → `ws://` mixed-content warning

**Symptom:** the LiveKit JS SDK logs a mixed-content error connecting to
`ws://<host>:7880/…` from an HTTPS page.

**Cause:** a stale client build (or a hand-rolled config) is pointing at
LiveKit's native 7880 instead of the same-origin `wss://<host>:8080/rtc`
proxy the hub exposes.

**Fix:** rebuild against the current `client-samples/web` or `web-xr` —
both auto-detect the page's protocol and use the wss proxy. If you're
holding a `LiveKitConfig` directly, set `port` to the hub's
`web_server_port` (8080) and let the SDK build `wss://host:8080`. The
`secure` toggle is gone from the Android, iOS, and visionOS samples —
wss is the only mode.

### Android — connection fails with TLS / certificate errors

**Symptom:** the Android sample fails to connect; the error shows an
`SSLHandshakeException` or similar TLS error.

**Cause:** the hub uses a self-signed cert by default. The Android sample
no longer bypasses cert validation globally — it now validates against the
system + user CA store, the same as iOS.

**Fix:** install the hub's cert via the in-app button before connecting:

1. In the Connection section, tap **Install hub certificate** (enabled once
   Host is non-empty).
2. The app fetches the cert from `https://<host>:<port>/cert` and opens the
   system cert-install dialog.
3. Confirm the install. After install, tap **Connect** — validation succeeds
   automatically.

Repeat for each hub host. Replace the auto-generated cert with one from a
public CA via `cert_file` / `key_file` in `xr_media_hub.yaml` for
production.

### iOS / visionOS — connection fails with cert-trust errors

**Symptom:** the iOS or visionOS sample fails to connect; the LiveKit
WebSocket reports a TLS error (e.g. `NSURLErrorServerCertificateUntrusted`,
`-1202`, "The certificate for this server is invalid").

**Cause:** the LiveKit Swift SDK's `URLSession` does not expose a
server-trust auth-challenge hook (`WebSocket.swift` Delegate only
implements `didOpenWithProtocol` and `didCompleteWithError`), and ATS
does not bypass certificate-chain validation regardless of
`NSAllowsArbitraryLoads`. Until the hub's self-signed cert is trusted at
the OS level, the wss handshake fails. The `TrustingSessionDelegate`
inside `LiveKitBackend.swift` only covers the `/token` HTTP fetch.

**Fix:** install the hub's cert as a trusted profile on the device:

1. In Safari on the device, open `https://<host>:8080/cert` and tap
   **Show Details → visit this website** past the cert warning.
2. Approve the **Download Configuration Profile** prompt.
3. Install via **Settings → General → VPN & Device Management**.
4. Toggle **Settings → General → About → Certificate Trust Settings →
   Enable Full Trust** for the new cert.

If step 4 shows no toggle, the cached cert on the hub is from an older
xr-ai build that wrote `BasicConstraints CA:FALSE` and iOS will not
expose the trust toggle for it. Remove the installed profile via
**VPN & Device Management** and restart the hub — it auto-detects the
stale cert and regenerates as a self-signed CA (logged as `TLS: cached
cert is not a CA cert — regenerating…`).

If the toggle was enabled but the wss handshake still fails with
`errSSLBadCert` / NSURLErrorDomain `-1202` and a message like *"pretending
to be 10.29.90.196"* (i.e. the IP you typed into the app), the cert's
SubjectAlternativeName doesn't cover that IP. The hub now detects local
IPv4 addresses via a UDP-connect probe and auto-regenerates the cert
whenever the SAN is missing one (logged as `TLS: cached cert SAN is
missing local IP(s) … — regenerating…`); just restart the hub and
re-install the profile on the device. To force regen explicitly, delete
`~/.local/share/xr-ai/web-server.crt` and `web-server.key` before
restarting.

If the cert is trusted (no `-1202`) but the room connection still fails
with HTTP 401 / "no permissions to access the room", the hub's wss /rtc
proxy is dropping the `Authorization: Bearer <token>` header the Swift
SDK sends (the JS SDK puts the JWT in the query string and never hit
this code path). The current proxy forwards `Authorization` plus every
other end-to-end header on both `/rtc/validate` and the WebSocket;
pulling the latest hub and restarting fixes this without any
client-side change.

Repeat the install step per hub host, or replace the auto-generated cert
with a public-CA cert via `cert_file` / `key_file` in `xr_media_hub.yaml`
for production.

### Chrome — Immersive Web extension cannot be enabled

**Symptom:** the Immersive Web extension for Chrome cannot be enabled.

**Status:** known issue, no workaround currently.

**Workaround:** use a native client (Quest 3, Vision Pro) on the same LAN, or
the IWER emulator built into the web client itself for desktop dev.

### vLLM cold start takes 3–8 minutes

**Symptom:** `vlm_server` / `nemotron3_nano_llm_server` weight load is fast,
but the server then sits silent for several minutes before becoming healthy.

**Cause:** CUDA graph capture + FlashInfer FP4 MoE autotune happen on first
run after weight load. They are silent.

**Fix:** the shipped YAMLs default to `enforce_eager: true` which avoids both.
Eager mode is 10–20% slower per token but starts in ~5 s after weight load —
imperceptible at <250 tokens/turn where STT+VAD+TTS dominate latency. Don't
flip `enforce_eager: false` unless you have a measured reason.

### `xr_render_demo` exits but VRAM is still pinned

**By design.** The vLLM-backed servers (`vlm_server`,
`llama_nemotron_llm_server`, `nemotron3_nano_llm_server`) survive stack
restarts so model weights stay loaded across worker crashes and debug
restarts. See [`docs/ai-services.md`](ai-services.md) → *vLLM model
persistence*.

**Fix:** to fully release VRAM:

```bash
cd xr-ai/agent-samples/xr-render-demo
uv run xr_render_demo --stop
```

For pip-mode servers this sends `SIGTERM` to each persisted process, waits up
to 20 s, then `SIGKILL`s. For docker-mode servers it runs
`docker stop <container_name>` (escalating to `docker kill` after 20 s). Safe
to run while the stack is down.

### First run downloads models silently

**Symptom:** `uv run simple_vlm_example` appears to hang at startup the first
time.

**Cause:** model weights are downloading from HuggingFace into `models/` at
the repo root (gitignored, ~16 GB for Cosmos-Reason1-7B alone).

**Fix:** wait. Subsequent runs use the cached weights and start in
~30–60 s. If a download fails, check that `HF_TOKEN` is set if the model
needs it (see [`docs/credentials.md`](credentials.md)).
