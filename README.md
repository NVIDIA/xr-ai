# xr-ai

Agentic AI for XR — a sample blueprint for multi-modal, real-time conversational AI
within the CloudXR ecosystem.

## Architecture

| Layer | Directory | Description |
|---|---|---|
| Clients | `client-samples/` | Android, iOS/visionOS, Web clients |
| Server runtime | `server-runtime/` | XR-Media-Hub + LiveKit internal transport |
| Launcher | `launcher/` | stdlib-only process manager used by samples |
| Agent interfaces | `agent-mcp-servers/` | MCP adapters for XR data & rendering |
| Agent demos | `agent-samples/` | End-to-end agent pipelines |
| Tests | `tests/` | Multi-client / multi-agent integration tests |

Each sample is self-contained: running it starts the hub and every other
required process automatically. No separate server launch step.

## Quickstart

### Simple VLM example (vision Q&A over voice + text)

End-to-end voice + vision sample.  Speak into the mic, type into the data
channel, or send the literal text `"ping"` — all routes go through the
same VLM pipeline against the latest video frame.  Replies arrive as
streaming Piper TTS audio plus a `vlm.response` text message.

Requires ~16 GB VRAM (default VLM is `nvidia/Cosmos-Reason1-7B`, NVIDIA
Open Model License + Apache 2.0).

```bash
cd xr-ai/agent-samples/simple-vlm-example
uv sync
uv run simple_vlm_example
```

On startup you will see hub output (prefixed `[hub]`) followed by the
worker connecting.  The hub prints connection details:

```
[hub]   LiveKit URL : ws://0.0.0.0:7880
[hub]   Room        : xr-room
[hub]   Token       : eyJ…   ← paste this into the client
[hub]   Web client  : http://localhost:8080
```

**Client protocol** — anything you send via the data channel is treated
as a query:
- `"ping"` → uses the configured default prompt (`"Describe what you see."`).
- Any other UTF-8 text → used as the query verbatim.
- Audio from the mic → STT (parakeet) → query.

Override the VLM model by editing `vlm_server.yaml` in the sample
directory.  Each sample has its own `xr_media_hub.yaml` controlling the
hub; see [`server-runtime/xr_media_hub.yaml`](server-runtime/xr_media_hub.yaml)
for the full option list.

The VLM and TTS models are loaded at startup (~30–60 s).  They are ready
before the first query.

---

### Hub only (server-runtime standalone)

```bash
cd xr-ai/server-runtime
uv sync
uv run xr_media_hub
```

Useful for development or when running an agent in a separate terminal.
The hub auto-discovers `server-runtime/xr_media_hub.yaml`.

---

### Web client

Open `http://localhost:8080` in a browser. Leave **Token URL** blank to use the
server's built-in `/token` endpoint, or paste the printed token directly.

#### HTTPS (required for camera access from remote devices)

Camera access in browsers is only permitted over `localhost` or HTTPS. When
connecting from another device on the network, enable TLS in `xr_media_hub.yaml`:

```yaml
web_server_tls: true
web_server_port: 8443   # conventional HTTPS alt-port (optional)
```

On first run a self-signed certificate is generated at
`~/.local/share/xr-ai/web-server.crt`. To trust it:

- **Chrome / Edge**: navigate to `https://<host>:8443`, click **Advanced →
  Proceed to … (unsafe)**.
- **Firefox**: click **Advanced → Accept the Risk and Continue**.
- **iOS / Safari**: open the cert URL, follow the prompt to install the profile,
  then enable it under **Settings → General → VPN & Device Management**.

To use your own certificate, set `cert_file` and `key_file` in
`xr_media_hub.yaml`.

---

### Android client

See [`client-samples/android/README.md`](client-samples/android/README.md) for
full setup. Quick steps:

1. Open `client-samples/android/` in Android Studio (Hedgehog or later).
2. Let Gradle sync finish — it downloads the LiveKit Android SDK automatically.
3. Run on a device or emulator (API 24+).
4. Enter the server IP, port (`7880`), and paste the printed token.

Permissions (`RECORD_AUDIO`, `CAMERA`) are requested at runtime on first use.

---

### iOS / visionOS client

See [`client-samples/ios-visionos/README.md`](client-samples/ios-visionos/README.md)
for full Xcode setup. Quick connection settings:

| Field | Value |
|---|---|
| Host | IP of the machine running the server |
| Port | `7880` |
| Token | Paste the token printed on server startup |

The token is valid for 24 hours. To get a fresh one restart the server or call
`GET http://<host>:8080/token?identity=<name>`.

## Firewall

The hub uses the following ports. Open them permanently if a firewall is active.

| Port | Protocol | Purpose |
|------|----------|---------|
| 7880 | TCP | LiveKit WebSocket signaling |
| 7881 | TCP | LiveKit WebRTC TCP fallback |
| 7882 | UDP | LiveKit WebRTC UDP media |
| 8080 | TCP | Web client / token server (HTTP) |
| 8443 | TCP | Web client / token server (HTTPS, if enabled) |

**Ubuntu / Debian (`ufw`)**

```bash
sudo ufw allow 7880/tcp
sudo ufw allow 7881/tcp
sudo ufw allow 7882/udp
sudo ufw allow 8080/tcp
sudo ufw allow 8443/tcp   # HTTPS only
sudo ufw reload
```

**RHEL / Fedora / CentOS (`firewall-cmd`)**

```bash
sudo firewall-cmd --permanent --add-port=7880/tcp
sudo firewall-cmd --permanent --add-port=7881/tcp
sudo firewall-cmd --permanent --add-port=7882/udp
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --permanent --add-port=8443/tcp   # HTTPS only
sudo firewall-cmd --reload
```

---

## Tests

`tests/` contains the multi-client / multi-agent integration suite. The
core IPC tests run without Docker or LiveKit — they spin up real
`HubEndpoint` / `ConnectorEndpoint` / `ProcessorEndpoint` instances over
`ipc://` sockets and verify routing, isolation, and the new
`ReturnAudioFlush` control path.

```bash
cd xr-ai/tests
uv sync
uv run pytest -v
```

See [`tests/README.md`](tests/README.md) for the full breakdown.

CI runs the suite on every push and pull request via
[`.github/workflows/tests.yml`](.github/workflows/tests.yml) on Python
3.11 and 3.12.

## Design

See [`docs/`](docs/) and the workspace-level `design.md` for architecture details.
