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

Each sample is self-contained: running it starts the hub and every other
required process automatically. No separate server launch step.

## Quickstart

### Echo agent (starts hub + agent in one command)

```bash
cd xr-ai/agent-samples/echo-agent
uv sync
uv run echo_agent
```

On startup you will see hub output (prefixed `[hub]`) followed by the agent
connecting. The hub prints connection details:

```
[hub]   LiveKit URL : ws://0.0.0.0:7880
[hub]   Room        : xr-room
[hub]   Token       : eyJ…   ← paste this into the client
[hub]   Web client  : http://localhost:8080
```

Edit `xr_media_hub.yaml` at the repo root to change ports, credentials, or
point `web_client_dir` at a different web client build.

---

### VLM agent (vision-language queries over live video)

Requires ~16 GB VRAM. Default model: `nvidia/Cosmos-Reason1-7B`
(NVIDIA Open Model License + Apache 2.0 — commercial use permitted).

```bash
cd xr-ai/agent-samples/vlm-agent
uv sync
uv run vlm_agent
```

Override the model via environment variable:

```bash
VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct uv run vlm_agent
```

**Client protocol** — send any data channel message (no specific topic required):
- Raw UTF-8 text: `"What objects are on the table?"`
- Or JSON: `{"query": "What objects are on the table?", "track_id": "optional"}`

The agent replies on topic `vlm.response` with plain UTF-8 text.

The model is loaded at startup (~30–60 s). It is ready before the first query.

---

### Hub only (server-runtime standalone)

```bash
cd xr-ai/server-runtime
uv sync
uv run xr_media_hub --config ../xr_media_hub.yaml
```

Useful for development or when running an agent in a separate terminal.

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

## Design

See [`docs/`](docs/) and the workspace-level `design.md` for architecture details.
