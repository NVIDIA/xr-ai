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

## Design

See [`docs/`](docs/) and the workspace-level `design.md` for architecture details.
