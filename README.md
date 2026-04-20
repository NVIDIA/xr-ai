# xr-ai

Agentic AI for XR — a sample blueprint for multi-modal, real-time conversational AI
within the CloudXR ecosystem.

## Architecture

| Layer | Directory | Description |
|---|---|---|
| Clients | `client-samples/` | Android, iOS/visionOS, Web clients |
| Server runtime | `server-runtime/` | XR-Media-Hub + LiveKit internal transport |
| Agent interfaces | `agent-mcp-servers/` | MCP adapters for XR data & rendering |
| Agent demos | `agent-samples/` | End-to-end Pipecat pipelines |

## Quickstart

### 1. Start the server

```bash
cd xr-ai/server-runtime
uv sync
uv run xr_media_hub --config ../xr_media_hub.yaml
```

On startup the server prints:

```
  LiveKit URL : ws://0.0.0.0:7880
  Room        : xr-room
  Token       : eyJ…   ← paste this into the client
  Web client  : http://localhost:8080
```

Edit `xr_media_hub.yaml` at the repo root to change ports, credentials, or
point `web_client_dir` at a different web client build.

---

### 2. Web client

Open `http://localhost:8080` in a browser. Leave **Token URL** blank to use the
server's built-in `/token` endpoint, or paste the printed token directly.

---

### 3. iOS / visionOS client

See [`client-samples/ios-visionos/README.md`](client-samples/ios-visionos/README.md)
for full Xcode setup. Quick connection settings:

| Field | Value |
|---|---|
| Host | IP of the machine running the server |
| Port | `7880` |
| Token | Paste the token printed on server startup |

The token is valid for 24 hours. To get a fresh one, restart the server or call
`GET http://<host>:8080/token?identity=<name>`.

## Design

See [`docs/`](docs/) and the workspace-level `design.md` for architecture details.
