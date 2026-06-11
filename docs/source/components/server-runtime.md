<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Server runtime

The `server-runtime/` package hosts the **XR-Media-Hub** — the single process
clients connect to and agents fan out from. It owns the internal LiveKit
transport, the shared-memory + ZMQ IPC boundary to agents, the per-participant
return path, and the same-origin `wss://` proxy that fronts LiveKit signaling.

It runs as one process:

```
uv run xr_media_hub                       # auto-discovers ./xr_media_hub.yaml
uv run xr_media_hub --config path.yaml    # explicit config
python -m xr_media_hub                    # equivalent module form
```

Configuration comes from a `xr_media_hub.yaml` file (defaults are used when
none is found). `server-runtime/xr_media_hub.yaml` is the reference copy
documenting every field; each sample ships its own copy under its `yaml/`
directory. Relative paths inside the YAML (such as `web_client_dir`) resolve
against the YAML file's own directory, not the working directory.

For where the hub sits in the wider system, refer to
{doc}`Architecture </overview/architecture>`.

## XR-Media-Hub

The hub is one hub, many clients, many agents. A single instance fans the
inbound media stream out to every connected agent and routes any return
traffic back to the originating client only.

On startup, `__main__.py` constructs a `HubEndpoint` (the IPC server),
registers hub-local callbacks (`on_frame`, `on_audio`, `on_data`,
`on_participant`), loads the configuration, and brings up the `LiveKitConnector`.
The hub task and the connector task then run concurrently until `SIGINT` /
`SIGTERM`. A periodic stats loop logs per-participant video, audio, and data
rates.

### Isolation contract

The hub is **not** a routing switch between participants. There is no supported
path for participant A's media or data to reach participant B. The only
supported flow is:

```
participant → hub → consumer (agent) → hub → same participant
```

This is enforced at several layers:

- `send_return_audio`, `send_return_data`, and `send_return_audio_flush`
  validate that the target participant is currently connected; messages for
  unknown participants are dropped with a warning.
- Return-traffic topics (`return_audio.*`, `return_audio_flush.*`,
  `return_data.*`) are connector-only — an agent's default subscription
  excludes them.
- On the LiveKit side, return audio is published as one track per participant
  with subscribe permissions restricted to that participant, and return data
  is addressed with `destination_identities` (refer to
  [the per-participant return path](#per-participant-return-path)).

```{note}
This isolation is a property of the hub's routing, not a limitation of the
transport. LiveKit natively supports client-to-client communication, and an
application is free to use those native features directly for peer-to-peer
media or data. Doing so is **outside the scope of XR AI**: the hub neither
routes nor guarantees that traffic, and because the transport is an
implementation detail, a future streaming backend may not offer the same
client-to-client capability. Build on the hub's participant ↔ agent contract
for behavior that ports across backends.
```

## Internal LiveKit transport

LiveKit is an internal transport implementation detail. It is not exposed to
the agent or MCP layer — agents only ever speak the IPC protocol below, and
never need to know which transport carries the media.

`LiveKitConnector` (`transport/livekit/`) owns the transport lifecycle:

1. Starts the LiveKit server in a Docker container, listening on the loopback
   interface only (signaling `ws://127.0.0.1:7880`, plus WebRTC TCP/UDP media
   ports 7881/7882).
2. Optionally starts the browser-facing web server and/or token server.
3. Registers itself as a `ConnectorEndpoint` with the IPC layer.
4. Connects a Python `RoomClient` to the LiveKit room. The room client is
   subscribe-only — it never publishes media of its own except per-participant
   return-audio tracks.

The connector translates LiveKit room events into IPC messages: it pushes
decoded frames, audio chunks, and data into the hub, and emits participant
join and leave events.

```{note}
The LiveKit connector requires NVENC and NVDEC hardware video codecs, which it
checks at startup.
```

## IPC boundary to agents

The hub and its producers and consumers communicate over ZMQ using msgpack-encoded
messages. The layer lives in `server-runtime/xr_media_hub/ipc/` and defines
three endpoints:

| Endpoint | Role | Who |
| --- | --- | --- |
| `ConnectorEndpoint` | producer + return-traffic receiver | LiveKit connector process |
| `HubEndpoint` | server: dispatch + fan-out | XR-Media-Hub process |
| `ProcessorEndpoint` | subscriber + publisher | agents, analytics, downstream processors |

The hub binds two sockets (defaults shown):

- `PULL` on `ipc:///tmp/xr_hub_in` — connectors `PUSH` inbound media here.
- `PUB` on `ipc:///tmp/xr_hub_pub` — consumers `SUB` here for the fanned-out
  stream.

```
connector_A ──PUSH──┐
connector_B ──PUSH──┤─► PULL   HubEndpoint   PUB ──SUB──► consumers (agents)
connector_N ──PUSH──┘    ↓ dispatch
                      on_frame / on_audio / on_data / on_participant
```

Each connector owns and creates its own shared-memory ring buffer and
announces it to the hub with a `ConnectorRegistration` message; the hub opens
that buffer on demand. Video frames travel zero-copy through the ring buffer:
the connector writes pixels into a slot and pushes a lightweight
`FRAME_SIGNAL` (metadata) at full frame rate; consumers that want the pixels
issue a `FRAME_REQUEST`, and the hub replies with the held slot's
`FRAME_DATA`. Audio, data, and participant events are carried inline as
msgpack payloads.

Messages are tagged with a `MsgType` and routed by topic. Topics follow the
`"<type>.<participant_id>.<track_or_topic>"` convention, and ZMQ's byte-prefix
subscription lets a consumer subscribe at any granularity:

```
audio                    — all audio, all participants
audio.alice              — all of alice's audio tracks
audio.alice.TR_mic_001   — alice's specific mic track
data.alice.chat          — alice's "chat" data channel only
participant              — join/leave events
```

The message types and codec are extensible: new `MsgType` IDs can be
registered at import time via `register_encoder` and `register_decoder`.

```{note}
Agent code should import the IPC types and `ProcessorEndpoint` from
`xr_ai_agent` directly, **not** from `xr_media_hub.ipc`. The agent SDK's only
runtime dependencies are `pyzmq` and `msgpack` — importing from the agent SDK
avoids pulling in the full server-runtime dependency tree (LiveKit, FastAPI,
uvicorn, GPU codecs). `xr_media_hub.ipc` re-exports the same names for the
server side.
```

## Per-participant return path

Agents send audio, data, and flush signals back toward a specific participant
through the same IPC channel. The hub guards every return path by participant
id:

- `send_return_audio` publishes on topic `return_audio.{pid}.`, dropping the
  chunk if `{pid}` is not connected.
- `send_return_data` publishes on `return_data.{pid}.{topic}`, with the same
  connectivity guard.
- `send_return_audio_flush` publishes on `return_audio_flush.{pid}.` so a
  processor can cleanly interrupt the agent's own audio playback.

The trailing `.` after the participant id terminates the pid segment so that a
subscription for `alice` does not byte-prefix-match a topic addressed to
`alice2`; the connector subscribes with the identical delimiter when a
participant joins, and unsubscribes when they leave.

On the LiveKit side the return path maps to per-participant resources: the room
client lazily publishes one `xr-hub-return-{pid}` audio track per participant
and refreshes subscribe permissions so each participant may subscribe only to
their own return track. Return data is sent with `destination_identities` set
to the target participant, so it is never broadcast to peers. Return audio is
paced into LiveKit at audio rate by a per-participant pipe, which a flush can
drain to interrupt playback.

## Same-origin wss proxy

LiveKit server itself runs plain `ws://` on the loopback interface
(`127.0.0.1:7880`) and nothing reaches that port from off-box. External
clients — browser, web-xr, Android, iOS, visionOS — connect only to a
same-origin `wss://` URL exposed by the hub's web server.

When `web_server_tls` is enabled (the default), the web server
(`_web_server.py`) terminates TLS on `web_server_port` (8080 by default) and
mounts a `/rtc` route that proxies LiveKit signaling bidirectionally to the
internal `ws://127.0.0.1:7880` (`_lk_proxy.py`). A self-signed certificate is
auto-generated on first run; supply `cert_file` and `key_file` to use your own.
The proxy forwards end-to-end headers so SDK authentication (such as the LiveKit
Swift SDK's `Authorization: Bearer`) reaches the server, and handles both the
versioned (`/rtc/v1`) and legacy (`/rtc`) signaling paths.

The web server's `/token` endpoint returns a signed LiveKit JWT together with
the URL the client should use. With TLS on, that URL is the same-origin
`wss://<host>:<web_server_port>` — so the client SDK never needs a
per-deployment toggle. A `/cert` endpoint serves the active certificate as an
installable iOS profile.

Set `web_server_tls: false` for the two cases where the hub should not
terminate TLS itself: a TLS-terminating reverse proxy (nginx, Caddy,
Cloudflare Tunnel) sits in front and speaks plain `http://` + `ws://` to the
hub on the loopback, or localhost-only development where browsers already grant
camera and microphone access on `http://localhost`. In that mode `/token`
returns a plain
`ws://` URL and the proxy carries plain WebSocket.

For runtime symptoms and fixes, refer to
{doc}`Troubleshooting </guides/troubleshooting>`.
