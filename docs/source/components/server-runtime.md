<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Server runtime

The `services/device-io-hub/` package hosts the **DeviceIOHub** — the single process
clients connect to and agents fan out from. It owns the internal LiveKit
transport, the shared-memory + ZMQ IPC boundary to agents, the per-participant
return path, and the same-origin `wss://` proxy that fronts LiveKit signaling.

From the repository root, run it as one process:

```
uv run --project services/device-io-hub device_io_hub \
  --config services/device-io-hub/device_io_hub.yaml
```

Configuration comes from a `device_io_hub.yaml` file. When none is found,
non-secret defaults are used, but LiveKit credentials must still come from
`LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.
`services/device-io-hub/device_io_hub.yaml` is the reference copy documenting
every field; each sample ships its own copy under its `yaml/` directory.
Relative paths inside the YAML (such as `web_client_dir`) resolve against the
YAML file's own directory, not the working directory.

For where the hub sits in the wider system, refer to
{doc}`Architecture </overview/architecture>`.

## DeviceIOHub

The hub is one hub, many clients, many agents. A single instance fans the
inbound media stream out to every connected agent and routes any return
traffic back to the originating client only.

On startup, `__main__.py` constructs a `HubEndpoint` (the IPC server),
registers hub-local callbacks (`on_frame`, `on_audio`, `on_data`,
`on_participant`), loads the configuration, and brings up the `LiveKitConnector`.
The hub task and the connector task then run concurrently until `SIGINT` or
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
  `return_data.*`) are transport-infrastructure-only. Connectors consume them
  for delivery and the optional capture process observes them read-only; an
  agent's default subscription excludes them.
- On the LiveKit side, return audio is published as one track per participant
  with subscribe permissions restricted to that participant, and return data
  is addressed with `destination_identities` (refer to
  [the per-participant return path](#per-participant-return-path)).

```{note}
This isolation is a property of the hub's routing, not a limitation of the
transport. LiveKit natively supports client-to-client communication, and an
application is free to use those native features directly for peer-to-peer
media or data. Doing so is **outside the scope of XR AI**: the hub neither
routes nor guarantees that traffic, and it is not portable across transports.
Build on the hub's participant ↔ agent contract for behavior that must port
across backends.
```

## Internal LiveKit transport

LiveKit is an internal transport implementation detail. It is not exposed to
agent APIs: agents speak only the IPC protocol below and never need to know
which transport carries the media.

`LiveKitConnector` (`transport/livekit/`) owns the transport lifecycle:

1. Starts the LiveKit server in a host-networked Docker container (plaintext
   signaling on port 7880, plus WebRTC TCP/UDP media ports 7881/7882). The
   generated configuration does not restrict the signaling listener to
   loopback, so deployment firewalls must control direct access.
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
messages. The layer lives in `services/device-io-hub/device_io_hub/ipc/` and defines
three endpoints:

| Endpoint | Role | Who |
| --- | --- | --- |
| `ConnectorEndpoint` | producer + return-traffic receiver | LiveKit connector process |
| `HubEndpoint` | server: dispatch + fan-out | DeviceIOHub process |
| `ProcessorEndpoint` | subscriber + publisher | agents, analytics, downstream processors |

The hub binds two sockets (defaults shown):

- `PULL` on `ipc:///tmp/xr_hub_in` — connectors `PUSH` inbound media here.
- `PUB` on `ipc:///tmp/xr_hub_pub` — consumers `SUB` here for the fanned-out
  stream.

```
connector_A ──PUSH──┐
connector_B ──PUSH──┤─► PULL   HubEndpoint   PUB ──SUB──► consumers (agents)
connector_N ──PUSH──┘    ↓ dispatch
                      on_frame, on_audio, on_data, on_participant
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
participant              — join and leave events
```

The message types and codec are extensible: new `MsgType` IDs can be
registered at import time via `register_encoder` and `register_decoder`.

```{note}
Import the IPC types and `ProcessorEndpoint` in agent code from
`xr_ai_hub` directly, **not** from `device_io_hub.ipc`. The agent SDK's only
runtime dependencies are `pyzmq` and `msgpack` — importing from the agent SDK
avoids pulling in the full DeviceIOHub dependency tree (LiveKit, FastAPI,
uvicorn, GPU codecs). `device_io_hub.ipc` re-exports the same names for the
server side.
```

## Media-hub session capture

`device_io_capture` is an optional process from the DeviceIOHub package. It
sits downstream of `HubEndpoint`, not inside the LiveKit connector:

```
device transport → connector → HubEndpoint → agents
                                  └────────→ capture
                         agent return media ─┘
```

This boundary is transport-independent. Incoming camera, microphone, and data
have already been decoded, timestamped, and tagged with participant and track
identities. The capture process uses a normal `ProcessorEndpoint` for that
stream and a private read-only subscription to the hub's routed return topics
for agent audio and data. It does not expose return traffic to agent APIs.

Run the hub first, then start capture from the repository root:

```
uv run --project services/device-io-hub device_io_capture \
  --config services/device-io-hub/media_capture.yaml
```

One connection produces one participant-scoped capture bundle: captioned H.264
video, a timestamp-aligned stereo PCM WAV (device left, agent right), exact raw
float32 audio chunks, the complete directional data timeline, and a manifest.
The caption panel is appended below the sensor image so composition does not
replace camera pixels. Only configured outbound text topics appear in the
panel; every data payload remains in `events.jsonl`.

Capture frame requests are coalesced in a bounded queue, and NVENC work runs in
dedicated threads in the capture process. Recorder overload therefore drops
capture frames without delaying the hub's publish path. PyNvVideoCodec receives
contiguous NV12 CPU input and emits H.264 Annex B segments with repeated
parameter sets and no B-frames. No FFmpeg process or additional media library
is involved.

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
paced before IPC by the built-in voice transport, then fed into LiveKit by a
per-participant pipe that a flush can drain to interrupt playback.
`return_audio_max_buffer_s` (3 seconds by default) is also a hard
per-participant duration bound: if a custom or faulty producer runs ahead of
playback, the oldest queued frames are dropped without affecting any other
participant. Set this value to at least `0.12` when using the built-in voice
transport, which maintains a 120 ms reserve. Smaller values remain available
for custom producers whose chunk size and pacing fit within the configured
bound.

## Agent status aggregation

`_agent.status` is the one exception to straight-through return data. The hub
does not forward an agent's status to the client — it records it per
`(agent_id, participant_id)` and publishes the aggregate, taking the least
available state across the agents *responsible for that participant*:
`loading` > `processing` > `idle` > `ready`. Agents that opt into readiness
announce themselves with `AGENT_PRESENCE` when their receive loop starts and
detach when it stops, so an agent that is still loading holds the room at
`loading` rather than being masked by a peer that is already ready.

`AGENT_PRESENCE` carries the agent's scope — the participants it answers for,
or *None* for all of them. A participant with no responsible agent reads
`loading`. Passive processors never register, and an agent scoped to one pid
is excluded from every other pid's aggregate. Repeat aggregates are suppressed, so the agents' periodic
re-announcements do not become per-agent client traffic.

A status payload without an `agent_id` comes from an SDK predating
aggregation and is forwarded verbatim.

The hub also answers `SUBSCRIPTION_PROBE` by echoing the token on
`_probe.{token}`. Subscription commands from one socket are applied in order,
so the echo tells a processor that its pending SUBSCRIBEs are live — that is
what keeps a client from being told `ready` before its traffic can reach the
agent.

## Same-origin wss proxy

The LiveKit server runs plaintext `ws://` on port 7880. Its host-networked
container and generated configuration do not limit the listener to loopback,
so it can be reachable through host interfaces unless a firewall blocks it.
Browser, web-xr, Android, iOS, and visionOS clients connect through the
same-origin `wss://` URL exposed by the hub's web server. The native C++ client
may connect directly to port 7880 only as a local or source-restricted trusted
network debugging path. Refer to {doc}`/getting_started/networking`.

When `web_server_tls` is enabled (the default), the web server
(`_web_server.py`) terminates TLS on `web_server_port` (8080 by default) and
mounts a `/rtc` route that proxies LiveKit signaling bidirectionally to the
internal `ws://127.0.0.1:7880` (`_lk_proxy.py`). A stable development root CA
and a signed `CA:FALSE` server leaf are auto-generated on first run (SAN
coverage and rotation: [Networking](../getting_started/networking.md)); supply
`cert_file` and `key_file` to use your own.
The proxy forwards end-to-end headers so SDK authentication (such as the LiveKit
Swift SDK's `Authorization: Bearer`) reaches the server, and handles both the
versioned (`/rtc/v1`) and legacy (`/rtc`) signaling paths.

The web server's `/token` endpoint returns a signed LiveKit JWT together with
the client connection URL. With TLS on, that URL is the same-origin
`wss://<host>:<web_server_port>` — so the client SDK never needs a
per-deployment toggle. A `/cert` endpoint serves only the auto-generated public
root CA as an installable client profile; neither private key is exposed.

Set `web_server_tls: false` for the two cases where the hub does not
terminate TLS itself: a TLS-terminating reverse proxy (nginx, Caddy,
Cloudflare Tunnel) sits in front and speaks plain `http://` + `ws://` to the
hub, or localhost-only development where browsers already grant camera and
microphone access on `http://localhost`. Bind the hub to loopback with
`web_server_host: 127.0.0.1`, or source-restrict it with a firewall, when a
reverse proxy is the public entry point. In this mode `/token` returns the
direct `ws://<request-host>:<lk_port_ws>` URL, not the hub's `/rtc` proxy.
Custom clients behind the TLS terminator must use the external same-origin
`wss://` proxy URL and must not expose port 7880 publicly.

For runtime symptoms and fixes, refer to
{doc}`Troubleshooting </guides/troubleshooting>`.
