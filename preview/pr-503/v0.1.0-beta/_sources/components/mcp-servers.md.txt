<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# MCP servers

The agent's tool surface lives in `agent-mcp-servers/`. Each subdirectory is a
standalone process that exposes its capabilities to the LLM as
[Model Context Protocol](https://modelcontextprotocol.io/) tools. Every server
is built on **FastMCP** and serves a single StreamableHTTP transport mounted at
`/mcp`. A worker (or any `fastmcp.Client`) reaches a server at
`http://<host>:<port>/mcp` (use the URL without a trailing slash).

These are MCP tool servers, reached over the MCP protocol by an MCP client, so
they do not expose a hand-callable REST API of their own: every operation,
including health checks, is an MCP tool rather than a separate REST route. This
is specific to the MCP servers. The rest of the system does use ordinary HTTP:
the AI inference services expose OpenAI-compatible HTTP APIs (refer to
{doc}`ai-services`), and the XR-Media-Hub serves its token and web endpoints
over HTTP.

```{note}
FastMCP is the library these samples use, not a requirement. A server only has
to speak the MCP protocol on a transport the worker's client connects to
(StreamableHTTP at `/mcp` here; stdio and SSE are also valid). Any MCP-compliant
server works — a different language or SDK, or a hand-rolled implementation.

To expose an existing REST service as an agent tool, wrap it in a thin MCP
server rather than calling it directly: `vlm-mcp` does exactly this, forwarding
its `ask_image` tool to the OpenAI-compatible (REST) `vlm-server`. The agent
always speaks MCP; the MCP server is free to call REST, gRPC, or anything else
behind it.
```

The servers split cleanly by concern: `render-mcp` owns the LOVR scene,
`oxr-mcp` reads head pose, `vec-mcp` does pose-free vector math, `video-mcp`
serves camera frames and recordings, `vlm-mcp` answers visual questions, and
`transcript-mcp` stores per-source transcript history. Each runs on its own
fixed port so several can coexist on one host.

| Server | Directory | Module | Port |
|---|---|---|---|
| `transcript-mcp` | `agent-mcp-servers/transcript-mcp/` | `transcript_mcp_server` | 8200 |
| `video-mcp` | `agent-mcp-servers/video-mcp/` | `video_mcp_server` | 8210 |
| `render-mcp` | `agent-mcp-servers/render-mcp/` | `render_mcp` | 8220 |
| `oxr-mcp` | `agent-mcp-servers/oxr-mcp/` | `oxr_mcp_server` | 8230 |
| `vlm-mcp` | `agent-mcp-servers/vlm-mcp/` | `vlm_mcp_server` | 8240 |
| `vec-mcp` | `agent-mcp-servers/vec-mcp/` | `vec_mcp_server` | 8250 |

## How the agent reaches the servers

The `xr-render-demo` sample wires five of these servers into its worker. The
base URLs live in `agent-samples/xr-render-demo/yaml/xr_render_demo_worker.yaml`:

```yaml
render_mcp_url: http://localhost:8220
oxr_mcp_url:    http://localhost:8230
vlm_mcp_url:    http://localhost:8240
video_mcp_url:  http://localhost:8210
vec_mcp_url:    http://localhost:8250
```

At worker startup `list_tools()` is called on every MCP client; the results are
converted to OpenAI tool format and held in memory for the agentic loop. When
the LLM emits a tool call the worker routes it to the owning server by tool name
(`oxr-mcp` for pose helpers, `vec-mcp` for the pure-math primitives, `vlm-mcp`
for `ask_image`, `video-mcp` for the video tools, and `render-mcp` for
everything else). `start_xr` and `get_health` are excluded from the LLM tool
list — the worker calls those directly.

`transcript-mcp` is a standalone store: none of the bundled sample agents wire it
into the agentic loop, so it is reached by any `fastmcp.Client` that connects to
`http://<host>:8200/mcp`.

Each server auto-discovers its YAML configuration by the launcher's `<command>.yaml`
convention, and a sample can override it by dropping a copy next to its
orchestrator. Paths inside a configuration resolve relative to the configuration
file's own directory.

## render-mcp

`render-mcp` owns the XR scene. It launches and supervises the **LOVR**
OpenXR/CloudXR rendering app as a child process and is the only thing that
pushes scene operations onto LOVR's socket. The server binds a ZMQ PUSH socket
(`scene_socket`, default `ipc:///tmp/xr_render_scene`) and the LOVR Lua app
(`xr_app/main.lua`) connects PULL and applies each op. LOVR itself is not
bundled; point `lovr_bin` (or `$LOVR_BIN`) at an existing build.

### render-mcp tools

- `start_xr()` — spawn LOVR if it isn't already running. Idempotent and
  non-blocking: the CloudXR-readiness wait and launch run in a background task,
  so it returns `starting`, `already_started`, or `error` immediately. Poll
  `get_health` until `lovr_started` flips before sending ops you can't drop.
- `add_primitive(prim_type, x, y, z, r, g, b, size)` — add a `sphere` or `box`
  (others fall back to sphere) at a world-space position (OpenXR Y-up, metres),
  with RGB color in `[0, 1]` and `size` in metres. Returns the server-assigned
  `id`.
- `update_primitive(obj_id, prim_type?, x?, y?, z?, r?, g?, b?, size?)` —
  partial update of an existing primitive; omitted fields keep their values.
  Passing `prim_type` converts the shape in place (preserving position, color,
  size).
- `remove_primitive(obj_id)` — delete a primitive by id.
- `get_scene_state()` — return `{objects: [...]}` with each object's `id`,
  `type`, `position`, `color`, and `size`.
- `get_health()` — server status; use `lovr_started` as the readiness signal.

### render-mcp configuration

`render_mcp.yaml`:

```yaml
xr_app_dir: ./xr_app                          # LOVR project directory
# lovr_bin: /home/you/hub/lovr/build/bin/lovr  # else falls back to $LOVR_BIN
host: 0.0.0.0
port: 8220
scene_socket: ipc:///tmp/xr_render_scene       # render-mcp BINDS PUSH; LOVR CONNECTS PULL
cloudxr_env_file: ~/.cloudxr/run/cloudxr.env   # sourced into the LOVR child env
```

The `cloudxr_env_file` is sourced into the LOVR child's environment so LOVR
inherits `XR_RUNTIME_JSON` and the CloudXR pin. A missing file is tolerated
(LOVR then uses the system OpenXR).

## oxr-mcp

`oxr-mcp` is the OpenXR tracking adapter. It opens a **second** OpenXR session
against CloudXR in headless mode (`XR_MND_HEADLESS`) — separate from LOVR's
rendering session — so the rendering client keeps full ownership of frame
submission while `oxr-mcp` reads pose. The session opens lazily on the first
tool call, and pose is fetched fresh per request via `xrLocateSpace` (no
background polling).

### oxr-mcp tools

The pose-aware helpers take named directions and always-positive distances so
the LLM never has to apply signs to user-frame axes:

- `get_head_pose()` — LLM-friendly pose with derived spatial vectors (no raw
  quaternions): `is_valid`, `position`, `forward`, `right`, `up`, `yaw_deg`,
  `pitch_deg`, `ts`.
- `position_ahead(distance)` — world position `distance` metres along the user's
  gaze.
- `position_relative(forward, right, up, origin_x?, origin_y?, origin_z?)` —
  convert user-frame offsets to a world-space position (origin defaults to the
  head).
- `place_user_relative(direction, distance)` — world position a named direction
  (`front`, `back`, `left`, `right`, `above`, or `below`) from the user.
- `place_object_relative(origin_x, origin_y, origin_z, direction, distance)` —
  same, anchored on an existing object (`direction` also accepts `next_to`;
  `front` means *toward the user*).
- `place_inside_by_id(movee_id, container_x, container_y, container_z)` —
  containment for "put X in Y"; returns `{obj_id, x, y, z}` ready to feed into
  `update_primitive`.
- `displace_object(current_x, current_y, current_z, right, up, forward)` —
  user-frame displacement of one object (multi-axis in one call).
- `displace_objects(object_ids, current_xs, current_ys, current_zs, right, up, forward)` —
  the same delta applied to N objects in one call.
- `get_health()` — `{status, session_open, open_attempts, last_open_error}`.

### oxr-mcp configuration

`oxr_mcp_server.yaml`:

```yaml
host: 0.0.0.0
port: 8230
cloudxr_env_file: ~/.cloudxr/run/cloudxr.env   # sourced so XR_RUNTIME_JSON is set before the OpenXR session opens
```

## vec-mcp

`vec-mcp` provides pure-math spatial primitives — the vector arithmetic the LLM
is unreliable at. It is pose-independent (no OpenXR session, no hub IPC) and
simply transforms numbers. All results are rounded to three decimals and
returned as dicts so they compose uniformly.

### vec-mcp tools

- `between_anchors(a_x, a_y, a_z, b_x, b_y, b_z)` — component-wise midpoint of
  two world positions ("between A and B", "halfway between"). Returns
  `{x, y, z}`.
- `world_offset(origin_x, origin_y, origin_z, dx, dy, dz)` — origin shifted by
  axis-aligned deltas (world Y-up), e.g. "30 cm above the sphere". Returns
  `{x, y, z}`.
- `along_direction(origin_x, origin_y, origin_z, target_x, target_y, target_z, distance)`
  — origin moved `distance` metres along the line toward the target ("closer to /
  further from"). Returns `{x, y, z}`.
- `scale_value(current, factor)` — scalar multiply for sizes ("3× bigger",
  "half"). Returns `{value}`.

### vec-mcp configuration

`vec_mcp_server.yaml`:

```yaml
host: 0.0.0.0
port: 8250
```

## video-mcp

`video-mcp` serves camera frames and recordings from two data paths:

- **Historical chunks** — reads the H.264 Annex B chunks the XR-Media-Hub's recorder
  writes to disk (tmpfs by default). `recordings_dir` must match the hub's
  `video_recording.out_dir`.
- **Live frames** — connects to the hub as a processor endpoint, tracks the most
  recent frame per participant, and pulls pixels on demand over the hub IPC
  sockets (`hub_pub` and `hub_push`).

All tools accept and return raw LiveKit participant identities; filesystem
sanitization happens internally and is recovered via `.identity` sidecars.

### video-mcp tools

`list_live_participants()` is always registered — identities currently connected
to the hub (live IPC roster). The frame tools depend on whether recording is
enabled (a chunk store on disk):

**Recording enabled** (`recordings_dir` set, hub `video_recording.enabled: true`):

- `get_frame_from_time(participant_id, second_ago, reference_time_us=0)` — frame
  at `anchor − second_ago` seconds, where the anchor is the wall clock
  (`reference_time_us=0`) or an explicit Unix-µs timestamp. Reads recorded NVENC
  chunks and decodes via NVDEC; `second_ago=0` short-circuits to the live IPC
  path. Returns a PNG path.
- `list_recorded_participants()` — identities with at least one chunk on disk.
- `get_video_stats(participant_id)` — `num_chunks`, `total_bytes`,
  `avg_chunk_bytes`, `earliest_us`, `latest_us`.
- `query_video(participant_id, start_us, end_us)` — concatenate the H.264 chunks
  overlapping the window into a file and return its path; the stream starts with
  an IDR frame.

**Recording disabled** (no chunk store): two live-only tools:

- `get_frame_from_time(participant_id, second_ago=0, reference_time_us=0)` — only
  `second_ago=0` is served; any non-zero past lookup returns an error. Use
  `list_live_participants` to confirm camera availability first.
- `get_latest_frame(participant_id)` — deprecated alias for the `second_ago=0`
  case; prefer `get_frame_from_time`.

### video-mcp configuration

`video_mcp_server.yaml`:

```yaml
recordings_dir: /dev/shm/xr-ai/recordings   # must match hub video_recording.out_dir
out_dir:        /tmp/xr_video_queries        # where query/frame outputs are written
hub_pub:        ipc:///tmp/xr_hub_pub        # hub PUB socket (live frames)
hub_push:       ipc:///tmp/xr_hub_in         # hub PUSH socket (frame requests)
host:           0.0.0.0
port:           8210
gpu_id:         0
```

## vlm-mcp

`vlm-mcp` is a thin FastMCP wrapper around the vision-language model in
`ai-services/vlm-server/` (Cosmos-Reason1-7B via vLLM). It has no hub IPC
subscription and no `xr-ai-agent` dependency: it just reads a local image and
forwards it to the VLM's OpenAI-compatible chat-completions endpoint via the
`xr-ai-models` SDK.

### vlm-mcp tools

- `ask_image(question, image_path)` — read the local PNG at `image_path`, encode
  it as a JPEG data URL, send it with `question` to `vlm-server`, and return the
  model's answer text. The file is read in an executor so the asyncio loop is
  never blocked.

For custom agents the two-step flow is: acquire a frame from `video-mcp`
(`get_frame_from_time(participant_id, second_ago=0)`) and pass that PNG path
straight into `ask_image`. The xr-render-demo worker uses a single-step
brain-local `look_at_current_frame(question)` tool instead, which turns the
camera on automatically and bypasses the MCP round-trip.

### vlm-mcp configuration

`vlm_mcp_server.yaml`:

```yaml
host:                  0.0.0.0
port:                  8240
models:
  vlm:
    kind:     preset:cosmos_vlm   # targets ai-services/vlm-server/
    base_url: http://localhost:8100
vlm_request_timeout_s: 60.0       # per-call httpx timeout
enable_thinking:       false      # true enables VLM chain-of-thought (slower, more accurate)
```

## transcript-mcp

`transcript-mcp` stores and queries transcript history keyed by a free-form
`source_id` string. Sources can be real LiveKit participant identities
(`alice@home`, `ipad-pro-1`) or synthetic names (`agent-vlm`, `tts`) — the store
does not interpret the value. Records persist as per-source JSONL files
(`{timestamp_us, text}` per line) alongside a `.identity` sidecar that recovers
the original name, so list and query round-trip cleanly across restarts.

### transcript-mcp tools

- `add_transcript(source_id, timestamp_us, text)` — append a segment; returns
  `{"ok": true}` (or an error if `text` is empty).
- `query_transcripts(source_id, start_us, end_us)` — all stored segments for
  `source_id` whose timestamp falls within `[start_us, end_us]` (Unix
  microseconds).
- `list_sources()` — all source IDs that have at least one stored transcript.
- `get_transcript_stats(source_id)` — summary statistics (`count`,
  `total_chars`, `earliest_us`, `latest_us`).

### transcript-mcp configuration

`transcript_mcp_server.yaml`:

```yaml
transcripts_dir: /tmp/xr_transcripts   # persistent JSONL storage
host:            0.0.0.0
port:            8200
```
