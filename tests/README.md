<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai integration tests

Multi-client / multi-agent coverage for the XR-Media-Hub IPC pipeline.

## Layout

| File                          | What it covers                                                          |
|-------------------------------|--------------------------------------------------------------------------|
| `conftest.py`                 | Shared fixtures: `hub`, `make_connector`, `make_processor`, `settle`.   |
| `_helpers.py`                 | `setup_client` / `teardown_clients` / `wait_for` / `wait_for_subscribed` / `silence`. |
| `_helpers_subprocess.py`      | `pick_free_port` / `port_is_free` / `health_ok` — shared port + HTTP-health primitives for subprocess-spawning tests. |
| `test_hub_data_routing.py`    | Topic preservation; data fanout to multiple agents; per-client attribution. |
| `test_participant_events.py`  | Join/leave fanout; auto-maintained `connected_participants` roster.      |
| `test_audio_routing.py`       | Inbound audio attribution; return audio targeted only at one connector. |
| `test_return_routing.py`      | Return data isolation; drops to unknown participants are silent.        |
| `test_return_audio_flush.py`  | `ReturnAudioFlush` control message routes to the right connector.       |
| `test_multi_agent.py`         | Multiple `ProcessorEndpoint`s observing the same hub.                    |
| `test_cross_talk.py`          | 3+ clients × 3+ agents matrix, interleaved fan-in, late-join / leave, disjoint-filter isolation — full no-cross-talk guarantee. |
| `test_subscriptions.py`       | Participant-keyed subscription API: `Subscribe` filters, `auto_subscribe` on/off, per-pid filter override, roster catch-up, prefix-collision isolation, idempotency. |

## Running

The IPC suite runs without Docker or LiveKit — it speaks ZMQ over
`ipc://` only.

```bash
cd xr-ai/tests
uv sync
uv run pytest -v
```

The same command runs in GitHub Actions on every push and pull request
via [`.github/workflows/tests.yml`](../.github/workflows/tests.yml),
matrixed across Python 3.11 and 3.12 on `ubuntu-latest`. CI invokes
pytest with `-m "not gpu"` so anything that needs real hardware is left
to the developer box.

## GPU / Docker / NVENC tests

Tests that need a real GPU, Docker, or NVENC carry the `gpu` marker and
are skipped in CI. Run them locally with:

```bash
bash tests/run_local_gpu_tests.sh        # or pass extra pytest args
```

Mark new tests with `@pytest.mark.gpu` whenever they need any of those
resources.

## MCP server smoke tests

CPU-viable MCP servers get a subprocess smoke test that spawns the server,
polls `McpClient.list_tools()` for readiness, then drives the tool surface
over StreamableHTTP (pattern: `test_transcript_mcp.py`). These carry the
`integration` marker and run in CI:

* `test_transcript_mcp.py` — JSONL transcript store.
* `test_vec_mcp.py` — pure-math spatial primitives (no external state).

**oxr-mcp is not CPU-viable.** It imports `isaacteleop` (native OpenXR +
HeadTracker bindings) at module top and opens a headless OpenXR session
against a running CloudXR runtime — neither installs nor runs on a CPU-only
CI box. It is therefore intentionally absent from `pyproject.toml`, and
`test_oxr_mcp.py` self-skips via `pytest.importorskip("isaacteleop")`. To
exercise it manually on a GPU host with CloudXR, install oxr-mcp into the
test venv (`uv pip install -e ../agent-mcp-servers/oxr-mcp`) with the
CloudXR runtime available, then spawn `python -m oxr_mcp_server --config
<yaml>` and call `get_health` / `get_head_pose` once a streaming client is
connected.

## Test taxonomy

* **Multi-client** — every test that creates two `ConnectorEndpoint`s
  represents two distinct clients; they share the hub but never each
  other's return traffic.
* **Multi-agent** — every test that creates two `ProcessorEndpoint`s
  represents two independent agents; both observe the full inbound
  stream and may emit return traffic for any participant.
* **Combined** — `test_cross_talk.py::test_three_clients_three_agents_full_matrix`
  is the canonical end-to-end multi-client + multi-agent scenario; every
  agent replies to every client and we assert no message is lost,
  duplicated, or delivered to the wrong client.

## No-cross-talk guarantee

`test_cross_talk.py` is the authoritative suite for the invariant
"participant *X*'s return traffic must never reach participant *Y*". It
asserts isolation under each of:

* 3 clients with 1 agent — each of data / audio / flush separately;
* 1 client with 3 agents — all agents see the inbound stream;
* 3 clients with 3 agents — full matrix of return-data deliveries;
* 4 clients with 100 interleaved messages — origin attribution + per-pid order;
* late-join (a new client doesn't retroactively see prior messages);
* leave (a left participant receives no further traffic);
* disjoint filter modes (agents on different `Subscribe` flags never observe each other's events).
