<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Marker tracking (agent-sdk/xr-ai-tools/xr_ai_tools/marker_tracking.py)

## Role in a turn
`track_markers` tool: CPU-only QR/ArUco detection on a live camera frame (or a registry image). No model server involved. On the critical path whenever the agent calls it; the cost is pure local compute plus frame fetch.

## Critical-path calls
1. If no `image` arg: `await frames.get(pid)` (fresh frame wait + hub `request_frame`, see tools-current-frame-pixels.md) `:347`. Else `images.resolve` `:357`.
2. `await endpoint.set_status("processing", pid)` `:362` (hub round trip, serial, when `manage_status=True`, the default).
3. `await to_thread(frame_to_pil, frame)` (or `_open_image` decode of bytes/path, `:366-368`) then a SECOND `await to_thread(self._extract, image)` `:369`: serial, two thread hops.
4. `_extract` (`:386-394`): `_full_frame_scans` -> grayscale + (if long edge < 3840) a LANCZOS upscaled copy up to long edge `min(3840, 6*long_edge)` (`:137-168`); QR: `zxingcpp.read_barcodes` on each scan (1x and enlarged) `:171-201`; ArUco: `detectMarkers` on each scan `:254-282`; merge of duplicates by O(n^2) overlap (`:219-243`).
5. `finally: await endpoint.set_status("idle", pid)` `:377-378` - another hub round trip that the tool result waits for.

## Fixed waits & timers
| Constant | Default | Where |
|---|---|---|
| `_MARKER_SCAN_LONG_EDGE` | 3840 px | `:29` |
| `_MARKER_SCAN_MAX_SCALE` | 6x | `:30` |
| scans per frame | 2 (native + upscaled) whenever long edge < 3840 (always for 720p/1080p) | `:142, :148-167` |
| `frame_max_age_s` / `frame_timeout_s` | 2.0 s / 5.0 s | `:297-298` |
| ArUco dictionary | DICT_4X4_50 | `:296` |
| `manage_status` | True (2 hub calls) | `:299` |
| Failure handling | exceptions -> "Marker extraction failed" result, no retry | `:370-375` |

## Cost drivers
Camera resolution; upscaled 3840-edge image (a 1080p frame becomes 3840x2160 = 8.3 Mpx gray, 4.3x pixels) fed to zxing and ArUco; #markers (merge O(n^2) per scan); enabled marker families (QR and ArUco each run on both scans).

## Estimate formula
`t_track = t_frame_get + 2*t_status_rpc + t_to_pil(px) + t_gray_convert + t_lanczos(3840x2160) + 2*t_zxing(scan) + 2*t_aruco(scan) + t_merge`.
Guess, needs measurement: LANCZOS resize to 4K gray 40-120 ms; zxing on 4K 50-300 ms when no QR (scan fails exhaustively), faster when found; ArUco 4K 20-80 ms. Best ~80 ms / typical 200-500 ms / worst >1 s on 720p->4K with no marker found.

## Hidden latency / improvement candidates
* The upscale pass doubles (or more) the compute even when the marker was already decoded at 1x. Run native scan first and only upscale if nothing found (the merge step then not needed). Potential saving 50-80 % of CPU time when markers are readable at native size.
* Two serial `to_thread` hops (`:366`, `:369`) could be one; saves ~0.2 ms (negligible).
* `set_status` before/after serial hub RPCs add 2 round trips (needs measurement, likely 1-5 ms each); the final "idle" is awaited inside the tool result path; fire-and-forget would remove it from the critical path.
* QR and ArUco are independent: run in parallel threads (OpenCV and zxing release the GIL) to cut ~50 % of detection time when both enabled.
* `thread` pool default size is limited (`asyncio.to_thread` default executor ~ min(32, cpu+4)); concurrent calls queue.

## What to measure live
`MarkerTrackingTool(endpoint=..., images=registry).execute(MarkerTrackingRequest(participant_id=..., image=ref))` with a stored 720p/1080p/4K image: with marker present/absent, `marker_types` QR-only / ArUco-only / both. Separately time `_full_frame_scans`, `zxingcpp.read_barcodes` via a benchmark script.

## Confidence
Call structure and constants: code-derived. All CPU milliseconds: guess / needs measurement.
