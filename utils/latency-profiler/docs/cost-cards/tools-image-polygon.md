<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Image polygon fill tool (agent-sdk/xr-ai-tools/xr_ai_tools/image_polygon.py, image.py)

## Role in a turn
`fill_image_polygon`: local PIL edit used for visual annotation (e.g. mark an object). On the critical path only when the model calls it; pure CPU.

## Critical-path calls
1. `images.resolve(ref)` `:155` (dict op).
2. `await asyncio.to_thread(fill_polygon_magenta, source, coordinates)` `:164-168`: `_open_image` (`Image.open` + `.load()` + `.convert`) `:82-96`, `_validate_polygon` `:99-119`, `ImageDraw.polygon`, then `image.save(PNG)` `:122-137`. Serial in one thread task.
3. `images.put_derived(png_bytes, source=ref)` `:177`.
Source of `source`: bytes (JPEG from `get_current_frame`) or local path; external URLs are rejected `:88-90`.

## Fixed waits & timers
None (no timeouts, no sleeps). PNG output is lossless (`format="PNG"` default compress level 6). Registry capacity 128 `image.py:221`.

## Cost drivers
Image resolution (full JPEG decode + PNG encode); PNG compression of a photographic image is slow and large; number of polygon vertices (negligible).

## Estimate formula
`t = t_decode(JPEG px) + t_draw(~0.1 ms) + t_png_encode(px)`. Guess: 1080p JPEG decode 8-15 ms; PNG encode (compress level 6) of a photo 100-300 ms; 4K ~4x. Needs measurement.

## Hidden latency / improvement candidates
* PNG (zlib level 6) on photographic frames is the dominant cost (est. 100-300 ms at 1080p, guess); `compress_level=1` or JPEG output would cut it 3-10x (changes losslessness contract, `image_polygon.py:60` says lossless PNG).
* The output PNG (2-5 MB) is later base64'd into a VLM request if shown to a VLM: much larger upload than the source JPEG (0.3-0.6 MB) (see VLM card).
* Always RGBA-detection via `getbands`/`info` cheap.

## What to measure live
`ImagePolygonFillTool(images=registry).execute(ImagePolygonFillRequest(image=ref, coordinates=[...]))` at 720p/1080p/4K; or time `fill_polygon_magenta(bytes, coords)` directly (public function, `__all__`).

## Confidence
Control flow: code-derived; ms numbers: guess.
