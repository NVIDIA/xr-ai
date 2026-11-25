<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Current-frame capture and pixel conversion (agent-sdk/xr-ai-tools/xr_ai_tools/current_frame.py, _pixels.py, image.py, capture.py; waits live in agent-sdk/xr-ai-hub/xr_ai_hub/_live_frames.py, _image_capture.py)

## Role in a turn
`get_current_frame` is the usual first step of any vision turn: acquire a camera frame, JPEG-encode it, register it, and return an `ImageReference` that the VLM tool then resolves. It is serial in front of the VLM call.

## Critical-path calls
1. `CurrentFrameTool._get_current_frame` (`current_frame.py:126-156`).
2. `_fresh_live_frame` (`:106-124`): `frames.participants()` check (sync scan of latest signals) `:107`; `await frames.get(pid)` (`_live_frames.py:79-120`): if a signal fresher than `max_age_s` already exists, immediately `await endpoint.request_frame(signal)` (hub IPC round trip with pixel payload; cost needs measurement); else waits for a new frame signal up to `timeout_s`.
3. `await asyncio.to_thread(encode_image_bytes(frame_to_pil(frame)))` `:113-115`: pixel-format conversion (`_pixels.py:26-48`) then JPEG q90 (`:51-56`). Serial after the frame arrives.
4. `images.put(bytes, owner)` + `ImageFrame(...)` pydantic build `:116-124`.
5. Fallback if no fresh live frame: `await _captures.capture(pid)` (`_image_capture.py`): asks the client to capture and upload one still, up to `frame_timeout_s`; then `to_thread(_encoded_image_size)` -> `PIL.Image.open` + `verify()` `:142-147` (decodes headers/verifies the whole image) before returning.
6. Return value is a reference only; JPEG base64 happens later in the VLM client.

## Fixed waits & timers
| Constant | Default | Where |
|---|---|---|
| `frame_max_age_s` | 2.0 s (frame is "fresh" if pts younger than this) | `current_frame.py:73`, `_live_frames.py:38` |
| `frame_timeout_s` | 5.0 s (wait for a fresh signal / for client still capture) | `current_frame.py:74`, `_live_frames.py:39`; `_ClientImageCaptureSource(timeout_s=frame_timeout_s)` `:86-89`; `_image_capture.py:30` default 10.0 |
| JPEG quality | 90 | `_pixels.py:55` |
| `ImageRegistry.capacity` | 128 | `image.py:221` |
| frame freshness check | wall clock `time.time_ns() - pts_us < max_age` | `_live_frames.py:66-67` |
| `capture.py` target path component limit | 96 chars | `capture.py:81` |
| Capture start/stop | one `endpoint.send_return_data` each, no ack | `capture.py:157-163` |

## Cost drivers
Camera resolution and pixel format (I420/NV12 conversion is float32 numpy over full frame, `_pixels.py:16-23, 40-47`; RGBA/BGRA are cheap slices); hub IPC for pixels (size = W*H*bytes/pixel; 1080p I420 3.1 MB, RGBA 8.3 MB); client camera wakeup in capture fallback.

## Estimate formula
`t_frame = t_freshness_wait (0 if recent frame; else <= 1/fps; fallback capture: client-dependent, up to 5 s) + t_request_frame(hub IPC, size) + t_convert(format, px) + t_jpeg(px) + t_register`.
Guess (needs measurement): 1080p I420: convert 30-80 ms (float32 temporaries, `repeat` upsampling of chroma), JPEG q90 10-25 ms; RGB24/RGBA: convert 2-10 ms. Best ~15 ms / typical 50-120 ms / worst 5 s (timeout) + capture fallback.

## Hidden latency / improvement candidates
* I420/NV12 -> RGB through float32 numpy and `np.stack` (`_pixels.py:16-23`) is much slower than libyuv/PIL YCbCr or cv2.cvtColor (est. save 20-60 ms at 1080p). Not needed for a model that accepts JPEG? Hub could pass the encoded frame instead.
* Fallback capture path decodes/verifies the image (`image.verify()` `:162`) solely to get dimensions: ~5-30 ms for large JPEG; header-only read would be enough.
* Encoded once into registry; each VLM query re-base64s (VLM card).
* `frames.get` waits for a signal newer than 2 s: a stale-by-3 s pipeline stalls until next signal (up to 1/fps) - fine; but a stopped camera costs the entire `frame_timeout_s=5 s` before failing over to still capture, which then costs up to another 5 s: worst-case turn delay ~10 s in the no-video state. Reorder or shorten the timeout where known.
* JPEG q90 at native resolution inflates bytes for no VLM benefit (see VLM card).

## What to measure live
`CurrentFrameTool.execute(CurrentFrameRequest(participant_id=...))` end-to-end with video on (fresh frame) and video off (capture fallback); separately time `hub.request_frame` and `encode_image_bytes(frame_to_pil(frame))` per pixel format and resolution (public helpers are importable from `xr_ai_tools._pixels` for a benchmark script).

## Confidence
Control flow and constants: code-derived. Hub IPC, conversion times: needs measurement (numbers above are guesses).
