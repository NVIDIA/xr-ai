<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Embedding and OCR clients (agent-sdk/xr-ai-models/xr_ai_models/_openai_compat.py:1044-1216, _ocr.py, presets/nemotron_embedding.py, nemotron_ocr.py)

## Role in a turn
Embedding: query-time vector for retrieval (RAG service does its own embedding server-side; client-side use is for custom agents). OCR: per-image text extraction when an agent reads text in a frame. Both single blocking request/response calls.

## Critical-path calls
### Embedding
1. `embed(texts)` -> `_embed_payload` (:1137-1155): empty input returns `[]` locally; else one `POST /v1/embeddings {model, input:[...]}`; sort rows by index; convert all floats in Python (`[[float(v) for v in row["embedding"]] ...]` :1155).
2. `_EmbeddingWithExtras.embed` (:1191-1216) when `default_extras.input_type` is set: splits a batch into `query` and `passage` groups and issues the requests SERIALLY (`for kind, group in groups.items(): await ...` :1208-1215), i.e. a mixed batch costs 2 sequential round trips.
3. `model_identity()` (:1073): GET `/v1/models`, used by consumers at startup for cache keys; not per turn.
### OCR
1. `_normalize_image` (sync file read/base64) :105.
2. `POST {base}/v1/ocr {"input":[{"type":"image_url","url": data-url}], "merge_levels":[...]}` :112-116; single image per call; no retry (`OCRRequestError` doc, :17-23).
3. `_parse_regions` validates every detection (python loops) :50-77.

## Fixed waits & timers
| Constant | Default | Where |
|---|---|---|
| Embedding timeout | 60.0 s | `_openai_compat.py:1056`, `_config.py:909` |
| OCR timeout | 120.0 s | presets/nemotron_ocr.py:10 |
| OCR health path | `/v1/health/ready` | presets/nemotron_ocr.py:9 |
| Retries | none | - |
| batch size limit | none client-side (server limit applies) | - |
| Pooling | httpx default | :1070, _ocr.py:88 |
| OCR input | inline PNG/JPEG only; no crop/resize "does not crop, sharpen... or change resolution" | _protocols.py:46-47, _ocr.py:108 |

## Cost drivers
Embedding: number of texts and tokens per text (server batched forward), vector dimension (JSON float list parse ~ dim*n floats; e.g. 2048-dim x 16 texts = 32 K floats ~ 3-5 ms Python conversion, guess). OCR: image resolution and text density, number of regions (response size and validation loop).

## Estimate formula
Embed: `t = t_rtt + t_queue + t_forward(sum_tokens) + t_json(n*dim)`, `*2` serial round trips for mixed query/passage batches. OCR: `t = t_b64 + t_upload + t_server_ocr(px, regions) + t_parse(regions * ~10 us)`. Needs measurement for server terms; client overhead 0.2-5 ms.

## Hidden latency / improvement candidates
* Mixed query/passage batches serialize two requests (:1208-1215): `asyncio.gather` over the two groups would cut that to one RTT+max(forward) (saves one RTT, ~1-10 ms plus overlap of GPU work).
* OCR on full-res frames: downscale/crop client-side (no resize in SDK) - saving depends on server (needs measurement).
* No retry: transient server error costs the caller a full re-request.
* Float conversion in pure Python (:1155): use list as returned (already floats from JSON) - saves ~0.1 ms/1k floats.

## What to measure live
`make_embedding(...).embed([...])` for batch 1/8/32 and with `input_type` mixed prefixes (`query: `/`passage: `); `make_ocr(...).read_text(png)` at 480p/1080p/4K with sparse vs dense text.

## Confidence
Structure/constants: code-derived; server times: needs measurement.
