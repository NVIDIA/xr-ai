<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Clef-Flash local decision server

The optional Clef-Flash 9B service exposes the upstream joint-schema model over
the typed SystemOne HTTP contract used by `make_decision`. It loads the pinned
release's `joint_schema_model.py` directly from the model directory and calls
its native `load_release_model` and `systemone` functions. The service does not
route decisions through a generation endpoint.

The service supports choice questions about text and ordered images. A request
must name the configured model, include state, and provide at least one question with a
string instruction and two or more named choices. An empty instruction uses
the upstream question-ID fallback. The service rejects videos, invalid image
data, unsupported question types, model-name mismatches, oversized bodies, and
tokenized inputs above `max_length`. This length check uses the release
multimodal processor before inference so state is never silently truncated by
the upstream encoder.

The server loads weights once, runs a two-choice warmup, and signals launcher
readiness only after the model and HTTP listener are ready. Inference executes
one request at a time on a dedicated worker thread; HTTP health and model
metadata remain responsive while requests wait. Client cancellation, including
repeated cancellation, does not release the inference lock until the running
worker call finishes. `GET /health` and
`GET /v1/models` include model and `model_revision` provenance.
`POST /v1/systemone` returns the standard answers and usage object with the
same provenance fields.

## Launching

Select `clef` in a sample's model-server stack, or start the service from the
repository root:

```bash
uv --config-file uv.toml run --project services/clef-server clef_server \
  --config services/clef-server/clef_server.yaml
```

The service listens on `127.0.0.1:8120`. Its `services/clef-server/clef_server.yaml` pins model revision
`17f0b0ad64efb65d273590632833508766b2aae6` from the
`Cloudflare/clef-flash` model repository. The service resolves the configured
pinned snapshot from the repository's ignored `models/` cache first and passes
its directory to the native loader. A cache
miss follows the online download path; with `HF_HUB_OFFLINE=1`, it fails before
model loading with the model, revision, and cache directory. Offline startup
assumes a previously working cache. Refer to
{ref}`starting-model-services-without-network-access` for cache behavior.

A sample stack that selects `clef` stops it with its `--stop` command, which
signals only a listener carrying the stack's managed-port markers. A server
already listening on the configured port is reused only when it has those
ownership markers and its health response matches the
complete requested server configuration; any other listener causes startup to
fail before loading model weights.

The YAML sets the bind address, port, device, inference dtype, maximum token
length, and maximum request-body size. The checked-in values use `cuda:0`,
BF16, 4096 tokens, and a 1 MiB body limit. Requests that exceed the context
limit receive HTTP 413, the same status as an oversized body, so a client can
retry with less input; increase `max_length` only when GPU memory allows it.

(clef-service-updates)=
## Updating a running service

Before updating the checkout or changing server configuration, stop the service,
then start it again. Model and configuration identity do not identify the
running application code version. Restarting ensures that requests reach the
updated implementation, including support for image decisions.

## HTTP contract

`POST /v1/systemone` accepts the SystemOne request fields `model`, `state`,
and `questions`. The response contains the configured model name, one choice
answer per question, input-token usage, and the pinned revision. The service
accepts an optional `images` array of inline JPEG or PNG base64 data URLs.
Image ordering is preserved; describe target and reference roles in `state` or
question instructions. URLs and paths are never fetched or opened by the server.
Videos, `noul`, and `score` questions remain unsupported. The pinned model
rounds native probabilities to four decimal places. Before returning them, the
service validates their shape and values, normalizes them to a complete
distribution, and derives confidence from the selected normalized probability.

`GET /health` reports the Clef service, model and revision, plus a deterministic
fingerprint of the active non-secret server configuration. The launcher uses
that complete identity only to decide whether an owned listener is reusable.

Workers use `xr_ai_models.make_decision` and the model profile's SystemOne
adapter. They do not call this HTTP endpoint directly.

## Image decisions

Use the typed decision client for local image files as `pathlib.Path` objects,
encoded bytes, or inline JPEG and PNG data URL strings. Files must match their
JPEG or PNG suffix; filesystem path strings are not accepted. For example, with
an existing configured `decision` service:

```python
from pathlib import Path
from xr_ai_models import DecisionQuestion

result = await decision.decide(
    "Image 1 is the target piece; image 2 is a labeled reference: white bishop.",
    {"piece": DecisionQuestion(
        instructions="Classify the target piece.",
        criteria={"pawn": "Pawn", "bishop": "Bishop", "other": "Another piece"},
    )},
    images=[Path("target.jpg"), Path("bishop-reference.png")],
)
```

Before sending images to a service started by an older checkout, follow
{ref}`clef-service-updates`.

The client embeds local inputs into data URLs and rejects remote URLs. It omits
`images` for text-only calls, preserving compatibility with older servers. Image
requests require an upgraded endpoint; errors are surfaced without a text-only
fallback. Other `DecisionService` implementations need to accept the optional
`images` keyword when supporting this capability.

The existing body-size limit includes base64 media. Before allocating RGB
buffers, the server derives image-count and aggregate source-pixel budgets from
the loaded processor's patch size, spatial merge size, and minimum resized
pixel area. With the pinned processor, image count is bounded to
`max_length // 64` and aggregate source pixels to `max_length * 32 * 32`,
using 16-pixel patches and a two-by-two spatial merge. This
conservative admission limit applies to original dimensions even if the model
would resize them. Resize large source photos on the client before submitting
crops. The processor then checks the actual combined text, image, and question
token count against `max_length`; invalid images receive HTTP 422, and excess
tokens or body bytes receive HTTP 413.
Request-controlled preprocessing options (`media_kwargs`) are rejected with
HTTP 422: resize settings belong to the server's loaded processor.

Inputs use Pillow's RGB conversion without EXIF rotation, alpha compositing,
or high-bit-depth normalization. Prepare upright, opaque, 8-bit images on the
client when those distinctions matter.

Image decoding, processing, and inference share the existing serialized worker.
Decoded images are released on completion or failure. Requests reuse the loaded
model weights; each request still processes its images, including references.
There is no cross-request reference cache. Adding references increases input
processing and does not itself guarantee better decisions.
