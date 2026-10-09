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

The service supports text-only choice questions. A request must name the
configured model, include state, and provide at least one question with a
string instruction and two or more named choices. An empty instruction uses
the upstream question-ID fallback. The service rejects
multimodal fields, unsupported question types, model-name mismatches, oversized
bodies, and tokenized inputs above `max_length`. This length check uses the
release tokenizer before inference so state is never silently truncated by the
upstream encoder.

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

From `model-server-samples/clef-flash/`, run:

```bash
export UV_CONFIG_FILE=../../uv.toml
uv sync
uv run clef_flash_model
```

The sample starts `services/clef-server` persistently on `127.0.0.1:8120`.
Its `yaml/clef_server.yaml` pins model revision
`17f0b0ad64efb65d273590632833508766b2aae6` from the
`Cloudflare/clef-flash` model repository and downloads it into the repository's
ignored `models/` cache on first launch. To use an already extracted copy of
that snapshot, set `model_path` to its directory in the YAML. The local path
must contain the release's required model files; its revision is supplied by
the configuration rather than checked against local artifact metadata.

Without `model_path`, the server resolves the configured revision from the
cache first and passes the snapshot directory to the native loader. An absent
snapshot follows the online download path. With `HF_HUB_OFFLINE=1`, a cache
miss instead fails before model loading with the model, revision, and cache
directory. Offline startup assumes a previously working cache. Refer to
{ref}`starting-model-services-without-network-access` for cache behavior.

With `UV_CONFIG_FILE` still set, use `uv run clef_flash_model --stop` from the
sample directory to stop only a listener carrying the matching Clef ownership
and port markers. It waits for that process to exit, including request and native
inference cleanup, before reporting success; closing the HTTP listener alone
does not complete shutdown. Linux process handles keep termination and timeout
escalation tied to the same verified process. The stop command never treats a
Docker container on that port as the Clef service. A server already listening on the configured port is reused
only when it has those ownership markers and its health response matches the
complete requested server configuration; any other listener causes startup to
fail before loading model weights.

The YAML sets the bind address, port, device, inference dtype, maximum token
length, and maximum request-body size. The checked-in values use `cuda:0`,
BF16, 4096 tokens, and a 1 MiB body limit. Requests that exceed the context
limit receive HTTP 422; increase `max_length` only when GPU memory allows it.

## HTTP contract

`POST /v1/systemone` accepts the SystemOne request fields `model`, `state`,
and `questions`. The response contains the configured model name, one choice
answer per question, input-token usage, and the pinned revision. The service
does not accept images, videos, `noul`, or `score` questions. The pinned model
rounds native probabilities to four decimal places. Before returning them, the
service validates their shape and values, normalizes them to a complete
distribution, and derives confidence from the selected normalized probability.

`GET /health` reports the Clef service, model and revision, plus a deterministic
fingerprint of the active non-secret server configuration. The launcher uses
that complete identity only to decide whether an owned listener is reusable.

Workers use `xr_ai_models.make_decision` and the model profile's SystemOne
adapter. They do not call this HTTP endpoint directly.
