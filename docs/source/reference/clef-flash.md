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
metadata remain responsive while requests wait. `GET /health` and
`GET /v1/models` include model and `model_revision` provenance.
`POST /v1/systemone` returns the standard answers and usage object with the
same provenance fields.

## Launching

From `model-server-samples/clef-flash/`, run:

```bash
uv --config-file ../../uv.toml sync
uv --config-file ../../uv.toml run clef_flash_model
```

The sample starts `services/clef-server` persistently on `127.0.0.1:8120`.
Its `yaml/clef_server.yaml` pins model revision
`17f0b0ad64efb65d273590632833508766b2aae6` from the
`Cloudflare/clef-flash` model repository and downloads it into the repository's
ignored `models/` cache on first launch. To use an already extracted copy of
that snapshot, set `model_path` to its directory in the YAML. The local path
must contain the release's required model files; its revision is supplied by
the configuration rather than checked against local artifact metadata.
Use `uv --config-file ../../uv.toml run clef_flash_model --stop` from the
sample directory to stop only the configured Clef listener. A server already
listening on the configured port is reused only when its health response names
the same model and revision; any other listener causes startup to fail before
loading model weights.

The YAML sets the bind address, port, device, inference dtype, maximum token
length, and maximum request-body size. The checked-in values use `cuda:0`,
BF16, 4096 tokens, and a 1 MiB body limit. Requests that exceed the context
limit receive HTTP 422; increase `max_length` only when GPU memory allows it.

## HTTP contract

`POST /v1/systemone` accepts the SystemOne request fields `model`, `state`,
and `questions`. The response contains the configured model name, one choice
answer per question, input-token usage, and the pinned revision. The service
does not accept images, videos, `noul`, or `score` questions.

Workers use `xr_ai_models.make_decision` and the model profile's SystemOne
adapter. They do not call this HTTP endpoint directly.
