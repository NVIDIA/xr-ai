<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Magpie TTS NIM

Launcher-owned NVIDIA Speech NIM service for Magpie Multilingual. It exposes
offline WAV synthesis at `/v1/audio/synthesize` and streaming PCM synthesis at
`/v1/audio/synthesize_online` on port 9000.

The first start downloads the NIM image and builds an optimized model store in
`models/nim-magpie-tts/`; this can take about 20 minutes. Later starts reuse the
exported store and normally become ready in under a minute.

Set `NGC_API_KEY`, then run:

```bash
uv sync
uv run magpie_tts_nim_server --config magpie_tts_nim_server.yaml
```

Docker Engine, NVIDIA Container Toolkit, and enough free GPU memory for the NIM
are required. The wrapper stops only containers carrying its ownership label.
