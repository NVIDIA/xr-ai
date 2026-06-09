<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Credentials

The launcher manages HuggingFace and NGC API tokens so they are never stored
in source files or YAML configs.

Tokens are cached in `~/.config/xr-ai/credentials.json` — outside any repo,
no `.gitignore` entry required. Values already in `os.environ` always take
priority (useful in CI or when you want to override the cache).

`run_stack` always calls `load_credentials()` before spawning child processes,
so any token saved in the credentials file, exported in the environment, or
stored by `huggingface-cli login` is injected into **every** subprocess in the
stack automatically — no per-sample wiring needed.

## HuggingFace token (`HF_TOKEN`)

**Optional.** The samples' default models are **public**, so they download
without a token. Set `HF_TOKEN` when you want:

- **Gated models** — any model whose HuggingFace page requires accepting a
  license / requesting access. These *require* a token (and license
  acceptance on your account) or the download fails.
- **Higher rate limits and faster downloads** — unauthenticated Hub requests
  are rate-limited and slower; an authenticated token lifts both.

The samples **do not prompt** for it. If `HF_TOKEN` is not set, the
orchestrator prints a one-line notice and continues (public models still
download). Provide it any one of these ways — all are picked up automatically:

```bash
# 1. Environment variable (highest priority; good for CI / one-off overrides)
export HF_TOKEN=hf_xxx

# 2. huggingface-cli login (writes ~/.cache/huggingface/token)
huggingface-cli login

# 3. Save it once for all samples (written to ~/.config/xr-ai/credentials.json
#    and ~/.cache/huggingface/token)
python3 -c "from xr_ai_launcher import ensure_credentials; ensure_credentials('HF_TOKEN')"
```

Get a token at <https://huggingface.co/settings/tokens>. `HF_TOKEN` is also
written to `~/.cache/huggingface/token` — the standard location
`huggingface_hub` checks — so child processes (e.g. `vlm-server`) find it even
when env-var inheritance is incomplete, and an existing `huggingface-cli login`
is reused without any further setup.

## NGC API key (`NGC_API_KEY`)

**Required only for the NIM model backend and `nvcr.io` image pulls.** It
authenticates both `nvcr.io` container pulls and **hosted NVIDIA NIM**
inference endpoints — a `models.yaml` entry with `api_key_env: NGC_API_KEY`
sends it as the `Authorization: Bearer` token (see
[`docs/ai-services.md`](ai-services.md#hosting-models-on-nvidia-nim)).

Samples that select the NIM backend call `ensure_credentials("NGC_API_KEY")`,
which **prompts once** (password-style, no echo) if the key isn't already
available and saves it for future runs — NIM cannot function without it, so the
prompt is intentional. Get a key at <https://ngc.nvidia.com/setup/api-key>, or
set it ahead of time:

```bash
export NGC_API_KEY=nvapi-xxx
```

## How a token is resolved

`load_credentials()` (always) and `ensure_credentials()` (NGC only) resolve in
this priority order, highest first:

1. Already set in `os.environ`
2. Saved in `~/.config/xr-ai/credentials.json`
3. Stored in `~/.cache/huggingface/token` (`HF_TOKEN` only)
4. *(interactive paths only)* prompted, then saved to both locations

`warn_if_missing(...)` runs steps 1–3 and, if the token is still absent, prints
an actionable notice and continues **without prompting** — this is how
`HF_TOKEN` is handled in the samples.

## Managing saved tokens

```bash
# View saved tokens
cat ~/.config/xr-ai/credentials.json

# Remove a token
python3 -c "
import json, pathlib
p = pathlib.Path.home() / '.config/xr-ai/credentials.json'
d = json.loads(p.read_text()); d.pop('HF_TOKEN', None); p.write_text(json.dumps(d, indent=2))
"
```
