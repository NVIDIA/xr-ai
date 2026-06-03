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

`HF_TOKEN` is additionally written to `~/.cache/huggingface/token`, the same
file used by `huggingface-cli login`. This means:
- Child processes find it without relying on env-var inheritance.
- If you've already run `huggingface-cli login`, no prompt appears.

## Prompting for a token

Call `ensure_credentials` **before** `run_stack(...)` in any orchestrator
that needs a token:

```python
from xr_ai_launcher import ensure_credentials, run_stack

def run() -> None:
    ensure_credentials("HF_TOKEN")          # prompts once, saves for future runs
    run_stack(PROCESSES, _BASE)
```

Supported tokens: `HF_TOKEN`, `NGC_API_KEY`. The user is shown a prompt
(password-style, no echo) that explains what the token is for and the
consequence of skipping it, alongside a link to generate one. Pressing
Enter without typing skips the token (left unset, not saved).

`NGC_API_KEY` authenticates both `nvcr.io` image pulls and **hosted NVIDIA
NIM** inference endpoints — a `models.yaml` entry with
`api_key_env: NGC_API_KEY` sends it as the `Authorization: Bearer` token (see
[`docs/ai-services.md`](ai-services.md#hosting-models-on-nvidia-nim)). Because
`run_stack` injects saved credentials into every subprocess, no per-sample
wiring is needed once the key is saved.

## Automatic injection

`run_stack` always calls `load_credentials()` internally before spawning child
processes, so any token already saved in the credentials file is available to
every subprocess in the stack — even orchestrators that never call
`ensure_credentials` directly.

## Managing saved tokens

```bash
# View saved tokens
cat ~/.config/xr-ai/credentials.json

# Remove a token (re-run will prompt again)
python3 -c "
import json, pathlib
p = pathlib.Path.home() / '.config/xr-ai/credentials.json'
d = json.loads(p.read_text()); d.pop('HF_TOKEN', None); p.write_text(json.dumps(d, indent=2))
"
```
