# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Credential manager — stores HF_TOKEN and NGC_API_KEY outside the repo.

Credentials are saved in ~/.config/xr-ai/credentials.json (never inside any
git repo, no .gitignore entry needed).  Values already present in os.environ
always take priority over saved values.

HF_TOKEN is additionally written to ~/.cache/huggingface/token — the standard
location checked by huggingface_hub regardless of environment variables.  This
means child processes (e.g. vlm-server) find the token even when env-var
inheritance is incomplete (e.g. under certain uv run configurations).  It also
integrates with tokens already set via `huggingface-cli login`.

Typical usage in an orchestrator::

    from xr_ai_launcher import ensure_credentials

    ensure_credentials("HF_TOKEN")          # prompt once, cache forever
    asyncio.run(run_stack(PROCESSES, _BASE))
"""
from __future__ import annotations

import getpass
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_CREDS_FILE   = Path.home() / ".config" / "xr-ai" / "credentials.json"
_HF_TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"
_HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
_HF_CHECK_TIMEOUT_S = 5

# Per-token prompt info: (label, signup URL, "why this matters" blurb).
# The blurb is shown to the user so they understand the consequence of
# skipping. Skipping is still allowed — see ensure_credentials().
_KNOWN: dict[str, tuple[str, str, str]] = {
    "HF_TOKEN": (
        "HuggingFace token",
        "https://huggingface.co/settings/tokens",
        "Without HF_TOKEN, Hub requests are rate-limited: large checkpoint "
        "downloads can stall indefinitely with no error output, and gated "
        "models fail to download.",
    ),
    "NGC_API_KEY": (
        "NGC API key",
        "https://ngc.nvidia.com/setup/api-key",
        "Some NGC-hosted models and containers require authentication to download.",
    ),
}


def _read() -> dict[str, str]:
    try:
        data = json.loads(_CREDS_FILE.read_text())
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write(data: dict[str, str]) -> None:
    _CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CREDS_FILE.write_text(json.dumps(data, indent=2) + "\n")
    _CREDS_FILE.chmod(0o600)


def _read_hf_token_file() -> str:
    """Read the token written by `huggingface-cli login`."""
    try:
        return _HF_TOKEN_FILE.read_text().strip()
    except OSError:
        return ""


def _write_hf_token_file(token: str) -> None:
    """Write to ~/.cache/huggingface/token so huggingface_hub finds it in-process."""
    _HF_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HF_TOKEN_FILE.write_text(token + "\n")
    _HF_TOKEN_FILE.chmod(0o600)


def load_credentials() -> None:
    """
    Inject saved credentials into os.environ.

    Values already set in os.environ are left unchanged.  For HF_TOKEN, also
    checks ~/.cache/huggingface/token (written by `huggingface-cli login`) as a
    fallback so existing HuggingFace logins are picked up automatically.
    """
    saved = _read()
    for key, value in saved.items():
        os.environ.setdefault(key, value)

    # Pick up tokens from huggingface-cli login, in case the user authenticated
    # that way rather than through ensure_credentials.
    if not os.environ.get("HF_TOKEN"):
        hf = _read_hf_token_file()
        if hf:
            os.environ["HF_TOKEN"] = hf


def warn_if_missing(*names: str) -> None:
    """
    Non-interactively surface missing optional tokens — NEVER prompts.

    Loads saved / `huggingface-cli login` / env tokens, then prints an
    actionable notice per still-missing token and continues, so a missing
    token is visible early without blocking startup on an interactive prompt.
    """
    load_credentials()
    for name in names:
        if os.environ.get(name):
            continue
        label, url, why = _KNOWN.get(name, (name, "", ""))
        print(f"\n[credentials] {label} not set — continuing without it.", file=sys.stderr)
        if why:
            print(f"  {why}", file=sys.stderr)
        if url:
            how = ("`export HF_TOKEN=...` or `huggingface-cli login`"
                   if name == "HF_TOKEN" else f"`export {name}=...`")
            print(f"  To enable it: get one at {url}, then {how}.", file=sys.stderr)
        print("  See docs/source/getting_started/credentials.md.", file=sys.stderr)


def _hf_token_rejected(token: str) -> bool:
    """True only when the Hub answers 401 for *token*; any other outcome is unknown.

    Offline machines and Hub outages must not block launches whose checkpoints
    are already cached, so network errors and non-401 statuses count as valid.
    """
    request = urllib.request.Request(
        _HF_WHOAMI_URL, headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_HF_CHECK_TIMEOUT_S):
            return False
    except urllib.error.HTTPError as exc:
        return exc.code == 401
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _reject_invalid_hf_token(from_env: bool) -> None:
    """Exit with the real cause when the resolved HF_TOKEN is rejected by the Hub.

    A rejected token is otherwise silent: huggingface_hub downgrades to
    anonymous requests and the Hub only logs "unauthenticated requests".
    """
    token = os.environ["HF_TOKEN"]
    if not _hf_token_rejected(token):
        return
    label, url, _ = _KNOWN["HF_TOKEN"]
    source = ("the HF_TOKEN environment variable (check your shell profile)" if from_env
              else f"{_CREDS_FILE} or {_HF_TOKEN_FILE}")
    print(f"\n[credentials] {label} is set but the Hugging Face Hub rejected it (HTTP 401).",
          file=sys.stderr)
    print(f"  It came from {source}; it is likely revoked or expired. "
          "Downloads would silently run unauthenticated.", file=sys.stderr)
    file_token = _read_hf_token_file()
    if from_env and file_token and file_token != token and not _hf_token_rejected(file_token):
        print(f"  The token in {_HF_TOKEN_FILE} is valid, but the environment variable "
              "takes priority. Unset or update HF_TOKEN to use it.", file=sys.stderr)
    else:
        print(f"  Create a new token at {url}, then update it.", file=sys.stderr)
    sys.exit(2)


def require_credentials(*names: str, allow_missing: bool = False) -> None:
    """
    Exit non-zero unless each named token is available; NEVER prompts.

    *allow_missing* downgrades to ``warn_if_missing`` behavior.
    """
    if allow_missing:
        warn_if_missing(*names)
        return
    from_env = bool(os.environ.get("HF_TOKEN"))
    load_credentials()
    missing = [name for name in names if not os.environ.get(name)]
    if not missing:
        if "HF_TOKEN" in names:
            _reject_invalid_hf_token(from_env)
        return
    for name in missing:
        label, url, why = _KNOWN.get(name, (name, "", ""))
        print(f"\n[credentials] {label} is required but not set.", file=sys.stderr)
        if why:
            print(f"  {why}", file=sys.stderr)
        if url:
            how = ("`export HF_TOKEN=...` or `huggingface-cli login`"
                   if name == "HF_TOKEN" else f"`export {name}=...`")
            print(f"  Get one at {url}, then {how}.", file=sys.stderr)
        print("  Configure the credential before starting. "
              "See docs/source/getting_started/credentials.md.", file=sys.stderr)
    sys.exit(2)


def ensure_credentials(*names: str) -> None:
    """
    Ensure each named token is available in os.environ.

    Priority (highest first):
        1. Already set in os.environ
        2. Saved in ~/.config/xr-ai/credentials.json
        3. Stored in ~/.cache/huggingface/token  (HF_TOKEN only)
        4. Prompted interactively and then saved to both locations

    Any token entered interactively is saved so future runs are non-interactive.
    Pressing Enter without typing skips the token (left unset, not saved).
    """
    saved   = _read()
    updated = False

    for name in names:
        if os.environ.get(name):
            continue                         # env var takes priority

        if name in saved:
            os.environ[name] = saved[name]   # use cached value
            continue

        # For HF_TOKEN, also check the huggingface-cli login location.
        if name == "HF_TOKEN":
            hf = _read_hf_token_file()
            if hf:
                os.environ["HF_TOKEN"] = hf
                saved["HF_TOKEN"]      = hf
                updated                = True
                continue

        label, url, why = _KNOWN.get(name, (name, "", ""))
        print(f"\n[credentials] {label} not found.", file=sys.stderr)
        if why:
            print(f"  {why}", file=sys.stderr)
        if url:
            print(f"  Get one at: {url}", file=sys.stderr)
        value = getpass.getpass(f"  {label} (press Enter to skip): ").strip()
        if not value:
            print(f"[credentials] Skipping {name} — left unset.", file=sys.stderr)
            continue

        os.environ[name] = value
        saved[name]      = value
        updated          = True

        if name == "HF_TOKEN":
            _write_hf_token_file(value)
            print(f"[credentials] HF_TOKEN saved to {_CREDS_FILE} and {_HF_TOKEN_FILE}",
                  file=sys.stderr)
        else:
            print(f"[credentials] {name} saved to {_CREDS_FILE}", file=sys.stderr)

    if updated:
        _write(saved)
