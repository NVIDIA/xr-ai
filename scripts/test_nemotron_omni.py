#!/usr/bin/env python3
"""
Manual test: nemotron_omni_llm_server (vLLM voice/conversational entry point).

Launches the service via `uv run`, waits for vLLM to finish loading weights,
runs tests, then stops it.

Usage:
    python3 scripts/test_nemotron_omni.py

    # Override repo root if needed:
    REPO=/path/to/xr-ai python3 ...

NOTE: vLLM weight loading takes several minutes on first run.
      TIMEOUT defaults to 600s (10 min) to accommodate this.
"""
import atexit
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(os.environ.get("REPO", str(Path(__file__).resolve().parents[1])))
PROJECT = REPO / "ai-services/llm/nemotron_omni"
COMMAND = "nemotron_omni_llm_server"
CONFIG  = PROJECT / f"{COMMAND}.yaml"

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8107")
# nemotron_omni serves as "llm" via --served-model-name
MODEL    = "llm"
TIMEOUT  = int(os.environ.get("TIMEOUT", "600"))

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

_proc: subprocess.Popen | None = None


def _stop():
    if _proc and _proc.poll() is None:
        print("\n[teardown] Stopping vLLM service…")
        _proc.terminate()
        try:
            _proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _proc.kill()

atexit.register(_stop)


def start_service():
    global _proc
    cmd = ["uv", "run", "--project", str(PROJECT), COMMAND, "--config", str(CONFIG)]
    print(f"[start] {' '.join(cmd)}")
    _proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def wait_ready() -> bool:
    print(f"Waiting for vLLM at {BASE_URL}/v1 (up to {TIMEOUT}s — weight load takes a few minutes)…")
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        if _proc and _proc.poll() is not None:
            print(f"{RED}Service process exited early (rc={_proc.returncode}){RESET}")
            return False
        try:
            urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=3)
            print(f"{GREEN}Ready.{RESET}")
            return True
        except Exception:
            time.sleep(5)
    return False


def _post(body: dict) -> dict:
    data = json.dumps({**body, "model": MODEL}).encode()
    req  = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test_plain_chat():
    print(f"\n{YELLOW}── Test 1: plain conversational response ──{RESET}")
    resp = _post({
        "messages": [{"role": "user", "content": "Say exactly: hello world"}],
        "max_tokens": 32,
    })
    content = resp["choices"][0]["message"]["content"]
    print(f"Response: {content!r}")
    assert "hello" in content.lower(), f"Expected 'hello', got: {content!r}"
    print(f"{GREEN}PASS{RESET}")


def test_reasoning_not_in_content():
    print(f"\n{YELLOW}── Test 2: <think> not leaked into content (TTS safety) ──{RESET}")
    resp = _post({
        "messages": [{"role": "user", "content": "What is 17 * 23? Think step by step."}],
        "max_tokens": 512,
    })
    msg       = resp["choices"][0]["message"]
    content   = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    print(f"content:          {content!r}")
    print(f"reasoning_content:{reasoning[:80]!r}{'…' if len(reasoning) > 80 else ''}")
    assert "<think>" not in content, \
        f"<think> block leaked into content — TTS would read it aloud!\n{content!r}"
    assert "391" in content or "391" in reasoning, \
        f"Expected 391 (17×23), got: {content!r}"
    print(f"{GREEN}PASS{RESET}")


def test_tool_call():
    print(f"\n{YELLOW}── Test 3: tool calling ──{RESET}")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_scene_description",
            "description": "Describe what the XR user is currently looking at",
            "parameters": {
                "type": "object",
                "properties": {
                    "detail_level": {
                        "type": "string",
                        "enum": ["brief", "detailed"],
                    },
                },
                "required": ["detail_level"],
            },
        },
    }]
    resp = _post({
        "messages": [{"role": "user",
                      "content": "Can you describe what I'm looking at in detail?"}],
        "tools":    tools,
        "max_tokens": 128,
    })
    msg    = resp["choices"][0]["message"]
    reason = resp["choices"][0]["finish_reason"]
    print(f"finish_reason: {reason}")
    if reason == "tool_calls" and msg.get("tool_calls"):
        call = msg["tool_calls"][0]["function"]
        args = json.loads(call["arguments"])
        print(f"{GREEN}PASS{RESET} — called {call['name']}({args})")
    else:
        print(f"{YELLOW}WARN{RESET} — answered conversationally: {msg.get('content','')!r}")
        print("  (acceptable; model may not always choose to call the tool)")


def test_multi_turn():
    print(f"\n{YELLOW}── Test 4: multi-turn memory ──{RESET}")
    resp = _post({
        "messages": [
            {"role": "user",      "content": "My name is Alex."},
            {"role": "assistant", "content": "Nice to meet you, Alex!"},
            {"role": "user",      "content": "What's my name?"},
        ],
        "max_tokens": 32,
    })
    content = resp["choices"][0]["message"]["content"]
    print(f"Response: {content!r}")
    assert "alex" in content.lower(), \
        f"Expected model to recall 'Alex', got: {content!r}"
    print(f"{GREEN}PASS{RESET}")


def main():
    if not PROJECT.exists():
        print(f"{RED}Service not found at {PROJECT}{RESET}")
        print(f"Check REPO env var (currently: {REPO})")
        sys.exit(1)

    start_service()

    if not wait_ready():
        print(f"{RED}Service did not become ready in {TIMEOUT}s.{RESET}")
        sys.exit(1)

    test_plain_chat()
    test_reasoning_not_in_content()
    test_tool_call()
    test_multi_turn()

    print(f"\n{GREEN}All tests complete.{RESET}")


if __name__ == "__main__":
    main()
