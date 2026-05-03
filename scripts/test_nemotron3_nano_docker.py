#!/usr/bin/env python3
"""
Manual test: nemotron3_nano_docker_llm_server (Docker Model Runner).

Launches the service via `uv run`, waits for it to be ready, runs tests,
then stops it.  No manual `docker model run` needed.

Usage:
    python3 scripts/test_nemotron3_nano_docker.py

    # Override repo root if needed:
    REPO=/path/to/xr-ai python3 ...

    # Use the small 9B model (faster for quick testing):
    USE_SMALL=1 python3 ...
"""
import atexit
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(os.environ.get("REPO", str(Path(__file__).resolve().parents[1])))
PROJECT  = REPO / "ai-services/llm/nemotron3_nano_docker"
COMMAND  = "nemotron3_nano_docker_llm_server"
CONFIG   = PROJECT / f"{COMMAND}.yaml"

BASE_URL = "http://localhost:12434/engines/llama.cpp/v1"
TIMEOUT  = int(os.environ.get("TIMEOUT", "300"))

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

_proc: subprocess.Popen | None = None


def _stop():
    if _proc and _proc.poll() is None:
        print("\n[teardown] Stopping service…")
        _proc.terminate()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()

atexit.register(_stop)



def _check_docker_model_runner():
    """Fail fast with install instructions if the plugin is missing."""
    import shutil
    if not shutil.which("docker"):
        print(f"{RED}docker not found on PATH.{RESET}")
        sys.exit(1)
    result = subprocess.run(["docker", "model", "version"], capture_output=True)
    if result.returncode != 0:
        print(f"{RED}Docker Model Runner plugin not installed.{RESET}")
        print()
        print("Install on Linux:")
        print("  sudo apt-get install docker-model-plugin")
        print()
        print("Official guide: https://docs.docker.com/model-runner/")
        sys.exit(1)


def start_service():
    _check_docker_model_runner()
    global _proc
    # Override use_small via a temp config if requested.
    if os.environ.get("USE_SMALL"):
        import yaml
        base = yaml.safe_load(CONFIG.read_text()) if CONFIG.exists() else {}
        base["use_small"] = True
        tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
        yaml.dump(base, tmp)
        tmp.flush()
        cfg_path = tmp.name
    else:
        cfg_path = str(CONFIG)

    cmd = ["uv", "run", "--project", str(PROJECT), COMMAND, "--config", cfg_path]
    print(f"[start] {' '.join(cmd)}")
    _proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def wait_ready() -> bool:
    print(f"Waiting for Docker Model Runner at {BASE_URL} (up to {TIMEOUT}s)…")
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        if _proc and _proc.poll() is not None:
            print(f"{RED}Service process exited early (rc={_proc.returncode}){RESET}")
            return False
        try:
            urllib.request.urlopen(f"{BASE_URL}/models", timeout=3)
            print(f"{GREEN}Ready.{RESET}")
            return True
        except Exception:
            time.sleep(3)
    return False


def _post(body: dict, model: str) -> dict:
    data = json.dumps({**body, "model": model}).encode()
    req  = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def get_loaded_model() -> str:
    """Return the first loaded model name from the runner."""
    with urllib.request.urlopen(f"{BASE_URL}/models", timeout=5) as r:
        data = json.loads(r.read())
    models = [m["id"] for m in data.get("data", [])]
    if not models:
        raise RuntimeError("No models loaded in Docker Model Runner")
    return models[0]


def test_plain_chat(model: str):
    print(f"\n{YELLOW}── Test 1: plain chat ──{RESET}")
    resp = _post({
        "messages": [{"role": "user", "content": "Say exactly: hello world"}],
        "max_tokens": 32,
    }, model)
    content = resp["choices"][0]["message"]["content"]
    print(f"Response: {content!r}")
    assert "hello" in content.lower(), f"Expected 'hello', got: {content!r}"
    print(f"{GREEN}PASS{RESET}")


def test_tool_call(model: str):
    print(f"\n{YELLOW}── Test 2: tool calling ──{RESET}")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit":     {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }]
    resp = _post({
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
        "tools":    tools,
        "max_tokens": 128,
    }, model)
    msg    = resp["choices"][0]["message"]
    reason = resp["choices"][0]["finish_reason"]
    print(f"finish_reason: {reason}")
    if reason == "tool_calls" and msg.get("tool_calls"):
        call = msg["tool_calls"][0]["function"]
        args = json.loads(call["arguments"])
        print(f"{GREEN}PASS{RESET} — called {call['name']}({args})")
    else:
        content = msg.get("content", "")
        print(f"{YELLOW}WARN{RESET} — answered conversationally: {content!r}")
        print("  (acceptable; small models sometimes answer directly)")


def test_reasoning(model: str):
    print(f"\n{YELLOW}── Test 3: arithmetic (checks correctness) ──{RESET}")
    resp = _post({
        "messages": [{"role": "user", "content": "What is 17 * 23? Just the number."}],
        "max_tokens": 32,
    }, model)
    content = resp["choices"][0]["message"].get("content", "")
    reasoning = resp["choices"][0]["message"].get("reasoning_content", "")
    print(f"content:   {content!r}")
    print(f"reasoning: {reasoning[:80]!r}")
    assert "391" in content or "391" in reasoning, \
        f"Expected 391 in response, got: {content!r}"
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

    model = get_loaded_model()
    print(f"Testing model: {model}\n")

    test_plain_chat(model)
    test_tool_call(model)
    test_reasoning(model)

    print(f"\n{GREEN}All tests complete.{RESET}")


if __name__ == "__main__":
    main()
