#!/usr/bin/env python3
"""
Manual test: vlm_server_docker (Cosmos-Reason1-7B via Docker Model Runner).

Launches the service via `uv run`, waits for it to be ready, runs tests,
then stops it.  No manual `docker model run` needed.

Usage:
    python3 scripts/test_vlm_server_docker.py

    # Override repo root if needed:
    REPO=/path/to/xr-ai python3 ...
"""
import atexit
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(os.environ.get("REPO", str(Path(__file__).resolve().parents[1])))
PROJECT = REPO / "ai-services/vlm-server-docker"
COMMAND = "vlm_server_docker"
CONFIG  = PROJECT / f"{COMMAND}.yaml"

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
    cmd = ["uv", "run", "--project", str(PROJECT), COMMAND, "--config", str(CONFIG)]
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
    with urllib.request.urlopen(f"{BASE_URL}/models", timeout=5) as r:
        data = json.loads(r.read())
    models = [m["id"] for m in data.get("data", [])]
    if not models:
        raise RuntimeError("No models loaded in Docker Model Runner")
    return models[0]


def _solid_color_jpeg(r: int, g: int, b: int) -> str:
    """Tiny solid-color image as a JPEG data URL. Uses PIL if available."""
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), (r, g, b))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Minimal valid JPEG fallback (ignores color, but gets us a valid image)
        raw = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
            0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
            0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
            0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
            0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
            0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
            0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F,
            0x00, 0xF5, 0x7F, 0xFF, 0xD9,
        ])
        b64 = base64.b64encode(raw).decode()
    return f"data:image/jpeg;base64,{b64}"


def test_text_only(model: str):
    print(f"\n{YELLOW}── Test 1: text-only prompt ──{RESET}")
    resp = _post({
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
        "max_tokens": 16,
    }, model)
    content = resp["choices"][0]["message"]["content"].strip()
    print(f"Response: {content!r}")
    assert "4" in content, f"Expected '4', got: {content!r}"
    print(f"{GREEN}PASS{RESET}")


def test_image_color(model: str):
    print(f"\n{YELLOW}── Test 2: image description (solid red 64×64) ──{RESET}")
    data_url = _solid_color_jpeg(255, 0, 0)
    resp = _post({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "What color is this image? One word."},
            ],
        }],
        "max_tokens": 16,
    }, model)
    content = resp["choices"][0]["message"]["content"].strip()
    print(f"Response: {content!r}")
    assert "red" in content.lower(), f"Expected 'red', got: {content!r}"
    print(f"{GREEN}PASS{RESET}")


def test_image_reasoning(model: str):
    print(f"\n{YELLOW}── Test 3: image + reasoning (blue 64×64) ──{RESET}")
    data_url = _solid_color_jpeg(0, 0, 200)
    resp = _post({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text",
                 "text": "What primary color do you see? Think before answering."},
            ],
        }],
        "max_tokens": 256,
    }, model)
    msg       = resp["choices"][0]["message"]
    content   = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    print(f"content:   {content!r}")
    print(f"reasoning: {reasoning[:100]!r}{'…' if len(reasoning) > 100 else ''}")
    assert "blue" in content.lower() or "blue" in reasoning.lower(), \
        f"Expected 'blue', got: {content!r}"
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

    test_text_only(model)
    test_image_color(model)
    test_image_reasoning(model)

    print(f"\n{GREEN}All tests complete.{RESET}")


if __name__ == "__main__":
    main()
