# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import ssl
import stat
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import HTTPRedirectHandler, build_opener

if __package__:
    from . import recordings
else:
    import recordings

app: str
page: bytes
state: Path
ui_auth: str
expected_auth: str


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


proxy = build_opener(NoRedirect)
inventory_lock = threading.Lock()
inventory_cache = (None, 0.0, {})


def redact(text):
    for name in ("hf", "ui.password"):
        try:
            secret = (state / name).read_text().strip()
        except OSError:
            continue
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", text)


def read_first(name):
    try:
        return (state / name).read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        return ""


def alive(raw):
    try:
        pid = int(raw)
        os.kill(pid, 0)
        return True
    except PermissionError:
        return Path(f"/proc/{raw}").exists()
    except (ValueError, OSError):
        return False


def sources():
    values = {"combined": ("Combined deployment", state / "sample.log")}
    run_id = read_first("sample.run")
    if run_id:
        for path in sorted((state / "logs").glob(f"log_*_{run_id}/*.log")):
            relative = path.relative_to(state / "logs").as_posix()
            namespace = path.parent.name.removeprefix("log_").removesuffix("_" + run_id)
            values[relative] = (f"{namespace} · {path.stem}", path)
    return values


def sample_ready():
    # Read from the beginning, not the UI log tail: the first poll may arrive
    # long after startup. Keep memory bounded even for logs without newlines.
    marker = b"XR_AI_SAMPLE_START"
    pending = b""
    try:
        with (state / "sample.log").open("rb") as stream:
            while chunk := stream.read(65536):
                pending += chunk
                if marker in pending:
                    if marker == b"All processes ready":
                        return True
                    pending = pending.partition(marker)[2]
                    marker = b"All processes ready"
                    if marker in pending:
                        return True
                pending = pending[-(len(marker) - 1):]
    except OSError:
        pass
    return False


def process_status():
    pid = read_first("sample.pid")
    current = read_first("sample.current")
    running = alive(pid)
    run_id = read_first("sample.run")
    latched = bool(run_id) and read_first("sample.ready") == run_id
    structured = read_first("sample.status-mode") == "structured"
    # Only old releases need the log adapter. Current stacks own their ready file.
    milestones = latched or (not structured and running and sample_ready())
    if running and milestones and not latched and run_id and not structured:
        try:
            (state / "sample.ready").write_text(run_id + "\n")
        except OSError:
            pass
    ready = running and milestones
    if ready:
        status = "ready"
    elif running:
        status = "starting"
    elif pid and read_first("sample.exit") not in {"", "0"}:
        status = "failed"
    else:
        status = "stopped"
    return {
        "sample": current, "state": status, "anonymous": not (state / "hf").is_file(),
        "recording": bool(read_first("sample.recording")),
    }


def inventory():
    global inventory_cache
    revision = read_first("runtime-root")
    with inventory_lock:
        key, updated, cached = inventory_cache
        if key != revision or time.monotonic() - updated >= 30:
            result = subprocess.run([app, "catalog"], text=True, capture_output=True, timeout=20, env=os.environ)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or "catalog failed")
            cached = json.loads(result.stdout)
            inventory_cache = (revision, time.monotonic(), cached)
        value = dict(cached)
    value["public_host"] = read_first("public-ip")
    value["revision"] = read_first("runtime-ref")
    try:
        der = ssl.PEM_cert_to_DER_cert((Path.home() / ".local/share/xr-ai/root-ca.crt").read_text())
        value["root_ca_sha256"] = ":".join(re.findall("..", hashlib.sha256(der).hexdigest().upper()))
    except (OSError, ValueError):
        value["root_ca_sha256"] = ""
    return value


class Handler(BaseHTTPRequestHandler):
    def authenticated(self):
        # Proxy mode delegates identity to the explicitly trusted upstream;
        # forwarding headers are never treated as proof of authentication.
        if ui_auth == "proxy":
            return True
        if hmac.compare_digest(self.headers.get("Authorization", ""), expected_auth):
            return True
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", 'Basic realm="XR AI Experiences"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def send_bytes(self, payload, status=HTTPStatus.OK, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, value, status=HTTPStatus.OK):
        self.send_bytes(redact(json.dumps(value)).encode(), status)

    def do_GET(self):
        if not self.authenticated():
            return
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                return self.send_bytes(page, content_type="text/html; charset=utf-8")
            if parsed.path == "/api/status":
                return self.send_json({"inventory": inventory(), "process": process_status()})
            if parsed.path == "/api/releases":
                result = subprocess.run([app, "releases"], capture_output=True, text=True, timeout=15)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip() or "release discovery failed")
                return self.send_json(json.loads(result.stdout))
            if parsed.path == "/api/recordings":
                return self.send_json({"recordings": recordings.recordings()})
            if parsed.path == "/api/recording":
                try:
                    paths = recordings.files(query.get("id", [""])[0])
                except (OSError, ValueError):
                    return self.send_json({"error": "finalized recording not found"}, HTTPStatus.NOT_FOUND)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", 'attachment; filename="xr-ai-session.zip"')
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    recordings.write_zip(self.wfile, paths)
                except (OSError, ValueError):
                    self.log_message("recording download interrupted")
                self.close_connection = True
                return
            if parsed.path == "/api/logs":
                return self.send_json({"sources": [{"id": key, "name": name} for key, (name, _) in sources().items()]})
            if parsed.path == "/api/log":
                source_id = query.get("id", ["combined"])[0]
                allowed = sources()
                if source_id not in allowed:
                    return self.send_json({"error": "unknown log source"}, HTTPStatus.NOT_FOUND)
                path = allowed[source_id][1]
                size = path.stat().st_size if path.exists() else 0
                raw_offset = query.get("offset", [""])[0]
                offset = int(raw_offset) if raw_offset.isdigit() else max(0, size - 80000)
                if offset > size:
                    offset = 0
                with path.open("rb") if path.exists() else open(os.devnull, "rb") as stream:
                    stream.seek(offset)
                    data = stream.read(65536)
                    if data and not data.endswith(b"\n"):
                        data += stream.readline(8192)
                    next_offset = stream.tell()
                return self.send_json({"text": data.decode(errors="replace"), "offset": next_offset, "size": size})
            if parsed.path not in {"/events", "/e", "/api/events", "/app.js", "/style.css"}:
                return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            path = {"/events": "/", "/e": "/healthz"}.get(parsed.path, parsed.path) + (
                "?" + parsed.query if parsed.query else ""
            )
            with proxy.open("http://127.0.0.1:8092" + path, timeout=1) as response:
                self.send_bytes(
                    redact(response.read(1048576).decode()).encode(), content_type=response.headers["Content-Type"]
                )
        except (OSError, ValueError, RuntimeError, HTTPException, subprocess.SubprocessError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        if not self.authenticated():
            return
        if self.headers.get("X-XR-AI-Request") != "1":
            return self.send_json({"error": "request header required"}, HTTPStatus.FORBIDDEN)
        if self.headers.get("Origin") and urlparse(self.headers["Origin"]).netloc != self.headers.get("Host"):
            return self.send_json({"error": "cross-origin request rejected"}, HTTPStatus.FORBIDDEN)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 4096 or self.headers.get("Transfer-Encoding"):
                return self.send_json({"error": "invalid request size"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            form = parse_qs(self.rfile.read(length).decode(), keep_blank_values=True)
            path = urlparse(self.path).path
            if path == "/api/start":
                sample = form.get("sample", [""])[0]
                if not re.fullmatch(r"[a-z0-9-]+", sample):
                    raise ValueError("invalid demo ID")
                env = os.environ | {"HF_TOKEN": form.get("h", [""])[0]}
                recording = form.get("recording", [])
                if recording not in ([], ["1"]):
                    raise ValueError("recording must be omitted or set once to 1")
                command = [app, "start", sample] + (["--session-recording"] if recording else [])
                result = subprocess.run(command, text=True, capture_output=True, timeout=120, env=env)
            elif path == "/api/commit":
                revision = form.get("c", [""])[0]
                result = subprocess.run(
                    [app, "commit", revision], text=True, capture_output=True, timeout=120, env=os.environ
                )
            elif path == "/api/stop":
                forget = form.get("forget", [])
                if forget not in ([], ["1"]):
                    raise ValueError("forget must be omitted or set once to 1")
                command = [app, "forget-token"] if forget else [app, "stop"]
                result = subprocess.run(
                    command, text=True, capture_output=True, timeout=90, env=os.environ
                )
            else:
                return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            message = (result.stdout or result.stderr).strip()
            self.send_json(
                {"message": message or "done"}
                if result.returncode == 0
                else {"error": message or "launcher command failed"},
                HTTPStatus.OK if result.returncode == 0 else HTTPStatus.CONFLICT,
            )
        except subprocess.TimeoutExpired:
            self.send_json({"error": "launcher command timed out"}, HTTPStatus.GATEWAY_TIMEOUT)
        except (OSError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt, *args):
        sys.stderr.write("ui: " + fmt % args + "\n")


def create_server(app_path, page_path, state_path, port):
    # One chooser per process; importing helpers must not bind a port or create credentials.
    global app, page, state, ui_auth, expected_auth, inventory_cache
    app, page, state = str(app_path), Path(page_path).read_bytes(), Path(state_path)
    inventory_cache = (None, 0.0, {})
    bind = os.environ.get("XR_AI_UI_BIND") or "127.0.0.1"
    ui_auth = os.environ.get("XR_AI_UI_AUTH") or "password"
    if ui_auth not in {"password", "proxy"}:
        raise SystemExit("XR_AI_UI_AUTH must be password or proxy")
    if ui_auth == "proxy" and os.environ.get("XR_AI_TRUST_PROXY") != "1":
        raise SystemExit("proxy authentication requires XR_AI_TRUST_PROXY=1 and an authenticated HTTPS proxy")
    if bind not in {"127.0.0.1", "::1"} and os.environ.get("XR_AI_TRUST_PROXY") != "1":
        raise SystemExit("non-loopback binding requires XR_AI_TRUST_PROXY=1 and a firewall-protected HTTPS proxy")
    expected_auth = ""
    if ui_auth == "password":
        password_path = state / "ui.password"
        try:
            fd = os.open(password_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w") as stream:
                stream.write(secrets.token_urlsafe(32) + "\n")
        info = password_path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise SystemExit("ui.password must be a private regular file owned by this user")
        password = password_path.read_text().strip()
        if len(password) < 32:
            raise SystemExit("ui.password is too short; remove it to generate a new password")
        expected_auth = "Basic " + base64.b64encode(("xrai:" + password).encode()).decode()

    class UIHTTPServer(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in bind else socket.AF_INET

    return UIHTTPServer((bind, port), Handler)


if __name__ == "__main__":
    with create_server(*sys.argv[1:4], int(sys.argv[4])) as server:
        (state / "ui.pid").write_text(str(os.getpid()))
        bind, port = server.server_address[:2]
        host = f"[{bind}]" if server.address_family == socket.AF_INET6 else bind
        print(f"XR AI Experiences listening on http://{host}:{port}", flush=True)
        server.serve_forever()
