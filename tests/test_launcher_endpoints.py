# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import threading
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import xr_ai_launcher._endpoints as _endpoints


def _check(tmp_path, endpoint: dict):
    profile = tmp_path / "models.json"
    profile.write_text(json.dumps({"model": endpoint}), encoding="utf-8")
    return _endpoints.endpoint_checks(profile)[0]


@contextmanager
def _server(
    hits,
    *,
    redirect: str | None = None,
    statuses: dict[str, int] | None = None,
):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append((self.path, self.headers.get("Authorization")))
            if self.path == "/broken/health":
                self.request.sendall(b"garbage-not-http\r\n")
                self.close_connection = True
                return
            moved = redirect and self.path == "/moved/health"
            status = 302 if moved else (statuses or {}).get(self.path, 204)
            self.send_response(status)
            if moved:
                self.send_header("Location", redirect)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.05), daemon=True,
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_health_probe_is_direct_bounded_and_does_not_follow_redirects(tmp_path, monkeypatch) -> None:
    source_hits, proxy_hits = [], []
    with _server(proxy_hits) as proxy:
        proxy_url = f"http://127.0.0.1:{proxy.server_port}"
        statuses = {"/missing/health": 404, "/starting": 503}
        with _server(source_hits, redirect=proxy_url + "/stolen", statuses=statuses) as source:
            for name in ("http_proxy", "HTTP_PROXY"):
                monkeypatch.setenv(name, proxy_url)
            for name in ("no_proxy", "NO_PROXY"):
                monkeypatch.delenv(name, raising=False)
            monkeypatch.setenv("MODEL_TOKEN", "secret-token")
            importlib.reload(_endpoints)

            base = f"http://127.0.0.1:{source.server_port}"
            authenticated = {"api_key_env": "MODEL_TOKEN", "readiness": "health"}
            healthy = _check(tmp_path, authenticated | {"base_url": base})
            redirect = _check(tmp_path, authenticated | {"base_url": base + "/moved"})
            unverified = _check(tmp_path, {"base_url": base + "/missing"})
            implicit_404 = _check(
                tmp_path,
                {"base_url": base + "/missing", "health_check": True},
            )
            explicit_503 = _check(
                tmp_path,
                {"base_url": base, "health_path": "/starting"},
            )
            disabled = _check(
                tmp_path,
                {"base_url": base + "/disabled", "health_check": False},
            )
            broken = _check(tmp_path, authenticated | {"base_url": base + "/broken"})

    refused = _check(tmp_path, {"base_url": base})

    assert healthy["status"] == "passed"
    assert redirect["status"] == "failed" and redirect["detected"].startswith("HTTP 302")
    assert unverified["status"] == "skipped" and "health unverified" in unverified["detected"]
    assert implicit_404["status"] == "failed" and implicit_404["detected"].startswith("HTTP 404")
    assert explicit_503["status"] == "failed" and explicit_503["detected"].startswith("HTTP 503")
    assert disabled["status"] == "skipped"
    assert broken["status"] == "failed" and "BadStatusLine" in broken["detected"]
    assert refused["status"] == "failed" and "ConnectionRefusedError" in refused["detected"]
    assert refused["remediation"] == _endpoints._START_MODELS
    assert source_hits == [
        ("/health", "Bearer secret-token"),
        ("/moved/health", "Bearer secret-token"),
        ("/missing/health", None),
        ("/starting", None),
        ("/broken/health", "Bearer secret-token"),
    ]
    assert proxy_hits == []
    assert "secret-token" not in json.dumps(
        [healthy, redirect, unverified, implicit_404, explicit_503, broken, refused]
    )


def test_profile_caches_probes_and_reports_every_role_and_skip(tmp_path, monkeypatch) -> None:
    endpoint = {
        "base_url": "https://models.example.test/v1",
        "api_key_env": "MODEL_TOKEN",
        "readiness": "health",
    }
    models = {
        "llm": {"adapter": {"preset": "nemotron_omni"}, "endpoint": endpoint},
        "agent_llm": {
            "adapter": {"preset": "nemotron_omni"},
            "endpoint": endpoint,
        },
        "disabled": {
            "endpoint": {
                "base_url": "https://models.example.test/v1",
                "health_check": False,
            }
        },
        "managed": {
            "endpoint": {"base_url": "http://localhost:8100"},
            "deployment": {"ownership": "managed", "service": "llm"},
        },
        "managed_auth": {
            "endpoint": {
                "base_url": "http://localhost:8101",
                "api_key_env": "MANAGED_TOKEN",
            },
            "deployment": {"ownership": "managed", "service": "other"},
        },
        "hosted": {
            "endpoint": {"base_url": "https://hosted.example"},
        },
        "speech": {
            "adapter": {"kind": "riva_grpc"},
            "endpoint": {"base_url": "grpc.nvcf.example:443"},
        },
    }
    profile = tmp_path / "models.json"
    profile.write_text(json.dumps({"models": models}), encoding="utf-8")
    requests = []

    def open_request(request, *, timeout):
        requests.append((request.full_url, request.get_header("Authorization"), timeout))
        if len(requests) <= 2:
            raise _endpoints.HTTPError(request.full_url, 401, "unauthorized", {}, None)
        return nullcontext(SimpleNamespace(status=204))

    monkeypatch.setenv("MODEL_TOKEN", "secret-token")
    monkeypatch.setattr(_endpoints._OPENER, "open", open_request)
    rows = _endpoints.endpoint_checks(profile, {"llm"})
    remote_failure = _check(tmp_path, endpoint)

    assert _endpoints.endpoint_checks(None) == []
    assert requests == 2 * [("https://models.example.test/v1/health", "Bearer secret-token", 3.0)]
    assert [(row["name"], row["status"], row["detected"]) for row in rows] == [
        ("endpoint:llm", "failed", "HTTP 401 from https://models.example.test/v1/health"),
        ("endpoint:agent_llm", "failed", "HTTP 401 from https://models.example.test/v1/health"),
        ("endpoint:disabled", "skipped", "not checked: health probing is disabled by the profile"),
        ("endpoint:managed", "skipped", "not checked before the configured service is launched"),
        ("endpoint:managed_auth", "failed", "credential MANAGED_TOKEN is not set"),
        ("endpoint:hosted", "skipped", "not checked: remote endpoint has no configured health route"),
        ("endpoint:speech", "skipped", "not checked: launcher does not probe gRPC endpoints"),
    ]
    assert rows[0]["remediation"] == f"Fix the configured endpoint for llm in {profile}."
    assert rows[1]["remediation"] == f"Fix the configured endpoint for agent_llm in {profile}."
    assert remote_failure["status"] == "failed"
    assert remote_failure["remediation"] == (
        f"Fix the configured endpoint for model in {tmp_path / 'models.json'}."
    )
    assert "secret-token" not in json.dumps([rows, remote_failure])
    assert all("ok" not in row for row in [*rows, remote_failure])

def test_credentials_profile_errors_and_unsafe_values_are_redacted(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    monkeypatch.setattr(_endpoints._OPENER, "open", lambda *_args, **_kwargs: 1 / 0)
    for policy in ({}, {"readiness": "health"}, {"health_check": False}):
        endpoint = {
            "base_url": "https://models.example",
            "api_key_env": "MODEL_TOKEN",
            **policy,
        }
        row = _check(tmp_path, endpoint)
        assert row["status"] == "failed"
        assert row["detected"] == "credential MODEL_TOKEN is not set"

    for name, contents in (("broken.json", "not json"), ("missing.json", None)):
        path = tmp_path / name
        if contents:
            path.write_text(contents, encoding="utf-8")
        row = _endpoints.endpoint_checks(path)[0]
        assert row["status"] == "failed"
        assert row["detected"] == "missing or invalid model profile"

    for field, value in (
        ("base_url", []),
        ("health_path", []),
        ("kind", []),
        ("readiness", []),
        ("health_check", "false"),
    ):
        model = {
            "base_url": "https://models.example",
            "health_path": "/health",
            "kind": "openai_compat",
            "readiness": "health",
        }
        model[field] = value
        profile = tmp_path / "bad-field.json"
        profile.write_text(json.dumps({"model": model}), encoding="utf-8")
        row = _endpoints.endpoint_checks(profile)[0]
        assert row["status"] == "failed"
        assert row["detected"] == "missing or invalid model profile"

    conflict = _check(
        tmp_path,
        {
            "base_url": "https://models.example",
            "readiness": "health",
            "health_check": False,
        },
    )
    assert conflict["status"] == "failed"

    yaml_profile = tmp_path / "models.yaml"
    yaml_profile.write_text("models: {}\n", encoding="utf-8")
    skipped = _endpoints.endpoint_checks(yaml_profile)[0]
    assert skipped["status"] == "skipped"

    invalid_fix = f"Fix base_url or health_path for model in {tmp_path / 'models.json'}."
    for endpoint in (
        {"base_url": "http://[broken", "readiness": "health"},
        {
            "base_url": "https://user:must-not-appear@models.example",
            "readiness": "health",
        },
        {
            "base_url": "https://models.example?key=must-not-appear",
            "readiness": "health",
        },
        {
            "base_url": "https://models.example",
            "health_path": "/health?token=must-not-appear",
        },
    ):
        row = _check(tmp_path, endpoint)
        assert row["status"] == "failed"
        assert row["detected"] == "invalid HTTP endpoint URL or health path"
        assert row["remediation"] == invalid_fix
        assert "must-not-appear" not in json.dumps(row)
