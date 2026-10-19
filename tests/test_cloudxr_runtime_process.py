# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CloudXR runtime service: SDK fetch, environment, EULA, process, and WSS proxy."""
from __future__ import annotations

import asyncio
import hashlib
import io
import os
import ssl
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import websockets
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

_PROJECT = Path(__file__).resolve().parents[1] / "services" / "cloudxr-runtime"
sys.path.insert(0, str(_PROJECT))

from cloudxr_runtime import _runtime, _sdk, _wss  # noqa: E402
from cloudxr_runtime._runtime import Layout  # noqa: E402

_SDK_FILES = {
    "./libcloudxr.so": b"cloudxr",
    "./libopenxr_cloudxr.so": b"openxr",
    "./openxr_cloudxr.json": b"{}",
    "./LICENSE.txt": b"NVIDIA SOFTWARE LICENSE AGREEMENT",
    "./include/cxrServiceAPI.h": b"header",
    "./plugins/libplugin.so": b"plugin",
}


def _write_tarball(path: Path) -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in _SDK_FILES.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


@pytest.fixture
def placed_tarball(tmp_path, monkeypatch):
    """An SDK tarball placed for offline install, with its digest pinned."""
    monkeypatch.setattr(_sdk.platform, "machine", lambda: "x86_64")
    tarball = tmp_path / "sdk" / f"CloudXR-{_sdk.SDK_VERSION}-Linux-amd64-sdk.tar.gz"
    monkeypatch.setitem(_sdk._SHA256, "amd64", _write_tarball(tarball))
    monkeypatch.setattr(_sdk, "_download", lambda *_: pytest.fail("downloaded despite a placed tarball"))
    return tarball


def test_sdk_is_unpacked_without_headers_or_plugins(tmp_path, placed_tarball):
    sdk = _sdk.ensure_sdk(tmp_path)

    names = sorted(path.name for path in sdk.native_dir.iterdir())
    assert names == ["LICENSE.txt", "libcloudxr.so", "libopenxr_cloudxr.so", "openxr_cloudxr.json"]
    assert sdk.license_file.read_bytes() == _SDK_FILES["./LICENSE.txt"]
    assert not placed_tarball.exists()
    # A second call reuses the unpacked SDK.
    assert _sdk.ensure_sdk(tmp_path) == sdk


def test_sdk_with_wrong_digest_is_rejected_and_removed(tmp_path, placed_tarball, monkeypatch):
    monkeypatch.setitem(_sdk._SHA256, "amd64", "0" * 64)

    with pytest.raises(_sdk.SDKError, match="expected 0000"):
        _sdk.ensure_sdk(tmp_path)

    assert not placed_tarball.exists()
    assert not (tmp_path / "sdk" / f"{_sdk.SDK_VERSION}-amd64").exists()


def test_sdk_download_failure_explains_offline_install(tmp_path, monkeypatch):
    monkeypatch.setattr(_sdk.platform, "machine", lambda: "aarch64")

    def refuse(*_args, **_kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(_sdk.urllib.request, "urlopen", refuse)

    with pytest.raises(_sdk.SDKError, match="place the file at .*arm64-sdk.tar.gz"):
        _sdk.ensure_sdk(tmp_path)


def test_sdk_pins_cover_every_supported_architecture():
    assert set(_sdk._SHA256) == set(_sdk._ARCH.values())


def test_web_client_sdk_matches_the_runtime_version():
    version_file = Path(__file__).resolve().parents[1] / "client-samples" / "web-xr-build" / ".sdk-version"
    assert version_file.read_text().strip() == _sdk.SDK_VERSION


def test_published_env_has_derived_paths_and_respects_overrides(tmp_path, monkeypatch):
    layout = Layout(tmp_path)
    env_file = tmp_path / "overrides.env"
    env_file.write_text("NV_DEVICE_PROFILE=auto-native\nXR_RUNTIME_JSON=/elsewhere.json\n")
    monkeypatch.setenv("NV_CXR_FILE_LOGGING", "false")
    for key in ("XR_RUNTIME_JSON", "NV_CXR_RUNTIME_DIR", "NV_CXR_OUTPUT_DIR", "XRT_NO_STDIN",
                "CXR_INSTALL_DIR", "CXR_HOST_VOLUME_PATH", "NV_DEVICE_PROFILE"):
        monkeypatch.delenv(key, raising=False)

    env = _runtime.publish_env(layout, env_file)

    assert env["NV_DEVICE_PROFILE"] == "auto-native"  # env file wins
    assert env["NV_CXR_FILE_LOGGING"] == "false"  # process env beats default
    assert env["NV_CXR_ENABLE_PUSH_DEVICES"] == "true"  # default
    assert env["XR_RUNTIME_JSON"] == str(tmp_path / "openxr_cloudxr.json")  # derived, not overridable
    assert os.environ["NV_CXR_RUNTIME_DIR"] == str(layout.run_dir)
    written = layout.env_file.read_text()
    assert f"export XR_RUNTIME_JSON={tmp_path / 'openxr_cloudxr.json'}\n" in written
    assert layout.env_file.stat().st_mode & 0o777 == 0o600


def test_eula_acceptance_is_recorded_once(tmp_path, monkeypatch):
    layout = Layout(tmp_path)
    sdk = _sdk.SDK(tmp_path)

    _runtime.check_eula(layout, sdk, accept=True)

    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("prompted after acceptance"))
    _runtime.check_eula(layout, sdk, accept=False)
    assert layout.eula_marker.read_text() == "accepted\n"


def test_declined_eula_exits_without_a_marker(tmp_path, monkeypatch):
    layout = Layout(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    with pytest.raises(SystemExit):
        _runtime.check_eula(layout, _sdk.SDK(tmp_path), accept=False)

    assert not layout.eula_marker.exists()


def test_run_dir_gets_the_runtime_manifest_and_loses_stale_state(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    for name in ("libopenxr_cloudxr.so", "openxr_cloudxr.json"):
        (native / name).write_text(name)
    layout = Layout(tmp_path / "install")
    layout.run_dir.mkdir(parents=True)
    (layout.run_dir / "runtime_started").touch()
    (layout.run_dir / "eula_accepted").touch()

    _runtime.prepare_run_dir(layout, _sdk.SDK(native))

    assert layout.runtime_json.read_text() == "openxr_cloudxr.json"
    assert not layout.ready_sentinel.exists()
    assert layout.eula_marker.exists()


def test_service_process_never_loads_python_openssl():
    # libcloudxr.so links the system libssl; a Python OpenSSL in the same
    # process conflicts with it (see cloudxr_runtime._service).
    result = subprocess.run(
        [sys.executable, "-c", "import sys, cloudxr_runtime._service; print('ssl' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(_PROJECT)},
    )
    assert result.stdout.strip() == "False"


@pytest.mark.asyncio
async def test_wss_proxy_forwards_websockets_and_serves_the_certificate_page(tmp_path, unused_tcp_port_factory):
    backend_port, proxy_port = unused_tcp_port_factory(), unused_tcp_port_factory()
    seen_paths = []

    async def echo(connection):
        seen_paths.append(connection.request.path)
        async for message in connection:
            await connection.send(message)

    client_tls = ssl.create_default_context()
    client_tls.check_hostname = False
    client_tls.verify_mode = ssl.CERT_NONE
    stop = asyncio.get_running_loop().create_future()
    async with serve(echo, "127.0.0.1", backend_port):
        proxy = asyncio.create_task(
            _wss.run(tmp_path / "certs", stop, port=proxy_port, backend_uri=f"ws://127.0.0.1:{backend_port}")
        )
        try:
            for _ in range(50):
                try:
                    async with connect(f"wss://127.0.0.1:{proxy_port}/sign_in?x=1", ssl=client_tls) as client:
                        await client.send(b"\x00frame")
                        assert await client.recv() == b"\x00frame"
                    break
                except OSError:
                    await asyncio.sleep(0.1)
            else:
                pytest.fail("proxy never accepted a connection")

            reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port, ssl=client_tls)
            writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            response = await reader.read(4096)
            writer.close()
        finally:
            stop.set_result(None)
            await proxy

    assert seen_paths == ["/sign_in?x=1"]
    assert response.startswith(b"HTTP/1.1 200")
    assert b"Access-Control-Allow-Origin: *" in response
    assert b"Certificate Accepted" in response
    assert (tmp_path / "certs" / "server.key").stat().st_mode & 0o777 == 0o600


def test_wss_proxy_refuses_half_a_certificate_pair(tmp_path):
    (tmp_path / "server.crt").write_text("cert")

    with pytest.raises(RuntimeError, match="missing server.key"):
        _wss.ensure_certificate(tmp_path)


def test_websockets_is_new_enough_for_the_asyncio_api():
    assert int(websockets.__version__.split(".")[0]) >= 14
