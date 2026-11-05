# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CloudXR runtime service: SDK fetch, environment, EULA, process, and WSS proxy."""
from __future__ import annotations

import asyncio
import errno
import hashlib
import io
import os
import ssl
import subprocess
import sys
import tarfile
import textwrap
import threading
from http.client import IncompleteRead
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

_PROJECT = Path(__file__).resolve().parents[1] / "services" / "cloudxr-runtime"
sys.path.insert(0, str(_PROJECT))

from cloudxr_runtime import _runtime, _sdk, _wss  # noqa: E402
from cloudxr_runtime._runtime import Layout  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_environment():
    with patch.dict(os.environ):
        yield

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


def test_sdk_interrupted_response_removes_partial_download(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, *_):
            raise IncompleteRead(b"truncated")

    monkeypatch.setattr(_sdk.urllib.request, "urlopen", lambda *_a, **_k: Response())
    target = tmp_path / "sdk.tar.gz"
    with pytest.raises(_sdk.SDKError, match="To install offline"):
        _sdk._download("https://example.invalid/sdk", target)
    assert not target.exists()
    assert not target.with_suffix(".partial").exists()


def test_sdk_missing_native_library_is_not_published(tmp_path, placed_tarball, monkeypatch):
    monkeypatch.setattr(_sdk, "_unpack", lambda _archive, target: target.mkdir())
    with pytest.raises(_sdk.SDKError, match="libcloudxr.so"):
        _sdk.ensure_sdk(tmp_path)
    assert not (tmp_path / "sdk" / f"{_sdk.SDK_VERSION}-amd64").exists()


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
    monkeypatch.setenv("NV_DEVICE_PROFILE", "auto-webrtc")
    for key in ("XR_RUNTIME_JSON", "NV_CXR_RUNTIME_DIR", "NV_CXR_OUTPUT_DIR", "XRT_NO_STDIN",
                "CXR_INSTALL_DIR", "CXR_HOST_VOLUME_PATH"):
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
    assert b"Certificate Accepted" in response
    assert (tmp_path / "certs" / "server.key").stat().st_mode & 0o777 == 0o600


def test_wss_proxy_refuses_half_a_certificate_pair(tmp_path):
    (tmp_path / "server.crt").write_text("cert")

    with pytest.raises(RuntimeError, match="missing server.key"):
        _wss.ensure_certificate(tmp_path)


@pytest.mark.parametrize("signals,join_result,stop_result", [(0, 0, 0), (2, 0, 0), (2, -6, -6)])
def test_native_shutdown_stops_once_and_destroys_on_owner_thread(
    tmp_path, monkeypatch, signals, join_result, stop_result,
):
    from cloudxr_runtime import _service

    monkeypatch.setenv("NV_CXR_FILE_LOGGING", "false")
    handlers = {}
    calls = []
    owner = threading.get_ident()

    class Library:
        def nv_cxr_service_create(self, out):
            out._obj.value = 1
            calls.append("create")
            return 0

        def nv_cxr_service_start(self, _handle):
            calls.append("start")
            return 0

        def nv_cxr_service_stop(self, _handle):
            calls.append("stop")
            return stop_result

        def nv_cxr_service_join(self, _handle):
            for _ in range(signals):
                handlers[_service.signal.SIGTERM](15, None)
            calls.append("join")
            return join_result

        def nv_cxr_service_destroy(self, _handle):
            assert threading.get_ident() == owner
            assert handlers[_service.signal.SIGTERM] == _service.signal.SIG_IGN
            calls.append("destroy")

    monkeypatch.setattr(_service.ctypes, "CDLL", lambda *_a, **_k: Library())
    monkeypatch.setattr(_service.signal, "signal", lambda sig, callback: handlers.__setitem__(sig, callback))
    _service.serve(tmp_path, tmp_path)
    assert calls == ["create", "start", *(["stop"] if signals else []), "join", "destroy"]


@pytest.mark.parametrize("failed_call", ["create", "start"])
def test_native_startup_error_does_not_use_invalid_handle(tmp_path, monkeypatch, failed_call):
    from cloudxr_runtime import _service

    monkeypatch.setenv("NV_CXR_FILE_LOGGING", "false")
    calls = []

    class Library:
        def nv_cxr_service_create(self, out):
            calls.append("create")
            if failed_call == "create":
                return -1
            out._obj.value = 1
            return 0

        def nv_cxr_service_start(self, _handle):
            calls.append("start")
            return -2

        def nv_cxr_service_destroy(self, _handle):
            calls.append("destroy")

    monkeypatch.setattr(_service.ctypes, "CDLL", lambda *_a, **_k: Library())
    monkeypatch.setattr(_service.signal, "signal", lambda *_: None)
    with pytest.raises(RuntimeError, match=f"nv_cxr_service_{failed_call}"):
        _service.serve(tmp_path, tmp_path)
    assert calls == (["create"] if failed_call == "create" else ["create", "start", "destroy"])


def test_preparation_finishes_before_the_async_lifecycle(tmp_path, monkeypatch):
    from cloudxr_runtime import __main__ as main

    calls = []
    sdk = _sdk.SDK(tmp_path / "native")
    monkeypatch.setattr(main, "publish_env", lambda *_: calls.append("environment"))
    monkeypatch.setattr(main, "ensure_sdk", lambda *_: calls.append("sdk") or sdk)
    monkeypatch.setattr(main, "check_eula", lambda *_a, **_k: calls.append("eula"))
    monkeypatch.setattr(main, "prepare_run_dir", lambda *_: calls.append("run_dir"))
    layout, prepared = main._prepare({"cloudxr_install_dir": str(tmp_path)})
    assert layout == Layout(tmp_path)
    assert prepared is sdk
    assert calls == ["environment", "sdk", "eula", "run_dir"]


def test_sdk_failure_exits_before_starting_an_event_loop(tmp_path, monkeypatch):
    from cloudxr_runtime import __main__ as main

    def fail(_):
        raise _sdk.SDKError("offline install required")

    monkeypatch.setattr(main, "_prepare", fail)
    monkeypatch.setattr(main, "setup_logging", lambda *_: None)
    monkeypatch.setattr(main.sys, "argv", ["cloudxr_runtime"])
    monkeypatch.setattr(main.sys.stdout, "reconfigure", lambda **_: None)
    monkeypatch.setattr(main.sys.stderr, "reconfigure", lambda **_: None)
    monkeypatch.setattr(main.asyncio, "run", lambda *_: pytest.fail("started async lifecycle after SDK failure"))
    with pytest.raises(SystemExit) as failure:
        main.run()
    assert failure.value.code == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["early_exit", "native", "proxy_error"])
async def test_runtime_orchestration_always_stops_its_child(tmp_path, monkeypatch, outcome):
    from cloudxr_runtime import __main__ as main

    calls = []
    process = SimpleNamespace(poll=lambda: 3 if outcome == "early_exit" else None)
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda *_: None)
    monkeypatch.setattr(main, "start_service", lambda *_: calls.append("start") or process)
    monkeypatch.setattr(main, "stop_service", lambda child: calls.append(("stop", child)))
    monkeypatch.setattr(main, "latest_runtime_log", lambda *_: None)
    monkeypatch.setattr(main, "read_device_profile", lambda *_: "auto-native" if outcome == "native" else "auto-webrtc")

    async def ready(_child, stop, _sentinel):
        if outcome == "native":
            stop.set_result(None)
        return outcome != "early_exit"

    async def proxy(*_):
        calls.append("proxy")
        raise RuntimeError("half a TLS certificate pair")

    monkeypatch.setattr(main, "_wait_with_progress", ready)
    monkeypatch.setattr(main._wss, "run", proxy)
    ready_file = tmp_path / "ready"
    operation = main._run(Layout(tmp_path), _sdk.SDK(tmp_path), ready_file=ready_file)
    if outcome == "early_exit":
        with pytest.raises(SystemExit) as failure:
            await operation
        assert failure.value.code == 3
        assert not ready_file.exists()
    elif outcome == "proxy_error":
        with pytest.raises(RuntimeError, match="half a TLS certificate pair"):
            await operation
    else:
        await operation
    assert calls == ["start", *(["proxy"] if outcome == "proxy_error" else []), ("stop", process)]


def test_openssl_failure_surfaces_captured_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr(_wss.shutil, "which", lambda *_: "/usr/bin/openssl")

    def fail(*_a, **_k):
        raise subprocess.CalledProcessError(1, "openssl", stderr=b"cannot write certificate")

    monkeypatch.setattr(_wss.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="cannot write certificate") as failure:
        _wss.ensure_certificate(tmp_path)
    assert isinstance(failure.value.__cause__, subprocess.CalledProcessError)
    assert "port" not in str(failure.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_number", [errno.EADDRINUSE, errno.EACCES])
async def test_only_address_in_use_gets_a_port_hint(tmp_path, monkeypatch, error_number):
    context = SimpleNamespace(load_cert_chain=lambda **_: None)
    monkeypatch.setattr(_wss, "ensure_certificate", lambda _: (tmp_path / "cert", tmp_path / "key"))
    monkeypatch.setattr(_wss.ssl, "SSLContext", lambda *_: context)

    def fail(*_a, **_k):
        raise OSError(error_number, "bind failed")

    monkeypatch.setattr(_wss, "ws_serve", fail)
    expected = RuntimeError if error_number == errno.EADDRINUSE else OSError
    with pytest.raises(expected) as failure:
        await _wss.run(tmp_path, asyncio.get_running_loop().create_future(), port=48444)
    assert ("fuser 48444/tcp" in str(failure.value)) == (error_number == errno.EADDRINUSE)


@pytest.mark.parametrize("natural_exit", [False, True])
def test_native_child_real_signals_cannot_reach_a_destroyed_handle(tmp_path, natural_exit):
    script = textwrap.dedent("""\
        import os
        import signal
        import sys
        import threading
        from pathlib import Path
        from cloudxr_runtime import _service

        owner = threading.get_ident()
        stopped = threading.Event()
        calls = []
        natural_exit = sys.argv[2] == "True"

        class Library:
            def nv_cxr_service_create(self, out):
                out._obj.value = 1
                return 0

            def nv_cxr_service_start(self, handle):
                return 0

            def nv_cxr_service_stop(self, handle):
                calls.append("stop")
                stopped.set()
                return 0

            def nv_cxr_service_join(self, handle):
                if not natural_exit:
                    os.kill(os.getpid(), signal.SIGTERM)
                    stopped.wait()
                    os.kill(os.getpid(), signal.SIGTERM)
                return 0

            def nv_cxr_service_destroy(self, handle):
                assert threading.get_ident() == owner
                assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
                signal.raise_signal(signal.SIGTERM)
                calls.append("destroy")

        _service.ctypes.CDLL = lambda *args, **kwargs: Library()
        _service.serve(Path(sys.argv[1]), Path(sys.argv[1]))
        assert calls == (["destroy"] if natural_exit else ["stop", "destroy"]), calls
        print("closed")
        """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(natural_exit)],
        env={**os.environ, "PYTHONPATH": str(_PROJECT), "NV_CXR_FILE_LOGGING": "false"},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "closed"
