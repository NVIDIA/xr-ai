# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-process tests for export, reload, locking, and interrupted Riva builds."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
import select
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest
import yaml

BASE = Path(__file__).resolve().parents[1] / "model-server-samples/model-servers-nim"
SPEC = importlib.util.spec_from_file_location("sample_riva_server", BASE / "riva-server/nim_riva_server/__main__.py")
assert SPEC and SPEC.loader
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)

_FAKE_NIM = r"""
import gzip, io, json, os, tarfile, time
from pathlib import Path
workspace = Path(os.environ['NIM_WORKSPACE'])
export = Path(os.environ['NIM_EXPORT_PATH'])
build = os.environ['NIM_DISABLE_MODEL_DOWNLOAD'] == 'false'
with open(os.environ['EVENTS'], 'a') as stream:
    stream.write(json.dumps({'phase': 'build' if build else 'serve',
                            'repository': str(export), 'workspace': str(workspace),
                            'multiprocessing': os.environ.get('NIM_USE_MULTIPROCESSING_FOR_INFERENCE')}) + '\n')
if build:
    assert os.environ['NIM_USE_MULTIPROCESSING_FOR_INFERENCE'] == 'true'
    mode = os.environ.get('MODE', '')
    (workspace / 'model.rmir').touch()
    if mode == 'partial':
        (workspace / 'second.rmir').touch()
    if mode == 'failure':
        raise SystemExit(7)
    if mode == 'empty':
        raise SystemExit(0)
    with tarfile.open(export / 'model.tar.gz', 'w') as archive:
        files = {'model/config.pbtxt': b'name: "model"', 'model/1/encoder.plan': b'engine'}
        if mode in ('multi-model', 'later-header', 'gzip-later-header', 'truncated-header'):
            files.update({'decoder/config.pbtxt': b'name: "decoder"',
                          'decoder/1/decoder.plan': b'decoder engine'})
        if mode == 'no-engine':
            del files['model/1/encoder.plan']
        for name, content in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    if mode in ('later-header', 'gzip-later-header', 'truncated-header',
                'missing-end', 'single-end', 'garbage-after-end'):
        path = export / 'model.tar.gz'
        data = bytearray(path.read_bytes())
        with tarfile.open(path, 'r:') as archive:
            archive.getmembers()
            end = archive.offset
            if mode in ('later-header', 'gzip-later-header', 'truncated-header'):
                offset = archive.getmember('decoder/config.pbtxt').offset
        if mode in ('later-header', 'gzip-later-header'):
            data[offset] ^= 1  # Invalid checksum after a valid config/engine pair.
        elif mode == 'truncated-header':
            del data[offset + 100:]
        elif mode == 'missing-end':
            del data[end:]
        elif mode == 'single-end':
            del data[end + 512:]
        elif mode == 'garbage-after-end':
            data[-1] = 1
        path.write_bytes(data)
    if mode.startswith('gzip-'):
        path = export / 'model.tar.gz'
        # Stored DEFLATE blocks let us corrupt engine data without damaging
        # tar headers or the compression structure; only the CRC exposes it.
        data = bytearray(gzip.compress(path.read_bytes(), compresslevel=0, mtime=0))
        if mode == 'gzip-truncated':
            del data[-8:]
        elif mode == 'gzip-corrupt-payload':
            data[data.index(b'engine')] ^= 1
        path.write_bytes(data)
        if mode in ('gzip-truncated', 'gzip-corrupt-payload'):
            try:
                gzip.decompress(data)
            except (OSError, EOFError):
                pass
            else:
                raise AssertionError('fixture must contain genuine gzip corruption')
    if mode == 'truncated':
        (export / 'model.tar.gz').write_bytes(b'broken archive')
    gate = os.environ.get('GATE')
    while gate and not Path(gate).exists():
        time.sleep(0.02)
else:
    assert not list(workspace.iterdir()), 'serving workspace contains RMIRs'
    assert (export / 'complete.json').is_file()
    assert (export / 'model.tar.gz').is_file()
"""


@pytest.fixture
def runtime(tmp_path):
    executable = tmp_path / 'nvidia-smi'
    executable.write_text(f'#!{sys.executable}\nimport os\nprint(os.getenv("TEST_GPU", "Test GPU, 12.0, 580"))\n')
    executable.chmod(0o755)
    native = tmp_path / 'native.py'
    native.write_text(_FAKE_NIM)
    env = dict(os.environ, PATH=f'{tmp_path}{os.pathsep}{os.environ["PATH"]}',
               NIM_CACHE_PATH=str(tmp_path / 'cache'), EVENTS=str(tmp_path / 'events'),
               NIM_WORKSPACE=str(tmp_path / 'workspace'),
               NIM_USE_MULTIPROCESSING_FOR_INFERENCE='false',
               XR_AI_RIVA_COMMAND=json.dumps([sys.executable, str(native)]),
               XR_AI_RIVA_CONTRACT=json.dumps({'image': 'sha256:one', 'settings': 'one', 'build_format': 1}))
    # Keep temporary workspaces within the test directory, including after SIGKILL.
    env['TMPDIR'] = str(tmp_path)
    return env


def launch(env):
    return subprocess.run([sys.executable, str(server._SCRIPT)], env=env,
                          capture_output=True, text=True, timeout=20)


def events(env):
    path = Path(env['EVENTS'])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def wait_for_build(env):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if events(env):
            return
        time.sleep(0.02)
    pytest.fail('build did not start')


@pytest.mark.parametrize('mode', ['', 'gzip-valid', 'multi-model'])
def test_second_process_reuses_export_and_preserves_serving_settings(runtime, mode):
    runtime = dict(runtime, MODE=mode)
    first = launch(runtime)
    assert first.returncode == 0, first.stderr
    second = launch(runtime)
    assert second.returncode == 0, second.stderr
    rows = events(runtime)
    assert [row['phase'] for row in rows] == ['build', 'serve', 'serve']
    assert rows[1]['repository'] == rows[2]['repository']
    assert rows[1]['multiprocessing'] == rows[2]['multiprocessing'] == 'false'
    assert len({row['workspace'] for row in rows}) == 3
    assert 'Reusing compiled repository' in second.stdout


@pytest.mark.parametrize('mode', ['failure', 'empty', 'partial', 'no-engine', 'truncated',
                                  'gzip-truncated', 'gzip-corrupt-payload', 'later-header',
                                  'gzip-later-header', 'truncated-header', 'missing-end',
                                  'single-end', 'garbage-after-end'])
def test_failed_or_incomplete_export_is_never_published(runtime, mode):
    failed = launch(dict(runtime, MODE=mode))
    assert failed.returncode != 0
    assert not list(Path(runtime['NIM_CACHE_PATH']).rglob('complete.json'))
    assert [row['phase'] for row in events(runtime)] == ['build']
    retry = launch(runtime)
    assert retry.returncode == 0, retry.stderr
    assert [row['phase'] for row in events(runtime)] == ['build', 'build', 'serve']


def test_corrupted_export_is_rebuilt(runtime):
    assert launch(runtime).returncode == 0
    repository = Path(events(runtime)[1]['repository'])
    archive = repository / 'model.tar.gz'
    # Same-size corruption must not pass the cache-completeness check.
    data = bytearray(archive.read_bytes())
    data[512] ^= 1
    archive.write_bytes(data)
    retry = launch(runtime)
    assert retry.returncode == 0, retry.stderr
    assert [row['phase'] for row in events(runtime)] == ['build', 'serve', 'build', 'serve']


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('change', ['image', 'settings', 'gpu', 'driver', 'build_format'])
def test_incompatible_export_is_not_reused(runtime, change, legacy):
    assert launch(legacy_runtime(runtime) if legacy else runtime).returncode == 0
    changed = dict(runtime)
    if change in {'gpu', 'driver'}:
        changed['TEST_GPU'] = 'Different GPU, 8.9, 580' if change == 'gpu' else 'Test GPU, 12.0, 590'
    else:
        contract = json.loads(changed['XR_AI_RIVA_CONTRACT'])
        contract[change] = 2 if change == 'build_format' else 'two'
        changed['XR_AI_RIVA_CONTRACT'] = json.dumps(contract)
    result = launch(changed)
    assert result.returncode == 0, result.stderr
    rows = events(runtime)
    assert [row['phase'] for row in rows] == ['build', 'serve', 'build', 'serve']
    assert rows[1]['repository'] != rows[3]['repository']
    assert Path(rows[1]['repository'], 'complete.json').exists()


def test_concurrent_processes_build_once(runtime, tmp_path):
    gate = tmp_path / 'gate'
    first = subprocess.Popen([sys.executable, str(server._SCRIPT)],
                             env=dict(runtime, GATE=str(gate)), stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, text=True, start_new_session=True)
    second = None
    try:
        wait_for_build(runtime)
        second = subprocess.Popen([sys.executable, str(server._SCRIPT)],
                                  env=runtime, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  start_new_session=True)
        # A readiness line proves the second process reached the lock before
        # letting the first publish; no scheduling-dependent sleep is needed.
        assert second.stdout and select.select([second.stdout], [], [], 10)[0]
        assert 'Waiting for repository lock' in second.stdout.readline()
        gate.touch()
        assert first.wait(timeout=15) == 0, first.stderr.read()
        _, stderr = second.communicate(timeout=15)
        assert second.returncode == 0, stderr
    finally:
        for process in (first, second):
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    assert [row['phase'] for row in events(runtime)].count('build') == 1
    assert [row['phase'] for row in events(runtime)].count('serve') == 2


def test_killed_build_leaves_no_reusable_export_and_releases_lock(runtime, tmp_path):
    process = subprocess.Popen([sys.executable, str(server._SCRIPT)],
                               env=dict(runtime, GATE=str(tmp_path / 'never')),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    try:
        wait_for_build(runtime)
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    assert not list(Path(runtime['NIM_CACHE_PATH']).rglob('complete.json'))
    retry = launch(runtime)
    assert retry.returncode == 0, retry.stderr
    assert not list(Path(runtime['NIM_CACHE_PATH']).glob('riva-repositories/*.building'))
    assert [row['phase'] for row in events(runtime)] == ['build', 'build', 'serve']


@pytest.mark.parametrize('hardware,service', [
    ('96G_blackwell', 'stt'), ('96G_blackwell', 'tts'),
    ('dual_48G_ada', 'stt'), ('dual_48G_ada', 'tts'), ('spark', 'tts'),
])
def test_speech_launch_keeps_ownership_ports_gpu_and_native_entrypoint(hardware, service, tmp_path):
    cfg = yaml.safe_load((BASE / f'yaml/{hardware}/nim_{service}_server.yaml').read_text())
    image = {'Id': 'sha256:actual-image',
             'Config': {'Entrypoint': ['/bin/bash', '-c', '$SERVER_START_SCRIPT_PATH'], 'Cmd': None}}
    args = server._launch_args(cfg, tmp_path, image)
    assert f'xr-ai-vllm.port={cfg["http_port"]}' in args
    assert any(arg.startswith('xr-ai-vllm.config=') for arg in args)
    assert f'{cfg["http_port"]}:9000' in args
    assert f'{cfg["grpc_port"]}:50051' in args
    assert f'NVIDIA_VISIBLE_DEVICES={cfg["cuda_visible_devices"]}' in args
    assert f'{tmp_path}:/opt/nim/.cache' in args
    assert '--init' in args
    assert args[-2:] == ['sha256:actual-image', '/opt/xr-ai/riva_repository.py']
    native = next(arg for arg in args if arg.startswith('XR_AI_RIVA_COMMAND='))
    assert json.loads(native.partition('=')[2]) == image['Config']['Entrypoint']
    assert 'NGC_API_KEY' in args


def test_image_and_profile_changes_invalidate_container_and_repository_contract(tmp_path):
    cfg = yaml.safe_load((BASE / 'yaml/96G_blackwell/nim_tts_server.yaml').read_text())
    image = {'Id': 'sha256:one', 'Config': {'Entrypoint': ['start_server']}}
    original = server._launch_args(cfg, tmp_path, image)
    new_image = server._launch_args(cfg, tmp_path, dict(image, Id='sha256:two'))
    new_profile = server._launch_args(dict(cfg, env={'NIM_TAGS_SELECTOR': 'batch_size=32'}), tmp_path, image)
    for prefix in ('xr-ai-vllm.config=', 'XR_AI_RIVA_CONTRACT='):
        assert len({next(arg for arg in args if arg.startswith(prefix))
                    for args in (original, new_image, new_profile)}) == 3


@pytest.mark.parametrize('key', ['NIM_CACHE_PATH', 'NIM_WORKSPACE', 'NIM_EXPORT_PATH', 'NIM_DISABLE_MODEL_DOWNLOAD',
                                 'XR_AI_RIVA_CONTRACT', 'XR_AI_RIVA_COMMAND', 'XR_AI_RIVA_BOOTSTRAP'])
def test_reserved_cache_controls_cannot_silently_override_bootstrap(tmp_path, key):
    with pytest.raises(ValueError, match='manages these environment settings'):
        server._launch_args({'env': {key: 'override'}}, tmp_path, {})


def legacy_runtime(runtime):
    contract = json.loads(runtime['XR_AI_RIVA_CONTRACT'])
    contract.pop('build_format')
    contract['bootstrap'] = 'previous-bootstrap-source-hash'
    return dict(runtime, XR_AI_RIVA_CONTRACT=json.dumps(contract))


@pytest.mark.parametrize('mode', ['', 'gzip-valid', 'multi-model'])
def test_compatible_legacy_export_is_reused_in_place(runtime, mode):
    first = launch(dict(legacy_runtime(runtime), MODE=mode))
    assert first.returncode == 0, first.stderr
    repository = Path(events(runtime)[1]['repository'])
    manifest = (repository / 'complete.json').read_bytes()
    for _ in range(2):
        result = launch(runtime)
        assert result.returncode == 0, result.stderr
        assert 'Reusing validated legacy repository' in result.stdout
    rows = events(runtime)
    assert [row['phase'] for row in rows] == ['build', 'serve', 'serve', 'serve']
    assert {row['repository'] for row in rows[1:]} == {str(repository)}
    assert (repository / 'complete.json').read_bytes() == manifest


@pytest.mark.parametrize('state', ['locked', 'staging', 'malformed-manifest'])
def test_unavailable_legacy_export_is_not_adopted(runtime, state):
    assert launch(legacy_runtime(runtime)).returncode == 0
    repository = Path(events(runtime)[1]['repository'])
    with repository.with_suffix('.lock').open('a') as lock:
        if state == 'locked':
            fcntl.flock(lock, fcntl.LOCK_EX)
        elif state == 'staging':
            repository.rename(repository.with_suffix('.building'))
        else:
            (repository / 'complete.json').write_text('{')
        result = launch(runtime)
    assert result.returncode == 0, result.stderr
    assert [row['phase'] for row in events(runtime)] == ['build', 'serve', 'build', 'serve']


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('corruption', ['later-header', 'gzip-crc', 'gzip-truncated'])
def test_manifest_cannot_bless_malformed_cached_archives(runtime, legacy, corruption):
    first = launch(dict(legacy_runtime(runtime) if legacy else runtime, MODE='multi-model'))
    assert first.returncode == 0, first.stderr
    repository = Path(events(runtime)[1]['repository'])
    path = repository / 'model.tar.gz'
    data = bytearray(path.read_bytes())
    if corruption == 'later-header':
        with tarfile.open(path, 'r:') as archive:
            data[archive.getmember('decoder/config.pbtxt').offset] ^= 1
    else:
        data = bytearray(gzip.compress(data, compresslevel=0, mtime=0))
        if corruption == 'gzip-crc':
            data[data.index(b'engine')] ^= 1
        else:
            del data[-8:]
    path.write_bytes(data)
    # Simulate an older validator publishing damaged bytes with matching hashes.
    manifest_path = repository / 'complete.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['archives'][path.name] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    manifest_path.write_text(json.dumps(manifest))
    result = launch(runtime)
    assert result.returncode == 0, result.stderr
    assert [row['phase'] for row in events(runtime)] == ['build', 'serve', 'build', 'serve']


def test_bootstrap_edit_updates_container_without_invalidating_engines(tmp_path, monkeypatch):
    cfg = yaml.safe_load((BASE / 'yaml/96G_blackwell/nim_tts_server.yaml').read_text())
    image = {'Id': 'sha256:one', 'Config': {'Entrypoint': ['start_server']}}
    script = tmp_path / 'repository.py'
    script.write_bytes(server._SCRIPT.read_bytes())
    monkeypatch.setattr(server, '_SCRIPT', script)
    original = server._launch_args(cfg, tmp_path, image)
    script.write_text(script.read_text() + '\n# A validation-only change.\n')
    changed = server._launch_args(cfg, tmp_path, image)
    prefix = 'XR_AI_RIVA_CONTRACT='
    assert next(arg for arg in original if arg.startswith(prefix)) == next(
        arg for arg in changed if arg.startswith(prefix))
    for prefix in ('xr-ai-vllm.config=', 'XR_AI_RIVA_BOOTSTRAP='):
        assert next(arg for arg in original if arg.startswith(prefix)) != next(
            arg for arg in changed if arg.startswith(prefix))


def test_bootstrap_fingerprint_change_reuses_engines(runtime):
    assert launch(dict(runtime, XR_AI_RIVA_BOOTSTRAP='old')).returncode == 0
    result = launch(dict(runtime, XR_AI_RIVA_BOOTSTRAP='new'))
    assert result.returncode == 0, result.stderr
    assert [row['phase'] for row in events(runtime)] == ['build', 'serve', 'serve']


@pytest.mark.gpu
@pytest.mark.skipif(
    not os.getenv('XR_AI_TEST_RIVA_IMAGE'), reason='requires an explicitly selected local Riva image and Docker',
)
def test_real_docker_stop_remove_restart_and_stop_during_build(tmp_path, monkeypatch):
    """Exercise Docker lifecycle with tiny stand-in engines and an HTTP server."""
    import socket
    import uuid

    from xr_ai_vllm import stop_persistent_servers
    from xr_ai_vllm._lifecycle import health_ok

    selected_image = os.environ['XR_AI_TEST_RIVA_IMAGE']
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', selected_image]))[0]
    image['Config']['Entrypoint'] = ['python3', '/validation/native.py']
    image['Config']['Cmd'] = None
    native = _FAKE_NIM + '''
if not build:
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
    HTTPServer(('0.0.0.0', 9000), Handler).serve_forever()
'''
    (tmp_path / 'native.py').write_text(native)
    tmp_path.chmod(0o777)
    cache = tmp_path / 'cache'
    cache.mkdir(mode=0o777)
    cache.chmod(0o777)
    with socket.socket() as http_socket, socket.socket() as grpc_socket:
        http_socket.bind(('127.0.0.1', 0))
        grpc_socket.bind(('127.0.0.1', 0))
        port, grpc_port = http_socket.getsockname()[1], grpc_socket.getsockname()[1]
    name = f'xr-ai-riva-cache-test-{uuid.uuid4().hex[:10]}'
    cfg = {'container_name': name, 'http_port': port, 'grpc_port': grpc_port,
           'cuda_visible_devices': '0', 'env': {'EVENTS': '/validation/events'}}
    monkeypatch.setenv('NGC_API_KEY', 'unused-by-test-native-command')
    wrappers = []

    def start(config, attempt):
        args = server._launch_args(config, cache, image)
        args[-2:-2] = ['--mount', f'type=bind,src={tmp_path},dst=/validation']
        for index, value in enumerate(args):
            if value == '-p':
                args[index + 1] = f'127.0.0.1:{args[index + 1]}'
        ready = tmp_path / f'ready-{attempt}'
        kwargs = dict(argv=args, image=image['Id'], container_name=name, log_prefix=name,
                      port=port, health_url=f'http://127.0.0.1:{port}/v1/health/ready',
                      launch_banner='test launch', reuse_banner='test reuse', ready_banner='test ready')
        script = ('import json; from pathlib import Path; from xr_ai_vllm import _docker; '
                  f'_docker.run_container(**json.loads({json.dumps(kwargs)!r}), ready_file=Path({str(ready)!r}))')
        log = tmp_path / f'wrapper-{attempt}.log'
        with log.open('w') as stream:
            process = subprocess.Popen([sys.executable, '-c', script], stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        wrappers.append(process)
        return ready, process, log

    def wait_until(predicate, process, log):
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline and process.poll() is None:
            if predicate():
                return
            time.sleep(0.1)
        pytest.fail(log.read_text())

    try:
        for attempt in (1, 2):
            ready, process, log = start(cfg, attempt)
            wait_until(ready.is_file, process, log)
            assert health_ok(f'http://127.0.0.1:{port}/v1/health/ready')
            if attempt == 1:
                identity = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}', name])
                reused_ready, reused_process, reused_log = start(cfg, 'reuse')
                wait_until(reused_ready.is_file, reused_process, reused_log)
                reused_identity = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}', name])
                assert reused_identity == identity
            stopped = stop_persistent_servers([('riva-test', port)])
            assert stopped
            process.wait(timeout=15)
            assert not server._docker.container_exists(name)
        rows = events({'EVENTS': str(tmp_path / 'events')})
        assert [row['phase'] for row in rows] == ['build', 'serve', 'serve']
        assert list(cache.rglob('complete.json'))

        cfg['env']['GATE'] = '/validation/never'
        ready, process, log = start(cfg, 3)
        wait_until(lambda: len(events({'EVENTS': str(tmp_path / 'events')})) == 4, process, log)
        assert not ready.exists()
        stopped = stop_persistent_servers([('riva-test', port)])
        assert stopped
        process.wait(timeout=15)
        assert not server._docker.container_exists(name)
        assert len(list(cache.rglob('complete.json'))) == 1
    finally:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for process in wrappers:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        # Return ownership of artifacts written by the image's user so pytest
        # can remove this test's cache without leaving inaccessible directories.
        subprocess.run(['docker', 'run', '--rm', '--user', '0', '--entrypoint', 'chown',
                        '--mount', f'type=bind,src={tmp_path},dst=/validation', image['Id'],
                        '-R', f'{os.getuid()}:{os.getgid()}', '/validation'], check=True)


@pytest.mark.parametrize('gpu', ['', '[N/A]'])
def test_missing_gpu_identity_fails_before_build(runtime, gpu):
    result = launch(dict(runtime, TEST_GPU=gpu))
    assert result.returncode != 0
    assert 'cannot identify the visible GPU' in result.stderr
    assert not events(runtime)


def test_archive_digest_does_not_require_python_311_file_digest(tmp_path, monkeypatch):
    from nim_riva_server import repository
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    archive = tmp_path / 'archive'
    content = b'engine' * 500000
    archive.write_bytes(content)
    assert repository._digest(archive) == hashlib.sha256(content).hexdigest()


def test_entrypoint_splice_rejects_changed_shared_argument_contract(tmp_path, monkeypatch):
    cfg = yaml.safe_load((BASE / 'yaml/96G_blackwell/nim_tts_server.yaml').read_text())
    image = {'Id': 'sha256:one', 'Config': {'Entrypoint': ['start_server']}}
    monkeypatch.setattr(server, 'build_nim_run_argv', lambda **kwargs: ['docker', 'run', 'sha256:one', 'command'])
    with pytest.raises(RuntimeError, match='must end with the image'):
        server._launch_args(cfg, tmp_path, image)
