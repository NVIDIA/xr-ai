# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2] / "deployments" / "brev"
# The main pytest workflow runs from tests/, not the repository root.
with patch.object(sys, "path", [str(ROOT.parents[1]), *sys.path]):
    from deployments.brev import compat, releases, server


class DeploymentTests(unittest.TestCase):
    def test_stop_all_preserves_chooser_if_sample_shutdown_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.shell(
                'acquire_lock() { :; }; stop_locked() { return 1; }; '
                'stop_ui() { echo unexpected-ui-stop; }; stop_sample --all', directory,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("unexpected-ui-stop", result.stdout)

    def test_stop_ui_removes_dead_pid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "ui.pid").write_text("stale\n")
            result = self.shell("stop_ui; stop_ui", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((Path(directory) / "ui.pid").exists())

    @unittest.skipUnless(
        Path("/proc/self/cmdline").exists() and shutil.which("setsid"), "requires Linux procfs and setsid",
    )
    def test_start_ui_restarts_owned_chooser(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.shell('''
trap stop_ui EXIT
start_ui
old="$(first_line "$UI_PID_FILE")"
start_ui
[[ "$(first_line "$UI_PID_FILE")" != "$old" ]]
''', directory, XR_AI_UI_PORT="0", XR_AI_UI_AUTH="password", XR_AI_UI_BIND="127.0.0.1")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((Path(directory) / "ui.pid").exists())

    @unittest.skipUnless(
        Path("/proc/self/cmdline").exists() and shutil.which("flock"), "requires Linux procfs and flock",
    )
    def test_stop_all_closes_chooser_and_preserves_saved_state(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            for name in ("hf", "ui.password", "ui.log"):
                (state / name).write_text("saved-value-that-is-long-enough-for-a-password\n")
                (state / name).chmod(0o600)
            with subprocess.Popen(
                [sys.executable, str(ROOT / "server.py"), str(ROOT / "launch.sh"),
                 str(ROOT / "ui.html"), directory, "0"],
                env=os.environ | {"XR_AI_UI_BIND": "127.0.0.1", "XR_AI_UI_AUTH": "password"},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ) as chooser:
                try:
                    for _ in range(100):
                        if (state / "ui.pid").exists() or chooser.poll() is not None:
                            break
                        time.sleep(0.05)
                    self.assertIsNone(chooser.poll())
                    self.assertTrue((state / "ui.pid").exists())
                    result = self.shell('stop_locked() { echo models-stopped; }; stop_sample', directory)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIsNone(chooser.poll(), "plain stop must leave the chooser running")
                    result = self.shell('stop_locked() { echo models-stopped; }; stop_sample --all', directory)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("models-stopped", result.stdout)
                    self.assertIn("chooser stopped", result.stderr)
                    chooser.wait(timeout=5)
                    self.assertFalse((state / "ui.pid").exists())
                    result = self.shell('stop_locked() { :; }; stop_sample --all', directory)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    for name in ("hf", "ui.password", "ui.log"):
                        self.assertTrue((state / name).exists())
                finally:
                    if chooser.poll() is None:
                        chooser.terminate()
                    chooser.wait(timeout=5)

    @unittest.skipUnless(Path("/proc/self/cmdline").exists(), "requires Linux procfs")
    def test_chooser_stop_and_restart_refuse_unrelated_live_pid(self):
        for action in ("stop_ui", "start_ui"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory, \
                    subprocess.Popen(["sleep", "30"]) as unrelated:
                try:
                    pid_file = Path(directory) / "ui.pid"
                    pid_file.write_text(str(unrelated.pid))
                    result = self.shell(
                        'nohup() { echo unexpected-ui-start; }; ' + action, directory,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("does not match this launcher", result.stderr)
                    self.assertNotIn("unexpected-ui-start", result.stdout)
                    self.assertIsNone(unrelated.poll())
                    self.assertEqual(pid_file.read_text(), str(unrelated.pid))
                finally:
                    if unrelated.poll() is None:
                        unrelated.terminate()
                    unrelated.wait(timeout=5)

    @unittest.skipUnless(shutil.which("flock"), "requires Linux util-linux flock")
    def test_action_lock_released_after_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            script = "set -- help; source ./launch.sh >/dev/null; acquire_lock; echo locked; exec sleep 30"
            with subprocess.Popen(
                ["bash", "-c", script], cwd=ROOT,
                env=os.environ | {"XR_AI_STATE": directory}, stdout=subprocess.PIPE, text=True,
            ) as holder:
                try:
                    self.assertEqual(holder.stdout.readline().strip(), "locked")
                    busy = self.shell("acquire_lock", directory)
                    self.assertNotEqual(busy.returncode, 0)
                    self.assertIn("another launcher action", busy.stderr)
                finally:
                    holder.kill()
                    holder.wait(timeout=5)
            available = self.shell("acquire_lock; release_lock; acquire_lock", directory)
            self.assertEqual(available.returncode, 0, available.stderr)

    def shell(self, script, state, **env):
        return subprocess.run(
            ["bash", "-c", "set -- help; source ./launch.sh >/dev/null\n" + script],
            cwd=ROOT,
            env=os.environ | {"XR_AI_STATE": str(state), "XR_AI_DEPLOYMENT_PYTHON": sys.executable} | env,
            text=True,
            capture_output=True,
        )

    def test_no_automatic_privileged_host_mutation(self):
        source = (ROOT / "launch.sh").read_text()
        setup = source.split("\nsetup() {", 1)[1].split("\nusage()", 1)[0]
        self.assertNotIn("install_boot_service", setup)
        self.assertNotIn("upgrade_gpu_stack", source)
        self.assertNotIn("apt-get", source)
        self.assertNotIn("shutdown -r", source)
        self.assertNotIn("docker rm", source)
        self.assertNotIn("pkill", source)
        self.assertNotIn("sudo -n -E", source)
        self.assertNotRegex(setup, r"(?m)^\s*sudo\b")

    def test_saved_network_settings_and_boot_service(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.shell(
                "save_host_settings", directory,
                XR_AI_PUBLIC_HOST="192.168.1.50", XR_AI_UI_PORT="9091", XR_AI_WEB_PORT="9080",
                XR_AI_LIVEKIT_WS_PORT="9880", XR_AI_LIVEKIT_TCP_PORT="9881",
                XR_AI_LIVEKIT_UDP_PORT="9882", XR_AI_TOKEN_PORT="9000", HF_TOKEN="hf_secret",
                XR_AI_UI_AUTH="proxy", XR_AI_TRUST_PROXY="1",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            settings = Path(directory) / "host.env"
            self.assertNotIn("hf_secret", settings.read_text())
            self.assertEqual(settings.stat().st_mode & 0o777, 0o600)
            result = self.shell(
                'printf "%s %s %s %s %s %s %s %s %s" "$XR_AI_PUBLIC_HOST" "$UI_PORT" "$WEB_PORT" '
                '"$LIVEKIT_WS_PORT" "$LIVEKIT_TCP_PORT" "$LIVEKIT_UDP_PORT" "$TOKEN_PORT" '
                '"$XR_AI_UI_AUTH" "$XR_AI_TRUST_PROXY"', directory,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "192.168.1.50 9091 9080 9880 9881 9882 9000 proxy 1")
            result = self.shell(
                'printf "%s|%s" "$XR_AI_PUBLIC_HOST" "$WEB_PORT"', directory,
                XR_AI_PUBLIC_HOST="", XR_AI_WEB_PORT="8085",
            )
            self.assertEqual(result.stdout, "|8085")
            result = self.shell(
                'sudo() { if [[ "$1" == tee ]]; then command cat > "$STATE_DIR/unit"; fi; }; '
                'install_boot_service', directory,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"EnvironmentFile={directory}/host.env", (Path(directory) / "unit").read_text())
            self.assertIn("XR_AI_UI_AUTH=proxy", settings.read_text())
            self.assertIn("XR_AI_TRUST_PROXY=1", settings.read_text())

    def test_saved_network_settings_reject_unit_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.shell("save_host_settings", directory, XR_AI_PUBLIC_HOST="address\nOTHER=value")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsupported characters", result.stderr)

    def test_saved_defaults_remain_unset_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.shell("save_host_settings", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((Path(directory) / "host.env").read_text(), "XR_AI_PUBLIC_HOST=\n")
            result = self.shell(
                'printf "%s|%s|%s|%s" "${XR_AI_UI_BIND-unset}" "${XR_AI_WEB_PORT-unset}" '
                '"$UI_PORT" "$WEB_PORT"', directory,
            )
            self.assertEqual(result.stdout, "unset|unset|8091|8080")

    def test_setup_revision_stops_old_stack_before_activation(self):
        for fail in (False, True):
            with self.subTest(shutdown_fails=fail), tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                old, new = state / "old", state / "new"
                (state / "runtime-root").write_text(str(old) + "\n")
                (state / "models.active").write_text(f"{old}/models\nmodel_servers\n")
                result = self.shell('''
acquire_lock() { echo lock >> "$STATE_DIR/calls"; }
release_lock() { echo unlock >> "$STATE_DIR/calls"; }
resolve_source() { SOURCE_RUNTIME="$STATE_DIR/new"; }
pid_alive() { return 1; }
git() { echo commit; }
uv() { echo "$*" >> "$STATE_DIR/calls"; [[ "$(runtime_root)" == "$STATE_DIR/old" ]] && return "$STOP_RC"; }
ensure_source
''', directory, XR_AI_VERSION="v0.3.0-beta", STOP_RC="1" if fail else "0")
                self.assertEqual(result.returncode, 1 if fail else 0, result.stderr)
                self.assertEqual((state / "runtime-root").read_text().strip(), str(old if fail else new))
                calls = (state / "calls").read_text().splitlines()
                self.assertEqual(calls[0], "lock")
                self.assertIn(f"--project {old}/models model_servers --stop", calls[1])
                self.assertEqual(calls[-1], "unlock")

    def test_release_tags_resolve_to_commits_and_remain_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "upstream"
            sample = repo / "agent-samples/simple-vlm-example"
            sample.mkdir(parents=True)
            project = sample / "pyproject.toml"
            project.write_text("[project]\nname='initial'\n")

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()

            git("init", "-q")
            git("add", ".")
            git("commit", "-qm", "initial")
            commit = git("rev-parse", "HEAD")
            git("tag", "v1.0.0-beta.9")
            git("tag", "-a", "v2.0.0", "-m", "annotated release")
            # Exercise real fetch/peeling/worktrees without contacting GitHub.
            fetch = '''
git() {
  if [[ "${3:-}" == fetch ]]; then
    command git -C "$2" fetch --quiet --depth 1 "$TEST_REMOTE" "${@: -1}"
  else command git "$@"; fi
}
'''
            for revision in ("v1.0.0-beta.9", "v2.0.0", commit):
                for local in (False, True):
                    with self.subTest(revision=revision, local=local):
                        state = Path(directory) / f"state-{revision}-{local}"
                        result = self.shell(
                            fetch + 'resolve_source "$VERSION"; activate_source; '
                            'git -C "$(runtime_root)" rev-parse HEAD',
                            state, TEST_REMOTE=str(repo), VERSION=revision,
                            XR_AI_ROOT=str(repo) if local else "",
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stdout.strip(), commit)
            for revision in ("v99.0.0", "main", "--upload-pack=bad"):
                with self.subTest(invalid=revision):
                    result = self.shell(
                        fetch + 'resolve_source "$VERSION"', Path(directory) / "invalid",
                        TEST_REMOTE=str(repo), VERSION=revision,
                    )
                    self.assertNotEqual(result.returncode, 0)
            state = Path(directory) / "state-v2.0.0-False"
            project.write_text("[project]\nname='updated'\n")
            git("add", ".")
            git("commit", "-qm", "updated")
            git("tag", "-fa", "v2.0.0", "-m", "moved release")
            saved = self.shell('ensure_source; git -C "$(runtime_root)" rev-parse HEAD', state)
            self.assertEqual(saved.returncode, 0, saved.stderr)
            self.assertEqual(saved.stdout.strip(), commit)
            selected = self.shell(
                fetch + 'resolve_source v2.0.0; activate_source; git -C "$(runtime_root)" rev-parse HEAD',
                state, TEST_REMOTE=str(repo),
            )
            self.assertEqual(selected.returncode, 0, selected.stderr)
            self.assertEqual(selected.stdout.strip(), git("rev-parse", "HEAD"))
            # Latest resolves only when selected; saved restarts retain its exact commit.
            latest_state = Path(directory) / "latest"
            latest = self.shell(fetch + '''
deployment_python() { echo v2.0.0; }
resolve_source; activate_source
printf '%s|%s' "$(<"$STATE_DIR/runtime-ref")" "$(git -C "$(runtime_root)" rev-parse HEAD)"
''', latest_state, TEST_REMOTE=str(repo))
            self.assertEqual(latest.returncode, 0, latest.stderr)
            self.assertEqual(latest.stdout, f"v2.0.0|{git('rev-parse', 'HEAD')}")
            current = self.shell(
                'resolve_source current; activate_source; runtime_root', latest_state, XR_AI_ROOT=str(repo),
            )
            self.assertEqual(current.returncode, 0, current.stderr)
            self.assertEqual(Path(current.stdout.strip()), repo.resolve())
            switched = self.shell(fetch + 'resolve_source v1.0.0-beta.9; activate_source; '
                                  'git -C "$(runtime_root)" rev-parse HEAD', latest_state,
                                  XR_AI_ROOT=str(repo), TEST_REMOTE=str(repo))
            self.assertEqual(switched.returncode, 0, switched.stderr)
            self.assertEqual(switched.stdout.strip(), commit)
            saved = self.shell('ensure_source; cat "$STATE_DIR/runtime-ref"', latest_state, XR_AI_ROOT=str(repo))
            self.assertEqual(saved.stdout.strip(), "v1.0.0-beta.9")

    def test_release_discovery_orders_versions_and_filters_unsupported_tags(self):
        tags = ["v0.1.0", "v0.2.0-beta", "v0.2.0-beta.2", "v0.3.0-beta", "v0.3.0-beta.10",
                "v0.3.0-beta.2", "v0.3.0-rc.1", "v0.3.0", "v0.10.0-beta", "v0.9.0", "random"]
        output = "\n".join(f"{'a' * 40}\trefs/tags/{tag}" for tag in tags)
        with patch.object(releases.subprocess, "check_output", return_value=output):
            value = releases.releases()
        self.assertEqual(value["latest"], "v0.10.0-beta")
        self.assertEqual(value["tags"], ["v0.10.0-beta", "v0.9.0", "v0.3.0", "v0.3.0-rc.1", "v0.3.0-beta.10",
                                         "v0.3.0-beta.2", "v0.3.0-beta", "v0.2.0-beta.2"])
        with patch.object(releases.subprocess, "check_output", return_value=""), self.assertRaises(ValueError):
            releases.releases()

    def test_model_layout_start_stop(self):
        layouts = (
            "agent-samples/model-servers",
            "model-server-samples/model-servers",
            "agent-samples/simple-vlm-example/model-servers",
        )
        for layout in layouts:
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "repo"
                project = root / layout
                project.mkdir(parents=True)
                command = "sample_model_servers" if layout.endswith("example/model-servers") else "model_servers"
                (project / "pyproject.toml").write_text(f'[project.scripts]\n{command} = "main:run"\n')
                result = self.shell(
                    """
uv() { echo "$*" >> "$STATE_DIR/calls"; }
(run_sample agent-samples/simple-vlm-example simple_vlm_example)
pid_alive() { return 1; }
stop_locked
""",
                    directory,
                    XR_AI_ROOT=str(root),
                    HF_TOKEN="hf_test",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = (Path(directory) / "calls").read_text().splitlines()
                self.assertIn(f"--project {project} {command}", calls[0])
                self.assertIn("simple_vlm_example", calls[1])
                self.assertIn(f"--project {project} {command} --stop", calls[2])
                self.assertIn("XR_AI_SAMPLE_START", result.stdout)

    def test_model_shutdown_failure_blocks_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "models.active").write_text(f"{directory}/models\nmodel_servers\n")
            result = self.shell(
                "pid_alive() { return 1; }; uv() { return 1; }; stop_locked", directory, XR_AI_ROOT=directory
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("model shutdown failed", result.stderr)

    def test_start_resolves_once_before_stopping_active_sample(self):
        for sample, recording, flag in (
            ("auto", "", ""), ("simple-vlm", "", "--session-recording"), ("blocked", "", ""),
            ("simple-vlm", "--session-recording", "--session-recording"),
            ("simple-vlm", "--session-recording", "--capture"), ("simple-vlm", "--session-recording", ""),
        ):
            with self.subTest(sample=sample, recording=recording, flag=flag), \
                    tempfile.TemporaryDirectory() as directory:
                fails = sample == "blocked" or bool(recording and not flag)
                (Path(directory) / "sample.current").write_text("old-sample\n")
                result = self.shell('''
acquire_lock() { :; }
release_lock() { :; }
catalog() {
  echo "$*" >> "$STATE_DIR/calls"
  [[ "$2" != blocked ]] || { echo "insufficient GPU VRAM" >&2; return 1; }
  printf 'simple-vlm\tagent-samples/simple-vlm-example\tsimple_vlm_example\t%s\n' "$TEST_FLAG"
}
stop_locked() { echo stop >> "$STATE_DIR/calls"; }
refresh_public_ip() { PUBLIC_IP=127.0.0.1; }
configure_runtime() { :; }
nohup() { printf '%s\n' "$*" > "$STATE_DIR/runner-args"; }
pid_alive() { return 0; }
sleep() { :; }
start_sample "$TEST_SAMPLE" "$TEST_RECORDING"
wait
''', directory, TEST_SAMPLE=sample, TEST_RECORDING=recording, TEST_FLAG=flag)
                calls = (Path(directory) / "calls").read_text().splitlines()
                self.assertEqual(calls, [f"--resolve {sample}"] + ([] if fails else ["stop"]))
                if fails:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("insufficient GPU VRAM" if sample == "blocked" else "does not support", result.stderr)
                    self.assertFalse((Path(directory) / "runner-args").exists())
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = flag if recording else ""
                    args = (Path(directory) / "runner-args").read_text().strip()
                    suffix = "_run agent-samples/simple-vlm-example simple_vlm_example"
                    self.assertTrue(args.endswith(suffix + (" " + expected if expected else "")), args)
                    self.assertEqual((Path(directory) / "sample.recording").read_text().strip(), expected)

    def test_recording_flag_reads_cli_declarations_without_executing_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            self.assertEqual(compat.recording_flag(project), "")
            for flags, expected in (
                ((), ""), (("--capture",), "--capture"), (("--session-recording",), "--session-recording"),
                (("--capture", "--session-recording"), "--session-recording"),
            ):
                (project / "main.py").write_text(
                    '# --session-recording and --capture are not necessarily options.\n'
                    'raise RuntimeError("must not execute")\n'
                    + "".join(f'parser.add_argument("{flag}", action="store_true")\n' for flag in flags)
                )
                self.assertEqual(compat.recording_flag(project), expected)

    def test_saved_revision_survives_reboot(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "runtime-root").write_text(directory + "\n")
            result = self.shell("is_xr_ai() { return 0; }; resolve_source() { return 99; }; ensure_source", directory)
            self.assertEqual(result.returncode, 0, result.stderr)


class SecurityTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.state = Path(directory)
        self.enterContext(patch.multiple(
            server, create=True, state=self.state, app="/bin/echo", inventory_cache=(None, 0, {}),
        ))

    @contextmanager
    def serve(self, state, bind="", **env):
        settings = {"XR_AI_UI_BIND": bind, "XR_AI_UI_AUTH": "password", "XR_AI_TRUST_PROXY": "0"} | env
        with patch.dict(os.environ, settings):
            with server.create_server("/bin/echo", ROOT / "ui.html", state, 0) as listener:
                thread = threading.Thread(target=listener.serve_forever)
                thread.start()
                host = f"[{bind}]" if listener.address_family == socket.AF_INET6 else "127.0.0.1"
                url = f"http://{host}:{listener.server_address[1]}"
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

                def request(path="/", headers=None, data=None):
                    req = urllib.request.Request(url + path, headers=headers or {}, data=data)
                    try:
                        response = opener.open(req, timeout=2)
                    except urllib.error.HTTPError as exc:
                        response = exc
                    with response:
                        return response.code, response.read()

                try:
                    yield request, url
                finally:
                    listener.shutdown()
                    thread.join(timeout=5)

    def test_events_assets_use_authenticated_fixed_proxy(self):
        proxy = self.enterContext(patch.object(server, "proxy"))
        response = proxy.open.return_value.__enter__.return_value
        response.read.return_value = b"asset"
        response.headers = {"Content-Type": "text/javascript"}
        request = object.__new__(server.Handler)
        request.authenticated = MagicMock(return_value=True)
        request.send_bytes = MagicMock()
        request.send_json = MagicMock()
        for path in ("/app.js", "/style.css", "/api/events?after=2"):
            request.path = path
            request.do_GET()
            proxy.open.assert_called_with("http://127.0.0.1:8092" + path, timeout=1)
            request.send_bytes.assert_called_with(b"asset", content_type="text/javascript")
        proxy.reset_mock()
        request.path = "/private.key"
        request.do_GET()
        request.send_json.assert_called_with({"error": "not found"}, HTTPStatus.NOT_FOUND)
        proxy.open.assert_not_called()
        request.path = "/app.js"
        request.authenticated.return_value = False
        request.do_GET()
        proxy.open.assert_not_called()

    def test_only_sample_readiness_is_latched(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / "sample.pid").write_text(str(os.getpid()))
            (state / "sample.run").write_text("run1")
            (state / "sample.log").write_text("All processes ready\nAll processes ready\n")
            self.enterContext(patch.object(server, "state", state))
            status = server.process_status
            self.assertEqual(status()["state"], "starting")
            with (state / "sample.log").open("a") as stream:
                stream.write("XR_AI_SAMPLE_START\n")
            self.assertEqual(status()["state"], "starting")
            with (state / "sample.log").open("a") as stream:
                stream.write("All processes ready\n")
            self.assertEqual(status()["state"], "ready")
            (state / "sample.log").write_text("later output\n" * 10000)
            self.assertEqual(status()["state"], "ready")
            (state / "sample.run").write_text("run2")
            self.assertEqual(status()["state"], "starting")
            # Both startup chatter and delayed first polling can exceed the UI tail.
            for index, (before, after) in enumerate(((102000, 0), (0, 102000), (102000, 102000))):
                with self.subTest(before=before, after=after):
                    (state / "sample.run").write_text(f"large{index}")
                    (state / "sample.log").write_text(
                        "All processes ready\nXR_AI_SAMPLE_START\n"
                        + "x" * before + "\nAll processes ready\n" + "x" * after
                    )
                    self.assertEqual(status()["state"], "ready")

                    self.assertEqual((state / "sample.ready").read_text().strip(), f"large{index}")
            # Markers can straddle chunk boundaries; neither may be discarded.
            for index, prefix in enumerate((65530, 65500)):
                with self.subTest(prefix=prefix):
                    (state / "sample.run").write_text(f"split{index}")
                    (state / "sample.log").write_text(
                        "x" * prefix + "XR_AI_SAMPLE_START\nAll processes ready\n"
                    )
                    self.assertEqual(status()["state"], "ready")

            # Current stacks publish readiness without depending on log text or TTS.
            (state / "sample.run").write_text("structured1")
            (state / "sample.status-mode").write_text("structured")
            (state / "sample.log").write_text("XR_AI_SAMPLE_START\nAll processes ready\n")
            self.assertEqual(status()["state"], "starting")
            (state / "sample.ready").write_text("structured1\n")
            self.assertEqual(status()["state"], "ready")
            (state / "sample.ready").unlink()
            self.assertEqual(status()["state"], "starting")

    def test_inventory_cache_refreshes_on_revision_and_expiry(self):
        command = MagicMock(return_value=type("Result", (), {"returncode": 0, "stdout": '{"demos": []}'})())
        clock = MagicMock()
        clock.monotonic.return_value = 100
        revision = {"runtime-root": "one"}
        self.enterContext(patch.multiple(
            server, time=clock, read_first=lambda name: revision.get(name, ""),
            subprocess=MagicMock(run=command), Path=MagicMock(home=MagicMock(side_effect=OSError)),
        ))
        for _ in range(3):
            self.assertEqual(server.inventory()["demos"], [])
        self.assertEqual(command.call_count, 1)
        revision["runtime-root"] = "two"
        server.inventory()
        self.assertEqual(command.call_count, 2)
        clock.monotonic.return_value = 131
        server.inventory()
        self.assertEqual(command.call_count, 3)
    def test_authenticated_ui_and_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / "ui.password").write_text("x" * 40)
            (state / "ui.password").chmod(0o600)
            (state / "hf").write_text("hf_secret")
            (state / "sample.log").write_text("hf_secret\nAll processes ready\nAll processes ready\n")
            (state / "sample.pid").write_text(str(os.getpid()))
            (state / "sample.run").write_text("run1")
            auth = "Basic " + base64.b64encode(b"xrai:" + b"x" * 40).decode()
            with self.serve(state) as (request, _):
                self.assertEqual(request()[0], 401)
                self.assertEqual(request("/api/log")[0], 401)
                self.assertEqual(request("/api/releases")[0], 401)
                self.assertEqual(request(headers={"Authorization": auth})[0], 200)
                status, body = request("/api/log", {"Authorization": auth})
                self.assertEqual(status, 200)
                self.assertNotIn(b"hf_secret", body)
                self.assertIn(b"REDACTED", body)
                self.assertEqual(request("/ui.password", {"Authorization": auth})[0], 404)
                self.assertEqual(request("/api/stop", {"Authorization": auth}, b"")[0], 403)
                headers = {"Authorization": auth, "X-XR-AI-Request": "1", "Origin": "https://evil.invalid"}
                self.assertEqual(request("/api/stop", headers, b"")[0], 403)
                headers.pop("Origin")
                with patch.object(server.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, stdout='{"tags":["v0.3.0-beta"],"latest":"v0.3.0-beta"}', stderr="",
                )):
                    status, body = request("/api/releases", {"Authorization": auth})
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body)["latest"], "v0.3.0-beta")
                self.assertEqual(request("/api/stop", headers, b"x" * 4097)[0], 413)
                for data, expected in ((b"", b'"message": "stop"'), (b"forget=1", b'"message": "forget-token"')):
                    status, body = request("/api/stop", headers, data)
                    self.assertEqual(status, 200)
                    self.assertIn(expected, body)
                for data in (b"forget=", b"forget=0", b"forget=--help", b"forget=1&forget=1", b"forget=1&forget="):
                    with self.subTest(data=data):
                        status, body = request("/api/stop", headers, data)
                        self.assertEqual(status, 400)
                        self.assertIn(b"forget must be omitted or set once to 1", body)
                for data, expected in (
                    (b"", b"start simple-vlm"), (b"&recording=1", b"start simple-vlm --session-recording"),
                ):
                    status, body = request("/api/start", headers, b"sample=simple-vlm" + data)
                    self.assertEqual(status, 200)
                    self.assertIn(expected, body)
                for value in (b"", b"0", b"--capture", b"1&recording=1"):
                    status, _ = request("/api/start", headers, b"sample=simple-vlm&recording=" + value)
                    self.assertEqual(status, 400)

    def test_public_bind_requires_explicit_proxy_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["python3", str(ROOT / "server.py"), "/bin/false", str(ROOT / "ui.html"), directory, "8091"],
                capture_output=True,
                text=True,
                env=os.environ | {"XR_AI_UI_BIND": "0.0.0.0", "XR_AI_TRUST_PROXY": "0"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("HTTPS proxy", result.stderr)

    def test_recording_download_requires_auth_and_stays_inside_finalized_bundles(self):
        capture = self.state / "captures"
        complete = capture / "simple-vlm/session1"
        complete.mkdir(parents=True)
        (complete / "manifest.json").write_text(json.dumps({"complete": True, "participant_id": "headset"}))
        (complete / "video.mp4").write_bytes(b"recorded video")
        pending = capture / "simple-vlm/pending"
        pending.mkdir()
        (pending / "video.mp4").write_bytes(b"still recording")
        (self.state / "secret").write_text("private-key-material")
        (capture / "escape").symlink_to(self.state, target_is_directory=True)
        self.enterContext(patch.object(server.recordings, "ROOT", capture))
        with self.serve(self.state) as (request, _):
            self.assertEqual(request("/api/recordings")[0], 401)
            self.assertEqual(request("/api/recording?id=simple-vlm/session1")[0], 401)
            password = (self.state / "ui.password").read_text().strip()
            auth = {"Authorization": "Basic " + base64.b64encode(f"xrai:{password}".encode()).decode()}
            status, body = request("/api/recordings", auth)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["recordings"], [
                {"id": "simple-vlm/session1", "participant": "headset", "complete": True},
            ])
            status, data = request("/api/recording?id=simple-vlm/session1", auth)
            self.assertEqual(status, 200)
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                self.assertEqual(archive.read("simple-vlm/session1/video.mp4"), b"recorded video")
                self.assertNotIn("private-key-material", repr(archive.namelist()))
            for name in ("../secret", "/etc", "escape", "simple-vlm/pending"):
                self.assertEqual(request("/api/recording?id=" + name, auth)[0], 404)
            (complete / "private.key").symlink_to(self.state / "secret")
            self.assertEqual(request("/api/recording?id=simple-vlm/session1", auth)[0], 404)

    def test_proxy_mode_needs_explicit_trust_and_rejects_unknown_auth_modes(self):
        for mode, trust, message in (("proxy", "0", "authenticated HTTPS proxy"), ("none", "1", "password or proxy")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, str(ROOT / "server.py"), "/bin/echo", str(ROOT / "ui.html"), directory, "8091"],
                    capture_output=True, text=True,
                    env=os.environ | {"XR_AI_UI_BIND": "127.0.0.1", "XR_AI_UI_AUTH": mode, "XR_AI_TRUST_PROXY": trust},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse((Path(directory) / "ui.password").exists())

    def test_proxy_mode_skips_password_but_preserves_request_protection_and_redaction(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / "hf").write_text("hf_saved_secret")
            (state / "sample.log").write_text("Downloading with hf_saved_secret\n")
            with self.serve(state, XR_AI_UI_AUTH="proxy", XR_AI_TRUST_PROXY="1") as (request, url):
                self.assertEqual(request()[0], 200)
                self.assertFalse((state / "ui.password").exists())
                status, body = request("/api/log")
                self.assertEqual(status, 200)
                self.assertIn(b"REDACTED", body)
                self.assertNotIn(b"hf_saved_secret", body)
                for headers in ({}, {"X-XR-AI-Request": "1", "Origin": "https://evil.invalid"}):
                    self.assertEqual(request("/api/stop", headers, b"")[0], 403)
                status, body = request("/api/stop", {"X-XR-AI-Request": "1", "Origin": url}, b"")
                self.assertEqual(status, 200)
                self.assertIn(b'"message": "stop"', body)

    @unittest.skipUnless(socket.has_ipv6, "IPv6 is unavailable")
    def test_ipv6_loopback_listener_starts_and_requires_authentication(self):
        with socket.socket(socket.AF_INET6) as probe:
            try:
                probe.bind(("::1", 0))
            except OSError as exc:
                self.skipTest(f"IPv6 loopback is unavailable: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            with self.serve(directory, bind="::1") as (request, _):
                self.assertEqual(request()[0], 401)
                password = (Path(directory) / "ui.password").read_text().strip()
                auth = "Basic " + base64.b64encode(f"xrai:{password}".encode()).decode()
                status, body = request(headers={"Authorization": auth})
                self.assertEqual(status, 200)
                self.assertIn(b"XR AI Experiences", body)

    def test_password_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / "secret").write_text("x" * 40)
            (state / "ui.password").symlink_to(state / "secret")
            result = subprocess.run(
                ["python3", str(ROOT / "server.py"), "/bin/false", str(ROOT / "ui.html"), directory, "8091"],
                capture_output=True,
                text=True,
                env=os.environ | {"XR_AI_UI_BIND": "127.0.0.1"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("private regular file", result.stderr)
