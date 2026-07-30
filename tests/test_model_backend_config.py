# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-role ``model_backend`` key: the orchestrators'
regex read and process-row selection (sample main.py) and the workers'
per-role models composition (worker models_select.py). Pure config
coverage — no docker or GPU."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The samples are not packages — import their modules straight off disk.
simple_main = _load(
    _REPO_ROOT / "agent-samples/simple-vlm-example/main.py", "simple_vlm_main")
render_main = _load(
    _REPO_ROOT / "agent-samples/xr-render-demo/main.py", "xr_render_main")
simple_select = _load(
    _REPO_ROOT / "agent-samples/simple-vlm-example/worker/models_select.py",
    "simple_vlm_models_select")
render_select = _load(
    _REPO_ROOT / "agent-samples/xr-render-demo/worker/models_select.py",
    "xr_render_models_select")


def _backends(mod, tmp_path: Path, monkeypatch, text: str) -> dict[str, str]:
    cfg = tmp_path / mod._WORKER_CONFIG
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(text)
    monkeypatch.setattr(mod, "_BASE", tmp_path)
    return mod._model_backends()


# ── orchestrator: regex parse of model_backend ──────────────────────────────


class TestModelBackendsParse:
    def test_missing_file_defaults_local(self, tmp_path, monkeypatch):
        monkeypatch.setattr(simple_main, "_BASE", tmp_path)
        assert simple_main._model_backends() == {
            "stt": "local", "tts": "local", "vlm": "local"}

    def test_missing_key_defaults_local(self, tmp_path, monkeypatch):
        got = _backends(simple_main, tmp_path, monkeypatch, "models_yaml: models.yaml\n")
        assert got == {"stt": "local", "tts": "local", "vlm": "local"}

    def test_scalar_local(self, tmp_path, monkeypatch):
        got = _backends(simple_main, tmp_path, monkeypatch, "model_backend: local\n")
        assert set(got.values()) == {"local"}

    def test_scalar_nim_keeps_speech_local(self, tmp_path, monkeypatch):
        # Legacy meaning of the bare scalar: hosted chat models, local speech.
        got = _backends(render_main, tmp_path, monkeypatch, "model_backend: nim\n")
        assert got == {"stt": "local", "tts": "local", "llm": "nim", "vlm": "nim"}
        got = _backends(simple_main, tmp_path, monkeypatch, "model_backend: nim\n")
        assert got == {"stt": "local", "tts": "local", "vlm": "nim"}

    def test_map_default_nim_hosts_every_role(self, tmp_path, monkeypatch):
        # Unlike the scalar, an explicit map default applies to speech too.
        got = _backends(render_main, tmp_path, monkeypatch,
                        "model_backend:\n"
                        "  default: nim\n")
        assert got == {"stt": "nim", "tts": "nim", "llm": "nim", "vlm": "nim"}

    def test_scalar_quoted_with_comment(self, tmp_path, monkeypatch):
        got = _backends(simple_main, tmp_path, monkeypatch,
                        'model_backend: "nim_local"  # all self-hosted\n')
        assert set(got.values()) == {"nim_local"}

    def test_map_with_default(self, tmp_path, monkeypatch):
        got = _backends(simple_main, tmp_path, monkeypatch,
                        "model_backend:\n"
                        "  default: nim\n"
                        "  stt: local\n")
        assert got == {"stt": "local", "tts": "nim", "vlm": "nim"}

    def test_map_without_default_unlisted_roles_local(self, tmp_path, monkeypatch):
        got = _backends(render_main, tmp_path, monkeypatch,
                        "model_backend:\n"
                        "  llm: nim_local\n")
        assert got == {"stt": "local", "tts": "local",
                       "llm": "nim_local", "vlm": "local"}

    def test_map_unknown_role_key_ignored(self, tmp_path, monkeypatch):
        got = _backends(simple_main, tmp_path, monkeypatch,
                        "model_backend:\n"
                        "  vlm: nim\n"
                        "  reranker: nim\n")
        assert got == {"stt": "local", "tts": "local", "vlm": "nim"}

    def test_map_quoted_values_and_comments(self, tmp_path, monkeypatch):
        got = _backends(simple_main, tmp_path, monkeypatch,
                        "model_backend:\n"
                        "  # speech stays local\n"
                        "  vlm: 'nim_local'  # container\n"
                        '  tts: "nim"\n')
        assert got == {"stt": "local", "tts": "nim", "vlm": "nim_local"}

    def test_map_stops_at_next_top_level_key(self, tmp_path, monkeypatch):
        got = _backends(simple_main, tmp_path, monkeypatch,
                        "model_backend:\n"
                        "  vlm: nim\n"
                        "models_yaml: models.yaml\n"
                        "  stt: nim\n")
        assert got["stt"] == "local"
        assert got["vlm"] == "nim"

    def test_unknown_backend_value_exits(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit):
            _backends(simple_main, tmp_path, monkeypatch, "model_backend: cloud\n")


# ── orchestrator: process-row selection ─────────────────────────────────────


def _rows(mod, backends: dict[str, str]) -> list[tuple[str, str]]:
    return [(p.name, p.command) for p in mod._build_processes(backends)]


class TestSimpleVlmBuildProcesses:
    _LOCAL = {"stt": "local", "tts": "local", "vlm": "local"}

    def test_all_local(self):
        rows = _rows(simple_main, dict(self._LOCAL))
        assert ("vlm", "vlm_server") in rows
        assert ("stt", "stt_server") in rows
        assert ("tts", "piper_tts_server") in rows
        assert not [r for r in rows if r[1] == "nim_server"]

    def test_vlm_nim_local_swaps_in_container_row(self):
        rows = _rows(simple_main, {**self._LOCAL, "vlm": "nim_local"})
        assert ("vlm", "vlm_server") not in rows
        assert rows.count(("vlm", "nim_server")) == 1
        assert ("stt", "stt_server") in rows

    def test_vlm_nim_launches_neither(self):
        rows = _rows(simple_main, {**self._LOCAL, "vlm": "nim"})
        assert not [r for r in rows if r[0] == "vlm"]

    def test_scalar_nim_launches_local_speech_only(self, tmp_path, monkeypatch):
        # Bare `model_backend: nim` end to end: local stt/tts rows, no vlm row.
        rows = _rows(simple_main, _backends(
            simple_main, tmp_path, monkeypatch, "model_backend: nim\n"))
        assert ("stt", "stt_server") in rows
        assert ("tts", "piper_tts_server") in rows
        assert not [r for r in rows if r[0] == "vlm"]
        assert not [r for r in rows if r[1] == "nim_server"]

    def test_all_nim_local(self):
        rows = _rows(simple_main, {r: "nim_local" for r in self._LOCAL})
        nim = [r for r in rows if r[1] == "nim_server"]
        assert sorted(n for n, _ in nim) == ["stt", "tts", "vlm"]
        for cmd in ("vlm_server", "stt_server", "piper_tts_server"):
            assert cmd not in [c for _, c in rows]


class TestRenderBuildProcesses:
    _LOCAL = {"stt": "local", "tts": "local", "llm": "local", "vlm": "local"}

    def test_all_local(self):
        rows = _rows(render_main, dict(self._LOCAL))
        assert ("llm", "llama_nemotron_llm_server") in rows
        assert ("agent-llm", "nemotron3_nano_llm_server") in rows
        assert ("vlm", "vlm_server") in rows
        assert ("stt", "stt_server") in rows
        assert ("tts", "piper_tts_server") in rows
        assert not [r for r in rows if r[1] == "nim_server"]

    def test_llm_nim_local_replaces_both_llm_rows(self):
        rows = _rows(render_main, {**self._LOCAL, "llm": "nim_local"})
        assert ("llm", "llama_nemotron_llm_server") not in rows
        assert ("agent-llm", "nemotron3_nano_llm_server") not in rows
        assert rows.count(("llm", "nim_server")) == 1

    def test_llm_nim_launches_neither(self):
        rows = _rows(render_main, {**self._LOCAL, "llm": "nim"})
        assert not [r for r in rows if r[0] in ("llm", "agent-llm")]

    def test_scalar_nim_launches_local_speech_only(self, tmp_path, monkeypatch):
        # Bare `model_backend: nim` end to end: local stt/tts rows, no
        # llm/agent-llm/vlm rows.
        rows = _rows(render_main, _backends(
            render_main, tmp_path, monkeypatch, "model_backend: nim\n"))
        assert ("stt", "stt_server") in rows
        assert ("tts", "piper_tts_server") in rows
        assert not [r for r in rows if r[0] in ("llm", "agent-llm", "vlm")]
        assert not [r for r in rows if r[1] == "nim_server"]

    def test_tts_nim_local_swaps_in_container_row(self):
        rows = _rows(render_main, {**self._LOCAL, "tts": "nim_local"})
        assert ("tts", "piper_tts_server") not in rows
        assert rows.count(("tts", "nim_server")) == 1


# ── worker: model_backends parse + mis-indent warning ───────────────────────


class TestWorkerModelBackends:
    def test_scalar_and_map(self):
        assert simple_select.model_backends({}) == {
            "stt": "local", "tts": "local", "vlm": "local"}
        got = render_select.model_backends(
            {"model_backend": {"default": "nim", "stt": "local"}})
        assert got == {"stt": "local", "tts": "nim", "llm": "nim", "vlm": "nim"}

    def test_scalar_nim_keeps_speech_local(self):
        assert simple_select.model_backends({"model_backend": "nim"}) == {
            "stt": "local", "tts": "local", "vlm": "nim"}
        assert render_select.model_backends({"model_backend": "nim"}) == {
            "stt": "local", "tts": "local", "llm": "nim", "vlm": "nim"}

    def test_unknown_backend_value_raises(self):
        with pytest.raises(ValueError, match="cloud"):
            simple_select.model_backends({"model_backend": {"vlm": "cloud"}})

    def test_non_string_non_map_raises(self):
        with pytest.raises(ValueError, match="string or a mapping"):
            simple_select.model_backends({"model_backend": ["nim"]})

    def test_top_level_role_key_warns_about_misindent(self):
        records: list[str] = []
        sink = logger.add(lambda m: records.append(str(m)), level="WARNING")
        try:
            simple_select.model_backends({"model_backend": "local", "stt": "nim"})
        finally:
            logger.remove(sink)
        assert any("mis-indented" in r for r in records)


# ── worker: per-role models composition ─────────────────────────────────────


def _write_models(path: Path, tag: str, entries: tuple[str, ...]) -> None:
    body = []
    for name in entries:
        if name in ("stt", "tts"):
            body.append(f"{name}:\n  category: {name}\n  base_url: http://{tag}-{name}\n")
        else:
            body.append(
                f"{name}:\n  category: {'vlm' if name == 'vlm' else 'llm'}\n"
                f"  base_url: http://{tag}-{name}\n  model_name: {tag}-{name}\n")
    path.write_text("".join(body))


@pytest.fixture
def simple_yaml_dir(tmp_path: Path) -> Path:
    entries = ("stt", "tts", "vlm")
    _write_models(tmp_path / "models.yaml", "local", entries)
    _write_models(tmp_path / "models.nim.yaml", "nim", entries)
    _write_models(tmp_path / "models.nim_local.yaml", "nimlocal", entries)
    return tmp_path


class TestComposeModelsConfig:
    def _compose(self, yaml_dir: Path, backend_cfg) -> object:
        cfg = {"model_backend": backend_cfg}
        return simple_select.compose_models_config(cfg, yaml_dir / "worker.yaml")

    def test_each_role_reads_its_backend_file(self, simple_yaml_dir):
        models = self._compose(
            simple_yaml_dir, {"default": "nim", "stt": "local", "vlm": "nim_local"})
        assert models.stt("stt").base_url == "http://local-stt"
        assert models.tts("tts").base_url == "http://nim-tts"
        assert models.vlm("vlm").base_url == "http://nimlocal-vlm"

    def test_models_yaml_key_names_the_local_file(self, simple_yaml_dir):
        _write_models(simple_yaml_dir / "models.omni.yaml", "omni", ("stt", "tts", "vlm"))
        cfg = {"model_backend": "local", "models_yaml": "models.omni.yaml"}
        models = simple_select.compose_models_config(cfg, simple_yaml_dir / "worker.yaml")
        assert models.vlm("vlm").base_url == "http://omni-vlm"

    def test_scalar_nim_pulls_speech_from_local_file(self, simple_yaml_dir):
        models = self._compose(simple_yaml_dir, "nim")
        assert models.stt("stt").base_url == "http://local-stt"
        assert models.tts("tts").base_url == "http://local-tts"
        assert models.vlm("vlm").base_url == "http://nim-vlm"

    def test_missing_file_ok_when_no_role_uses_it(self, simple_yaml_dir):
        (simple_yaml_dir / "models.nim_local.yaml").unlink()
        models = self._compose(simple_yaml_dir, "nim")
        assert models.vlm("vlm").base_url == "http://nim-vlm"

    def test_missing_file_fails_when_a_role_uses_it(self, simple_yaml_dir):
        (simple_yaml_dir / "models.nim_local.yaml").unlink()
        with pytest.raises(ValueError, match="models.nim_local.yaml"):
            self._compose(simple_yaml_dir, {"vlm": "nim_local"})

    def test_missing_entry_in_chosen_file_fails(self, simple_yaml_dir):
        _write_models(simple_yaml_dir / "models.nim.yaml", "nim", ("stt", "tts"))
        with pytest.raises(ValueError, match="no 'vlm' entry"):
            self._compose(simple_yaml_dir, {"vlm": "nim"})

    def test_unknown_backend_value_fails(self, simple_yaml_dir):
        with pytest.raises(ValueError, match="expected one of"):
            self._compose(simple_yaml_dir, {"vlm": "cloud"})

    def test_each_file_loaded_at_most_once(self, simple_yaml_dir, monkeypatch):
        calls: list[str] = []
        real = simple_select.yaml.safe_load
        monkeypatch.setattr(
            simple_select.yaml, "safe_load",
            lambda text: (calls.append("x"), real(text))[1])
        self._compose(simple_yaml_dir, "nim_local")
        assert len(calls) == 1

    def test_render_llm_role_maps_both_llm_entries(self, tmp_path):
        entries = ("stt", "tts", "llm", "agent_llm", "vlm")
        _write_models(tmp_path / "models.yaml", "local", entries)
        _write_models(tmp_path / "models.nim.yaml", "nim", entries)
        cfg = {"model_backend": {"llm": "nim"}}
        models = render_select.compose_models_config(cfg, tmp_path / "worker.yaml")
        assert models.llm("llm").base_url == "http://nim-llm"
        assert models.llm("agent_llm").base_url == "http://nim-agent_llm"
        assert models.vlm("vlm").base_url == "http://local-vlm"


# ── shipped worker YAMLs still declare a valid backend ──────────────────────


@pytest.mark.parametrize("mod", [simple_main, render_main],
                         ids=["simple-vlm-example", "xr-render-demo"])
def test_shipped_worker_yaml_parses(mod) -> None:
    backends = mod._model_backends()
    assert set(backends.values()) <= {"local", "nim", "nim_local"}
