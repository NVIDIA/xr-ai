# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Lua discovery, validation, and header insertion through the CLI."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/check_spdx_headers.py"
_SPEC = importlib.util.spec_from_file_location("check_spdx_headers", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_CHECKER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CHECKER
_SPEC.loader.exec_module(_CHECKER)


@pytest.mark.parametrize("shebang", ["", "#!/usr/bin/env luajit\n"])
def test_lua_missing_header_is_detected_and_fixed(tmp_path, monkeypatch, capsys, shebang):
    monkeypatch.setattr(_CHECKER, "_REPO_ROOT", tmp_path)
    source = tmp_path / "scene.lua"
    body = shebang + "print('scene')\n"
    source.write_text(body)

    assert _CHECKER.main([]) == 1
    assert "scene.lua" in capsys.readouterr().err
    assert _CHECKER.main(["--fix", str(source)]) == 1
    capsys.readouterr()
    text = source.read_text()
    assert text.startswith(shebang + "-- SPDX-FileCopyrightText:")
    assert "\n-- SPDX-License-Identifier: Apache-2.0\n\n" in text
    assert text.endswith("print('scene')\n")
    assert _CHECKER.main([str(source)]) == 0
    assert "OK: 1 file(s)" in capsys.readouterr().out
    assert _CHECKER.main(["--fix", str(source)]) == 0
    assert source.read_text() == text


@pytest.mark.parametrize("line", [0, 1])
def test_lua_rejects_wrong_comment_marker(tmp_path, line):
    source = tmp_path / "scene.lua"
    header = _CHECKER._build_header(source, "dash").splitlines(keepends=True)
    header[line] = header[line].replace("--", "#", 1)
    source.write_text("".join(header) + "print('scene')\n")
    ok, reason = _CHECKER.check(source)
    assert not ok
    assert "must start with '--'" in reason


@pytest.mark.parametrize("shebang", ["", "#!/usr/bin/env luajit\n"])
@pytest.mark.parametrize("license_id", ["MIT", "Apache-2.0"])
def test_lua_header_allows_spdx_in_body(tmp_path, capsys, shebang, license_id):
    source = tmp_path / "scene.lua"
    text = (
        shebang
        + _CHECKER._build_header(source, "dash")
        + f'local s = "SPDX-License-Identifier: {license_id}"\n'
    )
    source.write_text(text)

    assert _CHECKER.main([str(source)]) == 0
    for _ in range(2):
        assert _CHECKER.main(["--fix", str(source)]) == 0
        assert source.read_text() == text
    capsys.readouterr()


@pytest.mark.parametrize("line", [0, 1])
def test_lua_header_declaration_in_code_is_detected_and_fixed(tmp_path, capsys, line):
    source = tmp_path / "scene.lua"
    header = _CHECKER._build_header(source, "dash").splitlines(keepends=True)
    header[line] = f'local s = "{header[line].removeprefix("-- ").strip()}"\n'
    body = "".join(header) + "print('scene')\n"
    source.write_text(body)

    assert _CHECKER.main([str(source)]) == 1
    assert _CHECKER.main(["--fix", str(source)]) == 1
    fixed = source.read_text()
    assert fixed.endswith(body)
    assert _CHECKER.main([str(source)]) == 0
    assert _CHECKER.main(["--fix", str(source)]) == 0
    assert source.read_text() == fixed
    capsys.readouterr()


def test_lua_block_header_fix_converges(tmp_path, capsys):
    source = tmp_path / "scene.lua"
    body = (
        "--[[\n"
        + _CHECKER._build_header(source, "dash").replace("-- ", "")
        + "]]\nprint('scene')\n"
    )
    source.write_text(body)

    assert _CHECKER.main(["--fix", str(source)]) == 1
    fixed = source.read_text()
    assert fixed.endswith(body)
    assert _CHECKER.main([str(source)]) == 0
    assert _CHECKER.main(["--fix", str(source)]) == 0
    assert source.read_text() == fixed
    capsys.readouterr()
