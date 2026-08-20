# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generated sample configuration discovery."""

from __future__ import annotations

from pathlib import Path
from runpy import run_path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_REFERENCE = run_path(str(_ROOT / "docs" / "source" / "_config_reference.py"))
load_config_catalog = _CONFIG_REFERENCE["load_config_catalog"]


def test_config_catalog_covers_every_top_level_sample() -> None:
    configs = load_config_catalog(_ROOT)
    projects = {path.parent.name for path in (_ROOT / "agent-samples").glob("*/pyproject.toml")}

    assert {config.sample for config in configs} == projects
    assert len({config.path for config in configs}) == len(configs)
    assert [config.path for config in configs] == sorted(config.path for config in configs)

    expected: set[Path] = set()
    for project_path in (_ROOT / "agent-samples").glob("*/pyproject.toml"):
        sample_dir = project_path.parent
        expected.update(
            path.relative_to(_ROOT)
            for path in sample_dir.joinpath("yaml").rglob("*")
            if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}
        )
        for capability_project in sample_dir.glob("*/pyproject.toml"):
            expected.update(
                path.relative_to(_ROOT)
                for path in capability_project.parent.iterdir()
                if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}
            )

    assert {config.path for config in configs} == expected


def test_config_catalog_preserves_source_and_language() -> None:
    configs = {config.path.as_posix(): config for config in load_config_catalog(_ROOT)}

    worker = configs["agent-samples/simple-vlm-example/yaml/simple_vlm_example_worker.yaml"]
    assert worker.language == "yaml"
    assert "# Frame freshness" in worker.content
    assert "frame_max_age_s: 5.0" in worker.content

    models = configs["agent-samples/simple-vlm-example/yaml/models.local.json"]
    assert models.language == "json"
    assert '"ownership": "managed"' in models.content

    scene = configs["agent-samples/xr-render-demo/scene/scene_service.yaml"]
    assert "lovr_bin" in scene.content

    for path in (
        "agent-samples/simple-vlm-example/yaml/device_io_hub.yaml",
        "agent-samples/xr-render-demo/yaml/device_io_hub.yaml",
    ):
        assert "# Development-only placeholders." in configs[path].content

    for config in configs.values():
        assert config.content == (_ROOT / config.path).read_text(encoding="utf-8")


def test_config_catalog_rejects_symbolic_links(tmp_path: Path) -> None:
    sample = tmp_path / "agent-samples" / "demo"
    yaml_dir = sample / "yaml"
    yaml_dir.mkdir(parents=True)
    sample.joinpath("pyproject.toml").write_text("[project]\nname = 'demo'\n")
    outside = tmp_path / "outside.yaml"
    outside.write_text("value: unsafe\n", encoding="utf-8")
    yaml_dir.joinpath("linked.yaml").symlink_to(outside)

    with pytest.raises(ValueError, match="symbolic links"):
        load_config_catalog(tmp_path)
