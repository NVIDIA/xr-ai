# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static sample configuration discovery and Sphinx rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_CONFIG_SUFFIXES = {".json", ".yaml", ".yml"}


@dataclass(frozen=True)
class SampleConfig:
    """One checked-in sample configuration file."""

    sample: str
    path: Path
    content: str

    @property
    def language(self) -> str:
        """Return the Pygments language for this configuration."""

        return "json" if self.path.suffix == ".json" else "yaml"


def load_config_catalog(repository_root: Path) -> tuple[SampleConfig, ...]:
    """Discover conventional config sources owned by installed top-level samples."""

    configs: list[SampleConfig] = []
    samples_dir = repository_root / "agent-samples"
    for project_path in sorted(samples_dir.glob("*/pyproject.toml")):
        sample_dir = project_path.parent
        yaml_dir = sample_dir / "yaml"
        candidates = {path for path in yaml_dir.rglob("*") if path.is_file() and path.suffix in _CONFIG_SUFFIXES}
        for capability_project in sample_dir.glob("*/pyproject.toml"):
            candidates.update(
                path
                for path in capability_project.parent.iterdir()
                if path.is_file() and path.suffix in _CONFIG_SUFFIXES
            )
        for path in sorted(candidates):
            if path.is_symlink() or not path.resolve().is_relative_to(sample_dir.resolve()):
                raise ValueError(f"{path}: sample configurations must not be symbolic links")
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                raise ValueError(f"{path}: sample configuration is empty")
            configs.append(
                SampleConfig(
                    sample=sample_dir.name,
                    path=path.relative_to(repository_root),
                    content=content,
                )
            )
    return tuple(configs)


def _directive_type():
    from docutils import nodes
    from docutils.parsers.rst import Directive

    class ConfigReferenceDirective(Directive):
        has_content = False

        def run(self):
            repository_root = Path(self.state.document.settings.env.srcdir).parents[1]
            container = nodes.container(classes=["xr-ai-config-reference"])
            by_sample: dict[str, list[SampleConfig]] = {}
            for config in load_config_catalog(repository_root):
                by_sample.setdefault(config.sample, []).append(config)

            for sample, configs in by_sample.items():
                sample_section = nodes.section(ids=[nodes.make_id(f"config-{sample}")])
                sample_section += nodes.title(text=sample)
                for config in configs:
                    relative_path = config.path.relative_to(Path("agent-samples") / sample)
                    file_section = nodes.section(ids=[nodes.make_id(f"config-{sample}-{relative_path.as_posix()}")])
                    file_section += nodes.title(text=relative_path.as_posix())
                    literal = nodes.literal_block(config.content, config.content)
                    literal["language"] = config.language
                    file_section += literal
                    sample_section += file_section
                container += sample_section
            return [container]

    return ConfigReferenceDirective


def setup(app):
    """Register the static sample configuration directive with Sphinx."""

    app.add_directive("xr-ai-config-reference", _directive_type())
    return {"parallel_read_safe": True, "parallel_write_safe": True}
