# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration for the XR AI versioned documentation site."""
from __future__ import annotations

import os
import re

_GITHUB_BLOB_PREFIX = "https://github.com/NVIDIA/xr-ai/blob/"
_GITHUB_TREE_PREFIX = "https://github.com/NVIDIA/xr-ai/tree/"
_SEMVER_IDENTIFIER = r"(?:0|[1-9]\d*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
_SEMVER_TAG = (
    rf"v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    rf"(?:-{_SEMVER_IDENTIFIER}(?:\.{_SEMVER_IDENTIFIER})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_GITHUB_REF_PATTERN = re.compile(rf"(?:main|[0-9a-f]{{40}}|{_SEMVER_TAG})$")


def _github_ref() -> str:
    ref = os.environ.get("SPHINX_MULTIVERSION_NAME") or os.environ.get(
        "XR_AI_DOCS_GITHUB_REF", "main"
    )
    if not _GITHUB_REF_PATTERN.fullmatch(ref):
        raise ValueError(f"unsupported documentation GitHub ref: {ref!r}")
    return ref


def _rewrite_github_links(_app, _docname: str, source: list[str]) -> None:
    ref = _github_ref()
    source[0] = source[0].replace(
        f"{_GITHUB_BLOB_PREFIX}main/", f"{_GITHUB_BLOB_PREFIX}{ref}/"
    )
    source[0] = source[0].replace(
        f"{_GITHUB_TREE_PREFIX}main/", f"{_GITHUB_TREE_PREFIX}{ref}/"
    )

# -- Project information -----------------------------------------------------
project = "XR AI"
copyright = "2026, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx_multiversion",
]

# MyST Markdown is the page format (matches the repo's existing docs/*.md).
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
]
myst_heading_anchors = 3

# Don't choke the build on the bundled long-form changelog or build output.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Keep release documentation immutable and publish main as development docs.
smv_tag_whitelist = rf"^{_SEMVER_TAG}$"
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = None
smv_released_pattern = r"^tags/.*$"
smv_latest_version = os.environ.get("XR_AI_DOCS_LATEST_VERSION", "main")
smv_outputdir_format = "{ref.name}"

# -- HTML output -------------------------------------------------------------
html_theme = "nvidia_sphinx_theme"
html_show_sphinx = False
html_title = "XR AI"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# The theme reserves navbar centre for pydata's JSON-driven switcher, which this
# site does not configure; sphinx-multiversion drives the sidebar one instead.
html_theme_options = {"navbar_center": ["navbar-external-links"]}
html_sidebars = {"**": ["versioning.html", "sidebar-nav-bs"]}


def setup(app):
    app.connect("source-read", _rewrite_github_links)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
