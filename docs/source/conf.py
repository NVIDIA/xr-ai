# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration for the XR AI versioned documentation site."""
from __future__ import annotations

import os

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
    "substitution",
]
myst_heading_anchors = 3
myst_substitutions = {
    # Historical pages must link to their matching source tree, not to main.
    "github_ref": os.environ.get("SPHINX_MULTIVERSION_NAME", "main"),
}

# Don't choke the build on the bundled long-form changelog or build output.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Keep release documentation immutable and publish main as development docs.
smv_tag_whitelist = r"^v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
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
