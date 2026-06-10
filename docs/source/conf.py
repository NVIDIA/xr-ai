# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration for the XR-AI documentation site.

Mirrors the NVIDIA/IsaacTeleop docs setup: the NVIDIA Sphinx theme + MyST so
the existing Markdown under ``docs/`` is reused directly. Single-version for
now; ``sphinx-multiversion`` is a deliberate follow-up (see docs changelog).
"""
from __future__ import annotations

# -- Project information -----------------------------------------------------
project = "XR-AI"
copyright = "2026, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
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

# Don't choke the build on the bundled long-form changelog or build output.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "nvidia_sphinx_theme"
html_show_sphinx = False
html_title = "XR-AI"
