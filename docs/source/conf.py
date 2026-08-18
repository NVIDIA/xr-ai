# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration for the XR AI versioned documentation site."""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from runpy import run_path

sys.path.insert(0, str(Path(__file__).parent))

_API_CONTRACT = run_path(str(Path(__file__).with_name("_api_contract.py")))
API_PACKAGE_DIRS = _API_CONTRACT["API_PACKAGE_DIRS"]
PUBLIC_API_MODULES = _API_CONTRACT["PUBLIC_API_MODULES"]
_PUBLIC_PACKAGE_EXPORTS = {
    package_dir.name: frozenset(
        _API_CONTRACT["_literal_exports"](
            ast.parse(
                (package_dir / "__init__.py").read_text(encoding="utf-8"),
                filename=str(package_dir / "__init__.py"),
            ),
            package_dir / "__init__.py",
        )
    )
    for package_dir in API_PACKAGE_DIRS
}
_PRIVATE_TYPE_REFERENCE = re.compile(
    rf"\b(?P<package>{'|'.join(map(re.escape, _PUBLIC_PACKAGE_EXPORTS))})"
    r"\._[A-Za-z0-9_.]*\.(?P<name>[A-Za-z][A-Za-z0-9_]*)"
)

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


def _canonicalize_public_type_references(value):
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        package = match.group("package")
        name = match.group("name")
        if name in _PUBLIC_PACKAGE_EXPORTS[package]:
            return f"{package}.{name}"
        return match.group(0)

    return _PRIVATE_TYPE_REFERENCE.sub(replace, value)


def _prepare_api_templates(environment) -> None:
    environment.finalize = _canonicalize_public_type_references


def _canonicalize_api_annotations(_app, env, _docnames) -> None:
    for module_annotations in getattr(env, "autoapi_annotations", {}).values():
        for name, value in module_annotations.items():
            module_annotations[name] = _canonicalize_public_type_references(value)
    for api_object in getattr(env, "autoapi_all_objects", {}).values():
        for attribute in ("annotation", "args", "return_annotation"):
            value = getattr(api_object, attribute, None)
            if isinstance(value, str):
                try:
                    setattr(
                        api_object,
                        attribute,
                        _canonicalize_public_type_references(value),
                    )
                except AttributeError:
                    pass

# -- Project information -----------------------------------------------------
project = "XR AI"
copyright = "2026, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------
extensions = [
    "_cli_reference",
    "_config_reference",
    "autoapi.extension",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx_multiversion",
]

# Build the public Python reference from source without importing SDK packages.
# This keeps optional voice, model, and native dependencies out of the docs
# environment. Package ``__all__`` declarations remain the publication boundary.
autoapi_type = "python"
autoapi_dirs = [str(path) for path in API_PACKAGE_DIRS]
autoapi_root = "reference/python"
autoapi_add_toctree_entry = True
autoapi_keep_files = False
autoapi_options = [
    "members",
    "show-module-summary",
    "imported-members",
]
autoapi_member_order = "bysource"
autoapi_python_class_content = "class"
autoapi_prepare_jinja_env = _prepare_api_templates

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

# Exclude generated documentation output.
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
html_extra_path = ["_redirects"]
html_css_files = ["css/custom.css"]

# Named ``versioning.html`` so it does not shadow the theme's own
# ``version-switcher.html`` navbar component, which stays inert unconfigured.
html_sidebars = {"**": ["versioning.html", "sidebar-nav-bs"]}


_PRIVATE_API_MEMBER_SUFFIXES = (
    ".HubVoiceTransport.input",
    ".HubVoiceTransport.output",
)


def _skip_private_api_details(_app, what, name, _obj, _skip, _options):
    """Hide private modules and transport-adapter accessors."""

    if what == "module" and name not in PUBLIC_API_MODULES:
        return True
    if name.endswith(_PRIVATE_API_MEMBER_SUFFIXES):
        return True
    return None


def setup(app):
    app.connect("env-before-read-docs", _canonicalize_api_annotations)
    app.connect("source-read", _rewrite_github_links)
    app.connect("autoapi-skip-member", _skip_private_api_details)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
