# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal Sphinx makefile. Usage:
#   pip install -r requirements.txt
#   make html             # build the checked-out revision
#   make versions         # build main and release tags
#   make strict           # strict checked-out revision build (PR gate)
#   make strict-versions  # strict multiversion build (deployment gate)

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXMULTIVERSION ?= sphinx-multiversion
SOURCEDIR     = source
BUILDDIR      = _build
CURRENTBUILDDIR = $(BUILDDIR)/current
DOCS_GITHUB_REF ?= $(shell git rev-parse HEAD)
LATEST_RELEASE := $(shell git tag --list 'v*' | python3 ../.github/scripts/select_latest_docs_release.py)
CURRENTENV    = XR_AI_DOCS_GITHUB_REF=$(DOCS_GITHUB_REF)
SMVENV        = XR_AI_DOCS_LATEST_VERSION=$(or $(LATEST_RELEASE),main)

.PHONY: help html versions strict strict-versions latest-release clean
help:
	@echo "Targets: html, versions, strict, strict-versions, latest-release, clean."

html:
	@$(CURRENTENV) $(SPHINXBUILD) -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(CURRENTBUILDDIR)"

versions:
	@$(SMVENV) $(SPHINXMULTIVERSION) $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)"

strict:
	@$(CURRENTENV) $(SPHINXBUILD) -W --keep-going -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(CURRENTBUILDDIR)"

strict-versions:
	@$(SMVENV) $(SPHINXMULTIVERSION) -W --keep-going $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)"

latest-release:
	@printf '%s\n' "$(LATEST_RELEASE)"

clean:
	rm -rf "$(BUILDDIR)"
