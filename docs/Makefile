# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal Sphinx makefile. Usage:
#   pip install -r requirements.txt
#   make html        # build main and release tags into _build/
#   make strict      # build with warnings-as-errors (CI gate)

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-multiversion
SOURCEDIR     = source
BUILDDIR      = _build
LATEST_RELEASE := $(shell git tag --list 'v*' | python3 ../.github/scripts/select_latest_docs_release.py)
SMVENV        = XR_AI_DOCS_LATEST_VERSION=$(or $(LATEST_RELEASE),main)

.PHONY: help html strict latest-release clean
help:
	@echo "Targets: html (versioned build), strict (warnings as errors), clean."

html:
	@$(SMVENV) $(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

strict:
	@$(SMVENV) $(SPHINXBUILD) -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

latest-release:
	@printf '%s\n' "$(LATEST_RELEASE)"

clean:
	rm -rf "$(BUILDDIR)"
