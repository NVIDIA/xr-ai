<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Documentation style

This reference is the authoritative house style for customer-facing Markdown
and reStructuredText. Match the surrounding document and keep edits scoped.
When maintainers adopt a repository-wide convention, update this reference in
the same change. Document scoped exceptions in the affected content.

## Opening overviews

Long reader-facing narrative pages open with an **Overview** section suited
to their audience and purpose. For capabilities and architecture, explain
what the developer can build and why the capability helps. Use concrete
behavior rather than generic claims about what an assistant needs.

Frame maintained samples as references for a broader class of applications.
Identify the reusable pattern, then explain how the sample demonstrates it.
Base generalizations on the implementation and its extension points; keep
sample-specific behavior distinct from what a developer must adapt.

For policies, references, prerequisites, troubleshooting, and migration notes,
state the scope, applicable setup or versions, and the task the reader can
complete. These openings do not need an application-building rationale.
Deprecated samples lead with their status and replacement. Keep all openings
concise; leave API lists, instructions, and technical detail in the body.

Fold existing introductory context into the overview and remove repeated
framing afterward. Follow it with the first useful example, procedure, or
design detail. Preserve unique technical guidance in the appropriate section.
Short navigation pages and redirects do not need an overview. Generated API
pages are also exempt; judge catalog pages by their rendered length.

## Cross-references

Introduce documentation links with **Refer to** or **refer to**, not "See" or
"see." Prefer MyST `{doc}` links in Markdown and `:doc:` or `:ref:` roles in
reStructuredText. Use a descriptive Markdown link when the target is source
code or an external resource, and link directly to the authoritative target
rather than through a thin redirect page. Treat published URL fragments as
stable. When rewording a heading, preserve its previous fragment with an
explicit compatibility anchor.

## Migration guides

Refer to {ref}`migration-release-rollover` for entry scope and release rollover.

## Slashes in prose

Do not use `/` to mean "or" or "and." Rewrite with the conjunction or a list.
Keep established technical forms such as `I/O`, `iOS/visionOS`, `STUN/TURN`,
`VR/AR`, `LLM/VLM`, `pub/sub`, and `and/or`. Apply these prose rules to
explanatory comments and diagram labels inside fenced examples. Keep executable
code, literal commands and output, identifiers, UI text, URLs, paths, and
endpoints unchanged.

## Terminology

Prefer full words over informal shortenings in paraphrased prose: for example,
use **certificate**, not **cert**. Preserve shortenings that are part of literal
UI text, logs, filenames, paths, endpoints, identifiers, commands, or
executable code.

## Clarity and consistency

Use direct word order and specific self-reference nouns such as "This sample,"
"This workflow," or "This reference" instead of "This page" or "This guide."
Keep article usage consistent within a procedure. Hyphenate compound adjectives
when they precede a noun.

## Requirement strength

Use **must** for requirements, an imperative for instructions, and explicit
softer wording such as **can** or **recommended** for guidance. Avoid ambiguous
**should** wording when the intended strength can be stated directly.

## Punctuation and headings

Use a colon to introduce an explanation or list. Use the document's established
em-dash style for an aside; do not use a narrative `--`. Keep code markup out
of headings when it makes the table of contents or navigation harder to scan.
In reStructuredText, make title underlines span the full title.
