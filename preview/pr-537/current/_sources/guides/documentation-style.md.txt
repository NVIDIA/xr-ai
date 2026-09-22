<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Documentation style

This reference is the authoritative house style for customer-facing Markdown
and reStructuredText. Match the surrounding document and keep edits scoped.
When maintainers adopt a repository-wide convention, update this reference in
the same change. Document scoped exceptions in the affected content.

## Cross-references

Introduce documentation links with **Refer to** or **refer to**, not "See" or
"see." Prefer MyST `{doc}` links in Markdown and `:doc:` or `:ref:` roles in
reStructuredText. Use a descriptive Markdown link when the target is source
code or an external resource, and link directly to the authoritative target
rather than through a thin redirect page. Treat published URL fragments as
stable. When rewording a heading, preserve its previous fragment with an
explicit compatibility anchor.

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
