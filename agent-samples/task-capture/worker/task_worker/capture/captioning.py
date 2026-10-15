# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare image-caption requests and parse the VLM's structured responses.

Build instructions for an initial image or an earlier/current image pair, with
previous activity and phase names as context. Parse responses into the activity,
phase, caption, and delta fields used by capturing.py and eval/caption.py.

Validate responses before saving them so malformed output cannot enter the
caption history. Format validation does not establish visual accuracy.
"""

from __future__ import annotations

import json


def parse_caption(text: str) -> dict[str, str]:
    """Parse the VLM's JSON into activity, phase, caption, and delta text.

    Reject missing, extra, or empty fields before the response enters caption
    history. Trim each value and retain at most 1,000 characters per field.
    """
    candidate = text.strip()
    # Tolerate a Markdown wrapper, but never extract observations from reasoning
    # or arbitrary prose (including an unfinished <think> response).
    if candidate.startswith("```json\n") and candidate.endswith("```"):
        candidate = candidate[8:-3].strip()
    elif candidate.startswith("```\n") and candidate.endswith("```"):
        candidate = candidate[4:-3].strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("caption response was not a JSON object")
    fields = {"activity", "phase", "caption", "delta"}
    if set(value) != fields or any(not isinstance(v, str) or not v.strip() for v in value.values()):
        raise ValueError("caption requires exactly four nonempty string fields: activity, phase, caption, delta")
    return {key: value[key].strip()[:1000] for key in fields}


def caption_query(previous: dict[str, str] | None) -> str:
    """Build the caption request for an initial image or a chronological pair.

    State which image is current and which is the comparison reference, then
    supply previous activity and phase names only as grouping hints.
    """
    query = (
        "You received exactly TWO images, in chronological order: FIRST=earlier, SECOND=current. "
        "Describe the SECOND image in caption. In delta use: Earlier: ... Current: ... Change: ... . "
        "Do not use Initial observation: there are two images. Then identify the broad parent activity "
        "and the specific current phase from the visual evidence."
        if previous is not None else
        "You received exactly ONE image: the current initial view. Describe that image in caption; "
        "set delta to Initial observation. Identify a supported broad activity and current phase/state."
    )
    if previous is not None:
        query += "\nPrevious grouping labels (fallible naming hints, not visual evidence): " + json.dumps({
            "activity": previous["activity"], "phase": previous["phase"],
        })
    return query
