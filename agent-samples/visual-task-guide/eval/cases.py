# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployed-model cases with distinctive details absent from prompt text."""

GUIDE_CASES = (
    {
        "name": "rag_hand_presentation_answer",
        "question": "How should I position my hands so the camera can count seven reliably?",
        "observation": "Two hands overlap near the edge of the frame.",
        "required_terms": ("side by side",),
        "max_words": 30,
        "requires_knowledge": True,
    },
    {
        "name": "latest_observation_answer",
        "question": "How many fingers does the latest observation report?",
        "observation": "Seven extended fingers are clearly visible across two separated hands.",
        "required_terms": ("seven",),
        "max_words": 30,
        "requires_knowledge": False,
    },
)

VLM_CASES = (
    {
        "name": "two_finger_fixture",
        "fixture": "two-extended-fingers.jpg",
        "question": "Apply the configured finger-count contract to this image.",
        "required_terms": ("count=2", "confidence="),
    },
    {
        "name": "closed_fist_fixture",
        "fixture": "closed-fist.jpg",
        "question": "Apply the configured finger-count contract to this image.",
        "required_terms": ("count=0", "confidence="),
    },
)

LEAKAGE_MARKERS = (
    "camera can count seven reliably",
    "seven extended fingers",
    "closed-fist.jpg",
    "two-extended-fingers.jpg",
)
