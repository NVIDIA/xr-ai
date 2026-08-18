# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for Magpie Multilingual served by NVIDIA Speech TTS NIM."""

MAGPIE_TTS_NIM = {
    "category": "tts",
    "kind": "nvidia_tts_nim",
    "timeout": 60.0,
    "language_code": "en-US",
    "voice": "Magpie-Multilingual.EN-US.Mia",
    "sample_rate": 22050,
}
