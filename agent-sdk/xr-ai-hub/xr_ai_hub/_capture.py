# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private media-capture topics shared by voice workers and DeviceIOHub."""

CAPTURE_PUBLISH_PREFIX = b"capture_data."
CAPTURE_START_TOPIC = "_capture.control.start"
CAPTURE_STOP_TOPIC = "_capture.control.stop"
CAPTURE_OBSERVATION_TOPIC = "_capture.observation"
CAPTURE_STT_TOPIC = "_capture.voice.stt"
CAPTURE_TTS_TOPIC = "_capture.voice.tts"
CAPTURE_TOPICS = frozenset((
    CAPTURE_START_TOPIC,
    CAPTURE_STOP_TOPIC,
    CAPTURE_OBSERVATION_TOPIC,
    CAPTURE_STT_TOPIC,
    CAPTURE_TTS_TOPIC,
))
