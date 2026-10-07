<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Speaker STT

Private local inference process for phrase-enrolled speaker transcription.
It shares Nemotron 3 Diarization and Multitalker Parakeet weights across voice
workers. Refer to the [AI services guide](https://nvidia.github.io/xr-ai/latest/components/ai-services.html#speaker-conditioned-stt)
for configuration and startup instructions.

Its private IPC contract accepts serialized, capture-ordered PCM. Session
creation supplies the first sample's capture timestamp; later event timing is
derived from accepted sample counts rather than packet arrival times.
