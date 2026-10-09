<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Magpie NIM speech

The `magpie_nim_tts` HTTP wrapper has been removed. Configure a `tts` entry with
the existing `riva_grpc` adapter to reach the native speech endpoint directly.
The SDK normalizes streamed PCM and buffered WAV for the voice worker.

Refer to {doc}`/guides/deploying-with-nim` for endpoint configuration and voice
behavior, and {doc}`migrations` for the removed service.
