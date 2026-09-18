// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit

/** A participant-scoped request from the remote agent to capture one image. */
data class ImageCaptureRequest(
    val requestId: String,
    val timeoutMs: Long,
)

/** Encoded still image returned by an application's opt-in capture handler. */
data class CapturedImage(
    val data: ByteArray,
    val mimeType: String = "image/jpeg",
    val name: String = "camera.jpg",
)
