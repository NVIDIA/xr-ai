// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit

/** Application metadata for a client-to-agent file transfer. */
data class FileSendOptions(
    val topic: String,
    val name: String? = null,
    val mimeType: String? = null,
    val attributes: Map<String, String> = emptyMap(),
)

/** Identifies a file transfer completed by the local transport. */
data class FileTransferInfo(
    val id: String,
    val topic: String,
    val name: String,
    val mimeType: String,
    val size: Long,
)
