// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit.backends.livekit

import com.nvidia.xrai.streamkitsample.streamkit.config.CameraEncodingConfig
import io.livekit.android.room.participant.VideoTrackPublishDefaults
import io.livekit.android.room.track.CustomVideoPreset
import io.livekit.android.room.track.VideoPreset169
import io.livekit.android.room.track.VideoPreset43
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** SDK-only first-publication mapping; reduce lower budgets together when needed. */
internal fun cameraLayerDefaults(
    encoding: CameraEncodingConfig?, defaults: VideoTrackPublishDefaults, width: Int, height: Int,
): VideoTrackPublishDefaults {
    if (encoding == null || !defaults.simulcast) return defaults
    require(width > 0 && height > 0) { "Camera dimensions are not resolved" }
    val aspect = max(width, height).toDouble() / min(width, height)
    val layers = defaults.simulcastLayers ?: if (abs(aspect - 16.0 / 9.0) < abs(aspect - 4.0 / 3.0)) {
        listOf(VideoPreset169.H180, VideoPreset169.H360)
    } else {
        listOf(VideoPreset43.H180, VideoPreset43.H360)
    }
    val budgets = layers.map { it.encoding.maxBitrate }
    val largest = budgets.maxOrNull() ?: 0
    val reduced = largest > encoding.maxBitrateBps
    val scaled = if (reduced) budgets.map { budget ->
        require(budget > 0) { "Simulcast bitrate limits must be positive integers" }
        (budget.toLong() * encoding.maxBitrateBps / largest).toInt()
    } else budgets
    require(!reduced || (scaled.all { it > 0 } && budgets.indices.all { i ->
        budgets.indices.all { j -> budgets[i] == budgets[j] || scaled[i] != scaled[j] }
    })) { "Primary bitrate cannot preserve the SDK lower-layer budget ordering" }
    return defaults.copy(simulcastLayers = layers.mapIndexed { i, layer ->
        CustomVideoPreset(layer.capture, layer.encoding.copy(
            maxBitrate = scaled[i],
        ))
    })
}
