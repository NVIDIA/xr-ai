// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit.config

/**
 * Configures camera capture for a [StreamSession].
 *
 * Mirror of Swift `CameraConfig` / web `CameraConfig`.
 * Resolution and frame-rate are intentionally omitted: LiveKit and the
 * hardware negotiate the best supported format automatically.
 *
 * ## Presets
 * ```kotlin
 * CameraConfig.DEFAULT   // enabled, back-facing (primary camera on Android)
 * CameraConfig.FRONT     // enabled, front-facing (selfie camera)
 * CameraConfig.DISABLED  // camera off
 * ```
 */
data class CameraConfig(
    val enabled: Boolean = true,
    val facing: CameraFacing = CameraFacing.BACK,
    /**
     * Optional Camera2 camera id (e.g. `"0"`, `"1"`). When non-null, the
     * backend pins capture to that exact camera and ignores [facing]. When
     * null, the backend picks any camera matching [facing].
     *
     * Use this to choose between multiple cameras on the same side
     * (e.g. wide vs. ultra-wide vs. telephoto on the back).
     */
    val deviceId: String? = null,
) {

    /**
     * Camera facing direction.
     *
     * Mirror of Swift `CameraConfig.Position` and web `CameraFacing`.
     */
    enum class CameraFacing {
        /** Front-facing (selfie) camera. */
        FRONT,

        /** Rear-facing (primary) camera. */
        BACK,
    }

    companion object {
        /** Camera enabled, rear-facing — natural default on Android. */
        @JvmField val DEFAULT = CameraConfig(enabled = true, facing = CameraFacing.BACK)

        /** Camera enabled, front-facing (selfie). */
        @JvmField val FRONT = CameraConfig(enabled = true, facing = CameraFacing.FRONT)

        /** Camera disabled — nothing is captured or published. */
        @JvmField val DISABLED = CameraConfig(enabled = false, facing = CameraFacing.BACK)
    }
}
